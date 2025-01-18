
import torch
import torch.nn as nn
from utils import BiModalAttention, TriModalAttention, SelfAttention, ResidualGRU, ResidualAttention
from utils import S6

class Mamba(nn.Module):
    def __init__(self, dropout_rate=0):
        super().__init__()
        self.S6_text_branch = nn.Sequential(
            nn.Linear(100, 128),
            nn.LayerNorm(128),
            #nn.Dropout(dropout_rate),

            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 100),
            nn.LayerNorm(100),
            nn.LeakyReLU(negative_slope=0.01),
            S6(seq_len=63, d_model=100, state_size=300,)
        )
        self.S6_audio_branch = nn.Sequential(
            nn.Linear(73, 256),
            nn.LayerNorm(256),
            #nn.Dropout(dropout_rate),

            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.LeakyReLU(negative_slope=0.01),
            S6(seq_len=63, d_model=100, state_size=300,)
        )
        self.S6_video_branch = nn.Sequential(
            nn.Linear(100, 256),
            nn.LayerNorm(256),
            #nn.Dropout(dropout_rate),

            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.LeakyReLU(negative_slope=0.01),
            S6(seq_len=63, d_model=100, state_size=300,)
        )
        # self.early_fuse = nn.Sequential(
        #     nn.Linear(300, 50),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Dropout(dropout_rate),
        # )
        self.S6 = S6(seq_len=63, d_model=100, state_size=300,)
        self.bi_modal_attention = ResidualAttention(BiModalAttention(size=600))
        self.tri_modal_attention = ResidualAttention(TriModalAttention(feature_size=100))
        self.self_attention = ResidualAttention(SelfAttention(size=600))
        self.fusionBroadcast = ModalityFusionModule()
        self.output_layer = nn.Sequential(
            nn.Linear(600, 300),            
            #S6(seq_len=63, d_model=300, state_size=128,), #75.86

            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 300),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.Dropout(dropout_rate),
            nn.Linear(300, 50),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(50, 2),
        )
        self.classifier = nn.Linear(100, 2)

    def forward(self, text, audio, video, mask=None):
        text_output = self.S6_text_branch(text)
        audio_output = self.S6_audio_branch(audio)
        video_output = self.S6_video_branch(video)
        # print(text_output.shape)
        # print(audio_output.shape)
        # print(video_output.shape )
        #audio_output = self.fusionBroadcast(audio_output, video_output, text_output)
        # text_output = self.S6(text_output)
        # audio_output = self.S6(audio_output)
        # video_output = self.S6(video_output)
        bi_modal_output = self.bi_modal_attention(text_output, audio_output)
        tri_modal_output = self.tri_modal_attention(text_output, audio_output, video_output)
        self_attention_output = self.self_attention(text_output)

        combined_features = torch.cat((bi_modal_output, tri_modal_output, self_attention_output), dim=2)

        logits = self.output_layer(combined_features)

        return logits



class ModalityFusionModule(nn.Module):
    def __init__(self, audio_dim=73, video_dim=100, text_dim=100, hidden_dim=100):
        super(ModalityFusionModule, self).__init__()
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.audio_attention = nn.Linear(hidden_dim, 1)
        self.video_attention = nn.Linear(video_dim, 1)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, audio_feature, video_feature, text_feature):
        # 将audio特征映射到hidden_dim维度
        audio_hidden = self.audio_linear(audio_feature)  # [batch_size, seq_len, hidden_dim]

        # 计算audio-text注意力权重
        audio_text_attn = self.audio_attention(torch.tanh(audio_hidden + self.text_linear(text_feature)))  # [batch_size, seq_len, 1]
        audio_text_attn = self.softmax(audio_text_attn)

        # 计算video-text注意力权重
        video_text_attn = self.video_attention(video_feature)  # [batch_size, seq_len, 1]
        video_text_attn = self.softmax(video_text_attn)

        # 融合注意力权重
        fusion_attn = audio_text_attn + video_text_attn  # [batch_size, seq_len, 1]

        # 应用注意力权重到audio特征上
        guided_audio_feature = fusion_attn * audio_hidden  # [batch_size, seq_len, hidden_dim]

        return guided_audio_feature