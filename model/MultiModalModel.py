import torch
import torch.nn as nn
from utils import BiModalAttention, TriModalAttention, SelfAttention, ResidualGRU, ResidualAttention
from utils import S6
#85.56
class MultiModalModel(nn.Module):
    def __init__(self, dropout_rate=0.7):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(100, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 100),
            nn.LayerNorm(100),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.audio_branch = nn.Sequential(
            nn.Linear(73, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.video_branch = nn.Sequential(
            nn.Linear(100, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.rnn_text = ResidualGRU(input_size=100, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_audio = ResidualGRU(input_size=100, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_video = ResidualGRU(input_size=100, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        
        self.early_fuse = nn.Sequential(
            nn.Linear(300, 50),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )
        self.bi_modal_attention = ResidualAttention(BiModalAttention(size=600))
        self.tri_modal_attention = ResidualAttention(TriModalAttention(feature_size=100))
        self.self_attention = ResidualAttention(SelfAttention(size=600))

        self.output_layer = nn.Sequential(
            nn.Linear(600, 300),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 300),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 50),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(50, 2),
        )
        self.classifier = nn.Linear(100, 2)
    
    def forward(self, text, audio, video, mask=None):
        text_embeddings = self.text_branch(text)
        audio_embeddings = self.audio_branch(audio)
        video_embeddings = self.video_branch(video)
        
        text_output = self.rnn_text(text_embeddings)
        audio_output = self.rnn_audio(audio_embeddings)
        video_output = self.rnn_video(video_embeddings)
        
        early_output = self.early_fuse(torch.cat((text_output, audio_output, video_output), dim=2))

        bi_modal_output = self.bi_modal_attention(text_output, audio_output)
        tri_modal_output = self.tri_modal_attention(text_output, audio_output, video_output)
        self_attention_output = self.self_attention(text_output)

        combined_features = torch.cat((bi_modal_output, tri_modal_output, self_attention_output), dim=2)
        
        logits = self.output_layer(combined_features)

        return logits
