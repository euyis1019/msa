import torch
import torch.nn as nn
from torchview import draw_graph
from utils import BiModalAttention, TriModalAttention, SelfAttention, ResidualGRU, ResidualAttention
from utils import Config, S6
import os

config = Config()
config.learning_rate = 0.0001
config.batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiModalModel(nn.Module):
    def __init__(self, dropout_rate=0.7):
        super(MultiModalModel, self).__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(300, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.audio_branch = nn.Sequential(
            nn.Linear(74, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.video_branch = nn.Sequential(
            nn.Linear(35, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.rnn_text = ResidualGRU(input_size=300, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_audio = ResidualGRU(input_size=300, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_video = ResidualGRU(input_size=300, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.s6_text = S6(seq_len=98, d_model=300, state_size=300, device=device)
        self.s6_audio = S6(seq_len=98, d_model=100, state_size=300, device=device)
        self.s6_video = S6(seq_len=98, d_model=100, state_size=300, device=device)
        self.bi_modal_attention = ResidualAttention(BiModalAttention(size=600))
        self.tri_modal_attention = ResidualAttention(TriModalAttention(feature_size=300))
        self.self_attention = ResidualAttention(SelfAttention(size=600))
        self.early_fuse = nn.Sequential(
            nn.Linear(900, 50),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1800, 600),
            nn.LeakyReLU(negative_slope=0.01),
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

    def forward(self, text, audio, video, mask=None):
        text_embeddings = self.text_branch(text)
        audio_embeddings = self.audio_branch(audio)
        video_embeddings = self.video_branch(video)
        text_output = self.rnn_text(text_embeddings)
        audio_output = self.rnn_audio(audio_embeddings)
        video_output = self.rnn_video(video_embeddings)
        bi_modal_output = self.bi_modal_attention(text_output, audio_output)
        tri_modal_output = self.tri_modal_attention(text_output, audio_output, video_output)
        self_attention_output = self.self_attention(text_output)
        combined_features = torch.cat((bi_modal_output, tri_modal_output, self_attention_output), dim=2)
        logits = self.output_layer(combined_features)
        return logits

model = MultiModalModel(dropout_rate=config.dropout_rate).to(device)

text_input = torch.randn(1, 98, 300).to(device)
audio_input = torch.randn(1, 98, 74).to(device)
video_input = torch.randn(1, 98, 35).to(device)

model_graph = draw_graph(model, input_size=[text_input.shape, audio_input.shape, video_input.shape], expand_nested=True, save_graph=True, filename="torchview", directory=".", format="svg")
