import torch
import torch.nn as nn
from utils import ResidualGRU
from utils import S6

#85.56
class Simple(nn.Module):
    def __init__(self, dropout_rate=0.7):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(100, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 100),
            nn.LayerNorm(100),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            S6(seq_len=63, d_model=100, state_size=300,)


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

        #self.rnn_text = ResidualGRU(input_size=100, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        
        self.classifier1 = nn.Linear(100, 50)  
        self.classifier2 = nn.Linear(50, 2)

    def forward(self, text, audio=None, video=None, mask=None):
        text_embeddings = self.text_branch(text)
        audio_embeddings = self.audio_branch(audio)
        video_embeddings = self.video_branch(video)
        #text_output = self.rnn_text(text_embeddings)
        text_output = text_embeddings
        o1 = self.classifier1(text_output)
        o2 = self.classifier2(o1)

        return o2