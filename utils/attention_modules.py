import torch
import torch.nn as nn
import torch.nn.functional as F

class BiModalAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        m1 = torch.bmm(x, y.transpose(1, 2))
        m2 = torch.bmm(y, x.transpose(1, 2))
        n1 = self.softmax(m1)
        n2 = self.softmax(m2)
        o1 = torch.bmm(n1, y)
        o2 = torch.bmm(n2, x)
        a1 = o1 * x
        a2 = o2 * y
        return torch.cat((a1, a2), dim=-1)

class TriModalAttention(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.tanh = nn.Tanh()
        self.dense_tv = nn.Linear(feature_size * 2, feature_size)
        self.dense_ta = nn.Linear(feature_size * 2, feature_size)
        self.dense_av = nn.Linear(feature_size * 2, feature_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, v, t, a):
        Ftv = self.tanh(self.dense_tv(torch.cat((t, v), dim=2)))
        Fta = self.tanh(self.dense_ta(torch.cat((t, a), dim=2)))
        Fav = self.tanh(self.dense_av(torch.cat((a, v), dim=2)))
        c1 = torch.bmm(a, Ftv.transpose(1, 2))
        c2 = torch.bmm(v, Fta.transpose(1, 2))
        c3 = torch.bmm(t, Fav.transpose(1, 2))
        p1 = self.softmax(c1)
        p2 = self.softmax(c2)
        p3 = self.softmax(c3)
        t1 = torch.bmm(p1, a)
        t2 = torch.bmm(p2, v)
        t3 = torch.bmm(p3, t)
        Oatv = t1 * Ftv
        Ovta = t2 * Fta
        Otav = t3 * Fav
        return torch.cat((Oatv, Ovta, Otav), dim=2)

class SelfAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m = torch.bmm(x, x.transpose(1, 2))
        n = self.softmax(m)
        o = torch.bmm(n, x)
        a = o * x
        return a

class ResidualGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        if input_size == 73:
            input_size = 100
        self.projection = nn.Linear(2 * hidden_size, input_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.projection(out)
        return F.relu(x + out)

class ResidualAttention(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module

    def forward(self, *inputs):
        attention_output = self.attention_module(*inputs)
        batch_size, seq_len, features = inputs[0].shape
        _, _, attention_features = attention_output.shape
        if attention_features % features == 0:
            scale_factor = attention_features // features
            expanded_input = inputs[0].unsqueeze(-1).expand(batch_size, seq_len, features, scale_factor).reshape(batch_size, seq_len, attention_features)
        else:
            raise ValueError("attention_output的最后一维不是inputs[0]的整数倍，无法对齐")
        output = expanded_input + attention_output
        return F.relu(output)
