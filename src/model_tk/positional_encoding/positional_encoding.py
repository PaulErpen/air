import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dimensions, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Compute the positional encodings in advance
        pe = torch.zeros(max_seq_len, dimensions)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, dimensions, 2).float() * (-math.log(10000.0) / dimensions))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
