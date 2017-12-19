import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1, pad=0):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        hidden = self.conv(x)
        hidden = self.dropout(hidden)
        output = self.tanh(hidden)
        return output