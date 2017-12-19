import torch.nn as nn
import torch.nn.functional as F
import torch

class DAN(nn.Module):

    def __init__(self, embed_dim):
        super(DAN, self).__init__()
        self.W_hidden = nn.Linear(embed_dim, 200)
        self.W_out = nn.Linear(200, 1)

    def forward(self, all_x):
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu( self.W_hidden(avg_x) )
        out = self.W_out(hidden)
        return out