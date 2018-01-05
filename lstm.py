import torch
from torch.autograd import Variable
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, kernel_size, is_bidirectional=False, is_cuda=False):
        super(LSTM, self).__init__()
        self.is_bidirectional = is_bidirectional
        self.is_cuda = is_cuda
        self.hidden_dim = hidden_dim
        if is_bidirectional:
            self.lstm = nn.LSTM(embedding_dim, self.hidden_dim//2, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden = self.init_hidden(1)
        self.dropout = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        if self.is_bidirectional:
            if self.is_cuda:
                h = Variable(torch.zeros(1, batch_size, self.hidden_dim//2)).double().cuda()
            else:
                h = Variable(torch.zeros(1, batch_size, self.hidden_dim//2)).double()
            return (h, h)
        else:
            if self.is_cuda:
                h = Variable(torch.zeros(1, batch_size, self.hidden_dim)).double().cuda()
            else:
                h = Variable(torch.zeros(1, batch_size, self.hidden_dim)).double()
            return (h, h)

    def forward(self, x, mask):
        # mask = (100 seq, 22 * batch_size)
        # x = (100 seq, 22 * batch_size, 200 hid_dim)
        self.hidden = self.init_hidden(x.size(1))
        output = self.dropout(x)
        output, self.hidden = self.lstm(output, self.hidden)

        # apply mask
        output = output * mask.unsqueeze(2).expand_as(output)
        # output = (100 seq, 22 * batch_size, 240 hid_dim)

        # mean pooling
        # output = torch.mean(output, dim=0) (without mask)
        # average over the word and divide by the actual length
        output = torch.sum(output, dim=0)
        # output = (22 * batch_size, 240 hid_dim)

        if self.is_cuda:
            eps = Variable(torch.cuda.DoubleTensor([.001]))
        else:
            eps = Variable(torch.DoubleTensor([.001]))
        output = output / (torch.sum(mask, keepdim=True, dim=0).permute(1, 0).expand_as(output) + eps)
        # output = (22 * batch_size, 240 hid_dim)

        output = self.dropout(output)
        return output
