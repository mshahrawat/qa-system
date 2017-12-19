import torch
import torch.nn.functional as F
from torch.autograd import Variable

def batch_cos_sim(batch):
    print batch.size()
    batch_size = batch.size(0) / 22
    print batch_size
    if batch_size < 1:
        batch_size = 1
    samples = torch.chunk(batch, batch_size)
    cos_sims = []
    for sample in samples:
        cos_sims.append(cos_sim(sample))
    cos_sims = torch.cat(cos_sims, dim=0)
    # pos_idxs = Variable(torch.cuda.LongTensor([0] * batch_size))
    pos_idxs = Variable(torch.LongTensor([0] * batch_size))
    return cos_sims, pos_idxs

def cos_sim(input):
    s_query_candidates = F.cosine_similarity(input[0].unsqueeze(0).expand_as(input[1:]), input[1:])
    return s_query_candidates

class OneSampleMaxMarginLoss(torch.nn.Module):
    def __init__(self):
        # C is a constant singleton Variable - the margin (eg. Variable(torch.Tensor([0.1])) )
        super(OneSampleMaxMarginLoss,self).__init__()

    def forward(self, input, target):
        # can only use Functional stuff
        s_query_candidates = F.cosine_similarity(input[0].unsqueeze(0).expand_as(input[2:]), input[2:])
        s_query_pos_sample = F.cosine_similarity(input[0], input[1], dim=0)
            # .expand_as(s_query_candidates)
        C = Variable(torch.DoubleTensor([0.001]))
        s = s_query_candidates - s_query_pos_sample + C
        return torch.max(s)
