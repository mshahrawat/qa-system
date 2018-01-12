import torch
import torch.nn.functional as F
from torch.autograd import Variable

def batch_cos_sim(batch, is_cuda):
    batch_size = batch.size(0) / 22
    if batch_size < 1:
        batch_size = 1
    samples = torch.chunk(batch, batch_size)
    cos_sims = []
    for sample in samples:
        cos_sims.append(cos_sim(sample))

    cos_sims = torch.cat(cos_sims, dim=0)
    if is_cuda:
        pos_idxs = Variable(torch.cuda.LongTensor([0] * batch_size))
    else:
        pos_idxs = Variable(torch.LongTensor([0] * batch_size))
    return cos_sims, pos_idxs

def cos_sim(inp):
    s_query_candidates = F.cosine_similarity(inp[0].unsqueeze(0).expand_as(inp[1:]), inp[1:]).unsqueeze(0)
    return s_query_candidates
