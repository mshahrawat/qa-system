import torch
import metrics
from load_data import *
from load_android_eval import *
import itertools
from torch.autograd import Variable
from lstm import LSTM
from tqdm import tqdm
import maxmarginloss
import meter

def eval(dataloader, model, is_cuda, auc=False):
    if auc:
        m = meter.AUCMeter()
    num_samples = 0
    mrr_total = 0
    p1 = 0
    p5 = 0
    map_total = 0
    if is_cuda:
        eps = Variable(torch.cuda.DoubleTensor([.001]))
    else:
        eps = Variable(torch.DoubleTensor([.001]))

    for title_batch, body_batch, title_mask, body_mask in tqdm(dataloader):
        if is_cuda:
            title_batch = torch.cat(title_batch.cuda(), 0)
            body_batch = torch.cat(body_batch.cuda(), 0)
            title_mask = torch.cat(title_mask.cuda(), 0).double()
            body_mask = torch.cat(body_mask.cuda(), 0).double()
        else:
            title_batch = torch.cat(title_batch, 0)
            body_batch = torch.cat(body_batch, 0)
            title_mask = torch.cat(title_mask, 0).double()
            body_mask = torch.cat(body_mask, 0).double()

        title_inputs = title_batch.permute(2, 0, 1)
        body_inputs = body_batch.permute(2, 0, 1)
        title_mask = title_mask.permute(1, 0)
        body_mask = body_mask.permute(1, 0)

        title_inputs = Variable(title_inputs)
        body_inputs = Variable(body_inputs)
        title_mask = Variable(title_mask)
        body_mask = Variable(body_mask)

        titles_hidden = model(title_inputs)
        bodies_hidden = model(body_inputs)

        titles_encoded = titles_hidden * title_mask.unsqueeze(2).expand_as(titles_hidden)
        bodies_encoded = bodies_hidden * body_mask.unsqueeze(2).expand_as(bodies_hidden)

        titles_encoded = torch.sum(titles_encoded, dim=0)
        bodies_encoded = torch.sum(bodies_encoded, dim=0)

        titles_encoded = titles_encoded / (torch.sum(title_mask, keepdim=True, dim=0).permute(1, 0).expand_as(titles_encoded) + eps)
        bodies_encoded = bodies_encoded / (torch.sum(body_mask, keepdim=True, dim=0).permute(1, 0).expand_as(bodies_encoded) + eps)

        if is_cuda:
            qs_encoded = (titles_encoded + bodies_encoded) / Variable(torch.cuda.DoubleTensor([2]))
        else:
            qs_encoded = (titles_encoded + bodies_encoded) / Variable(torch.DoubleTensor([2]))

        if auc:
            cos_sims, y = maxmarginloss.batch_cos_sim(qs_encoded)
            others = cos_sims[1:].data
            labels = torch.zeros(others.size()[0])
            labels[0] = 1
            m.add(others, labels)
        else:
            q = qs_encoded[0].data
            others = qs_encoded[1:].data
            labels = torch.zeros(others.size()[0])
            if labels.size()[0] > 20:
                labels[:labels.size()[0] - 20] = 1
            mrr_total += metrics.mrr(q, others, labels)
            map_total += metrics.map(q, others, labels, labels.size()[0] - 20)
            p1 += metrics.p_at_n(1, q, others, labels)
            p5 += metrics.p_at_n(5, q, others, labels)
        num_samples += 1

    if auc:
        return m.value(max_fpr=.05)
    return map_total / num_samples, mrr_total / num_samples, p1 / num_samples, p5 / num_samples

if __name__ == '__main__':
    model = LSTM(200, 240, 5)
    model.double()
    model.load_state_dict(torch.load('./models/lstm_1_1p_', 
        map_location=lambda storage, loc: storage))
    is_cuda = False
    is_auc = False

    with open("data/part1/dev_dataloader", "rb") as f:
        dataloader = pickle.load(f)

    if is_auc:
        auc = eval(dataloader, model, is_cuda, is_auc)
        print "AUC: ", auc
    else:
        mp, mrr, p1, p5 = eval(dataloader, model, is_cuda, is_auc)
        print "MAP", mp
        print "MRR: ", mrr
        print "P@1: ", p1
        print "P@5: ", p5

