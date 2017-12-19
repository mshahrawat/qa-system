import torch
import metrics
from load_data import *
from load_android_eval import *
import itertools
from torch.autograd import Variable
from cnn import ConvNet
from tqdm import tqdm
import maxmarginloss
import meter

def eval(dataloader, net, is_cuda, auc=False):
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
    for  title, body, title_mask, body_mask in dataloader:
        # transform to correct dims
        if is_cuda:
            title = torch.cat(title.cuda(), 0)
            body = torch.cat(body.cuda(), 0)
            title_mask = torch.cat(title_mask.cuda(), 0).double()
            body_mask = torch.cat(body_mask.cuda(), 0).double()
        else:
            title = torch.cat(title, 0)
            body = torch.cat(body, 0)
            title_mask = torch.cat(title_mask, 0).double()
            body_mask = torch.cat(body_mask, 0).double()

        title_inputs = Variable(title)
        body_inputs = Variable(body)
        title_mask = Variable(title_mask)
        body_mask = Variable(body_mask)

        titles_hidden = net(title_inputs)
        bodies_hidden = net(body_inputs)

        # apply mask
        titles_encoded = titles_hidden * title_mask.unsqueeze(1).expand_as(titles_hidden)
        bodies_encoded = bodies_hidden * body_mask.unsqueeze(1).expand_as(bodies_hidden)

        # average over words (div by actual length)
        titles_encoded = torch.sum(titles_encoded, dim=2) 
        bodies_encoded = torch.sum(bodies_encoded, dim=2)

        titles_encoded = titles_encoded / (torch.sum(title_mask, keepdim=True, dim=1).expand_as(titles_encoded) + eps)
        bodies_encoded = bodies_encoded / (torch.sum(body_mask, keepdim=True, dim=1).expand_as(bodies_encoded) + eps)

        # average title and body
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
    net = ConvNet(300, 667, 3, pad=1, dropout=0)
    net.double()
    net.load_state_dict(torch.load('./models/transfer1/encoder/epoch8', 
        map_location=lambda storage, loc: storage)) # transform back to CPU tensors
    is_cuda = False

    print 'test data'
    with open("data/part2/test_dataloader", "rb") as f:
        dataloader = pickle.load(f)

    auc = eval(dataloader, net, is_cuda, auc=True)

    print "AUC: ", auc
    # print "MRR: ", mrr
    # print "P@1: ", p1
    # print "P@5: ", p5