import torch
from torch.autograd import Variable
import torch.optim as optim
from lstm import LSTM
import pickle
import time
from load_data import *
import maxmarginloss
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

is_cuda = False
is_bidirectional = False
EMBEDDING_DIM = 200
HIDDEN_DIM = 240
kernel_size = 5
num_epochs = 20
word2vec_vocab_size = 83916
glove_vocab_size = 122703
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, kernel_size, is_bidirectional, is_cuda)
model.double()
criterion = torch.nn.MultiMarginLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
if is_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    eps = Variable(torch.cuda.DoubleTensor([.001]))
else:
    eps = Variable(torch.DoubleTensor([.001]))

with open("data/part1/train_dataloader_1p", "rb") as f:
    dataloader = pickle.load(f)

start_time = time.time()
for epoch in xrange(num_epochs):
    total_epoch_loss = 0.
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
        # print "title input", title_inputs.size()
        body_inputs = body_batch.permute(2, 0, 1)
        title_mask = title_mask.permute(1, 0)
        # print "title mask", title_mask.size()
        body_mask = body_mask.permute(1, 0)
        # titles = (100 seq, 22 * batch_size, 200 hid_dim)
        # mask = (100 seq, 200 hid_dim)

        title_inputs = Variable(title_inputs)
        body_inputs = Variable(body_inputs)
        title_mask = Variable(title_mask)
        body_mask = Variable(body_mask)

        optimizer.zero_grad()

        # attempted question by question
        # title_outs = [model(row) for row in title_inputs.split(1, dim=1)]
        # body_outs = [model(row) for row in body_inputs.split(1, dim=1)]
        # titles_hidden = torch.cat(title_outs, dim=1)
        # bodies_hidden = torch.cat(body_outs, dim=1)

        titles_hidden = model(title_inputs)
        bodies_hidden = model(body_inputs)
        # titles hidden = (100 seq, 22 * batch_size, 240 hid_dim)

        # print "title hidden", titles_hidden.size()
        # apply mask
        titles_encoded = titles_hidden * title_mask.unsqueeze(2).expand_as(titles_hidden)
        bodies_encoded = bodies_hidden * body_mask.unsqueeze(2).expand_as(bodies_hidden)
        # print "titles mult expanded mask", titles_encoded.size()
        # titles encoded = (100 seq, 22 * batch_size, 240 hid_dim)

        # average over the word and divide by the actual length
        titles_encoded = torch.sum(titles_encoded, dim=0)
        bodies_encoded = torch.sum(bodies_encoded, dim=0)
        # print "titles summed", titles_encoded.size()
        # titles encoded = (22 * batch_size, 240 hid_dim)

        titles_encoded = titles_encoded / (torch.sum(title_mask, keepdim=True, dim=0).permute(1, 0).expand_as(titles_encoded) + eps)
        bodies_encoded = bodies_encoded / (torch.sum(body_mask, keepdim=True, dim=0).permute(1, 0).expand_as(bodies_encoded) + eps)
        # print "titles div over summed mask", titles_encoded.size()
        # titles encoded = (22 * batch_size, 240 hid_dim)

        if is_cuda:
            qs_encoded = (titles_encoded + bodies_encoded) / Variable(torch.cuda.DoubleTensor([2]))
        else :
            qs_encoded = (titles_encoded + bodies_encoded) / Variable(torch.DoubleTensor([2]))

        # print "qs_encoded", qs_encoded.size()
        
        cos_sims, y = maxmarginloss.batch_cos_sim(qs_encoded)
        loss = criterion(cos_sims, y)
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.data[0]

    print epoch, " total training loss per epoch: ", total_epoch_loss
    print "cos sim pos:"
    print cos_sims[0]
    print 'cos sims negative:'
    print cos_sims[1:4]

torch.save(model.state_dict(), './models/lstm_20ep_q')
end_time = time.time()
print('Finished Training in', (end_time - start_time) / 60)

