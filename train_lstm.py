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
num_epochs = 2
word2vec_vocab_size = 83916
glove_vocab_size = 122703
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, is_bidirectional, is_cuda)
model.double()
criterion = torch.nn.MultiMarginLoss(margin=1)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
two = Variable(torch.DoubleTensor([2]))
if is_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    two = Variable(torch.cuda.DoubleTensor([2]))

with open("data/part1/train_dataloader_1p", "rb") as f:
    dataloader = pickle.load(f)

start_time = time.time()
for epoch in xrange(num_epochs):
    total_epoch_loss = 0.
    model.train()
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
        # titles = (100 seq, 22 * batch_size, 200 hid_dim)
        # mask = (100 seq, 200 hid_dim)

        title_inputs = Variable(title_inputs)
        body_inputs = Variable(body_inputs)
        title_mask = Variable(title_mask)
        body_mask = Variable(body_mask)
        print title_inputs.size()
        print title_inputs[1].size()
        print title_mask.size()

        optimizer.zero_grad()

        titles_encoded = model(title_inputs, title_mask)
        bodies_encoded = model(body_inputs, body_mask)
        print titles_encoded.size()

        qs_encoded = (titles_encoded + bodies_encoded) / two
        
        cos_sims, y = maxmarginloss.batch_cos_sim(qs_encoded)
        print cos_sims[0]
        print cos_sims[5]
        print cos_sims[15]

        loss = criterion(cos_sims, y)
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.data[0]

    print epoch, " total training loss per epoch: ", total_epoch_loss
    # print "cos sim pos:"
    # print cos_sims[0]
    # print 'cos sims negative:'
    # print cos_sims[1:4]

torch.save(model.state_dict(), './models/lstm_train11')
end_time = time.time()
print('Finished Training in', (end_time - start_time) / 60) 

