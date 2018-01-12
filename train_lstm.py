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
EMBEDDING_DIM = 300 # 300 if transfer, 200 o.w.
HIDDEN_DIM = 240
num_epochs = 5
word2vec_vocab_size = 83916
glove_vocab_size = 122703
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, is_bidirectional, is_cuda)
model.double()
criterion = torch.nn.MultiMarginLoss(margin=1)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
# print model parameters
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print params
two = Variable(torch.DoubleTensor([2]))
if is_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    two = Variable(torch.cuda.DoubleTensor([2]))

with open("data/part1/train_glove_dataloader", "rb") as f:
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

        title_inputs = Variable(title_inputs)
        body_inputs = Variable(body_inputs)
        title_mask = Variable(title_mask)
        body_mask = Variable(body_mask)

        optimizer.zero_grad()

        titles_encoded = model(title_inputs, title_mask)
        bodies_encoded = model(body_inputs, body_mask)

        qs_encoded = (titles_encoded + bodies_encoded) / two
        
        cos_sims, y = maxmarginloss.batch_cos_sim(qs_encoded, is_cuda)

        loss = criterion(cos_sims, y)
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.data[0]

    print epoch, " total training loss per epoch: ", total_epoch_loss

torch.save(model.state_dict(), './models/lstm_direct_5ep')
end_time = time.time()
print('Finished Training in', (end_time - start_time) / 60)

