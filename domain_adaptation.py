import torch
import torch.optim as optim
from torch.autograd import Variable

from lstm import LSTM
from domain_classifier import DomainClassif
from load_data import *
from load_android_eval import *
from load_android_data import *
import maxmarginloss
import eval_lstm

import pickle
import itertools
import time
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

is_cuda = False
num_epochs = 5
hidden_dim = 240
model_save_path = "./models/transfer/"
# dev_dataloader_path = "data/part2/dev_dataloader"
test_dataloader_path = "data/part2/test_dataloader"
b1_dataloader_path = "data/part1/train_glove_dataloader_10p"
b2_dataloader_path = "data/part2/dataloader"
aucs = []

encoder = LSTM(300, hidden_dim, is_cuda=is_cuda)
encoder.double()
domain_discriminator = DomainClassif(hidden_dim, 128)
domain_discriminator.double()

if is_cuda:
    encoder = encoder.cuda()
    domain_discriminator = domain_discriminator.cuda()
    lmbda = Variable(torch.cuda.DoubleTensor([.0001]))
    eps = Variable(torch.cuda.DoubleTensor([.001]))
    two = Variable(torch.cuda.DoubleTensor([2]))
else:
    lmbda = Variable(torch.DoubleTensor([.0001]))
    eps = Variable(torch.DoubleTensor([.001]))
    two = Variable(torch.DoubleTensor([2]))

labels_optimizer = optim.Adam(encoder.parameters(), lr = 0.001)
domain_optimizer = optim.Adam(domain_discriminator.parameters(), lr=-.001)

labels_criterion = torch.nn.MultiMarginLoss()
domain_criterion = torch.nn.BCELoss()

# ubuntu with glove
with open(b1_dataloader_path, "rb") as f:
    b1_dataloader = pickle.load(f)
# android dataset
with open(b2_dataloader_path, "rb") as f:
    b2_dataloader = pickle.load(f)
# android dev
with open(test_dataloader_path, "rb") as f:
    dev_dataloader = pickle.load(f)

start_time = time.time()
for epoch in xrange(num_epochs):
    total_epoch_loss = 0.
    encoder.train()
    domain_discriminator.train()
    for b1, b2 in tqdm(itertools.izip(b1_dataloader, b2_dataloader)):
        b1_title, b1_body, b1_title_mask, b1_body_mask = b1
        b2_title, b2_body, b2_title_mask, b2_body_mask, b2_labels = b2
        # transform to correct dims
        if is_cuda:
            b1_title, b1_body, b1_title_mask, \
            b1_body_mask, b2_title, b2_body, b2_title_mask, \
            b2_body_mask, b2_labels = map(lambda x: torch.cat(x.cuda(), 0).double(), 
                    [b1_title, b1_body, b1_title_mask, b1_body_mask, b2_title, b2_body, b2_title_mask, b2_body_mask, b2_labels])
        else:
            b1_title, b1_body, b1_title_mask, \
            b1_body_mask, b2_title, b2_body, b2_title_mask, \
            b2_body_mask, b2_labels = map(lambda x: torch.cat(x, 0).double(), 
                    [b1_title, b1_body, b1_title_mask, b1_body_mask, b2_title, b2_body, b2_title_mask, b2_body_mask, b2_labels])
        
        # wrap w Variable for autograd
        b1_title, b1_body = Variable(b1_title.permute(2, 0, 1)), Variable(b1_body.permute(2, 0, 1))
        b1_title_mask, b1_body_mask = Variable(b1_title_mask.permute(1, 0)), Variable(b1_body_mask.permute(1, 0))
        b2_title, b2_body, b2_labels = Variable(b2_title.permute(2, 0, 1)), Variable(b2_body.permute(2, 0, 1)), Variable(b2_labels)
        b2_title_mask, b2_body_mask = Variable(b2_title_mask.permute(1, 0)), Variable(b2_body_mask.permute(1, 0))
        
        # LABEL TRAINING
        labels_optimizer.zero_grad()
        b1_titles_encoded = encoder(b1_title, b1_title_mask)
        b1_bodies_encoded = encoder(b1_body, b1_body_mask)
        
        # DOMAIN TRAINING
        domain_optimizer.zero_grad()
        b2_titles_encoded = encoder(b2_title, b2_title_mask)
        b2_bodies_encoded = encoder(b2_body, b2_body_mask)

        # final encoding = title and body avged
        b1_encoded = (b1_titles_encoded + b1_bodies_encoded) / two
        b2_encoded = (b2_titles_encoded + b2_bodies_encoded) / two
        
        cos_sims, y = maxmarginloss.batch_cos_sim(b1_encoded, is_cuda)
        domain_preds = domain_discriminator(b2_encoded)

        b1_loss = labels_criterion(cos_sims, y)
        b2_loss = domain_criterion(domain_preds, b2_labels)

        total_loss = b1_loss  - lmbda * b2_loss

        total_loss.backward()
        labels_optimizer.step()
        domain_optimizer.step()
        total_epoch_loss += total_loss.data[0]

    torch.save(encoder.state_dict(), model_save_path + "encoder/epoch" + str(epoch))
    torch.save(domain_discriminator.state_dict(), model_save_path + "discrim/epoch" + str(epoch))

    print epoch, " total training loss per epoch: ", total_epoch_loss

    prev_encoder = LSTM(300, hidden_dim)
    prev_encoder.double()
    prev_encoder.load_state_dict(torch.load(model_save_path + "encoder/epoch" + str(epoch)))
    if is_cuda:
        prev_model.cuda()

    auc = eval_lstm.eval(dev_dataloader, prev_encoder, is_cuda, auc=True)
    print "AUC: ", auc
    aucs.append(auc)

end_time = time.time()
print('Finished Training in', (end_time - start_time) / 60)

with open(model_save_path + "dev_metrics/auc", "wb") as f:
    pickle.dump(aucs, f)
