import torch
import torch.optim as optim
from torch.autograd import Variable
from cnn import ConvNet
import pickle
from load_data import *
from load_android_eval import *
import maxmarginloss
import time
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import eval_network

is_cuda = True
num_epochs = 10
dataloader_path = "data/part1/train_glove_dataloader"
dev_dataloader_path = "data/part2/dev_dataloader"
model_save_path = './models/dir_transfer/'
net = ConvNet(300, 667, 3, pad=1)
net.double()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
criterion = torch.nn.MultiMarginLoss()
mrrs = []
maps = []
p1s = []
p5s = []
aucs = []

if is_cuda:
    net = net.cuda()
    eps = Variable(torch.cuda.DoubleTensor([.001]))
else:
    eps = Variable(torch.DoubleTensor([.001]))

with open(dataloader_path, "rb") as f:
    dataloader = pickle.load(f)

with open(dev_dataloader_path, "rb") as f:
    dev_dataloader = pickle.load(f)

start_time = time.time()
for epoch in xrange(num_epochs):
    total_epoch_loss = 0.  
    for title, body, title_mask, body_mask in tqdm(dataloader):
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

        # wrap w Variable for autograd 
        title_inputs = Variable(title)
        body_inputs = Variable(body)
        title_mask = Variable(title_mask)
        body_mask = Variable(body_mask)

        # zero the parameter gradients
        optimizer.zero_grad()
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
        
        if is_cuda:
            qs_encoded = (titles_encoded + bodies_encoded) / Variable(torch.cuda.DoubleTensor([2]))
        else:
            qs_encoded = (titles_encoded + bodies_encoded) / Variable(torch.DoubleTensor([2]))

        cos_sims, y = maxmarginloss.batch_cos_sim(qs_encoded)
        loss = criterion(cos_sims, y)
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.data[0]

    torch.save(net.state_dict(), model_save_path + "epoch" + str(epoch))

    print epoch, " total training loss per epoch: ", total_epoch_loss
    print "cos sim pos:"
    print cos_sims[0]
    print 'cos sims negative:'
    print cos_sims[1:6], '\n'

    prev_encoder = ConvNet(300, 667, 3, pad=1, dropout=0)
    prev_encoder.double()
    prev_encoder.load_state_dict(torch.load(model_save_path + "epoch" + str(epoch)))
    if is_cuda:
        prev_encoder.cuda()
    auc = eval_network.eval(dev_dataloader, prev_encoder, is_cuda, auc=True)
    print "AUC: ", auc
    aucs.append(auc)

end_time = time.time()
print('Finished Training in', (end_time - start_time) / 60)

with open(model_save_path + "dev_metrics/auc", "wb") as f:
    pickle.dump(aucs, f)

#     prev_model = ConvNet(200, 667, 3, pad=1, dropout=0)
#     prev_model.double()
#     prev_model.load_state_dict(torch.load(model_save_path + epoch + str(epoch)))
#     if is_cuda:
#         prev_model.cuda()
#     MAP, mrr, p1, p5 = eval_network.eval(dev_dataloader, prev_model, is_cuda)
#     maps.append(MAP)
#     mrrs.append(mrr)
#     p1s.append(p1)
#     p5s.append(p5)

#     print "MAP on dev:", MAP
#     print "MRR on dev: ", mrr
#     print "P@1 on dev: ", p1
#     print "P@5 on dev: ", p5


# end_time = time.time()
# print('Finished Training in', (end_time - start_time) / 60)

# with open(model_save_path + "dev_metrics/map", "wb") as f:
#     pickle.dump(maps, f)
# with open(model_save_path + "dev_metrics/mrr", "wb") as f:
#     pickle.dump(mrrs, f)
# with open(model_save_path + "dev_metrics/p1", "wb") as f:
#     pickle.dump(p1s, f)
# with open(model_save_path + "dev_metrics/p5", "wb") as f:
#     pickle.dump(p5s, f)