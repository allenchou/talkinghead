import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import data
import model


path = '/Users/allenchou/Documents/LIPS2008/_wav'
train_data_ratio = 0.8
hidden_dim = 128
layers_num = 2
cuda = False
log_interval = 10
epochs = 6
lr = 0.1
#############################
# Load data
#############################

corpus = data.Corpus(path, cuda=cuda)


def split(data_list, radio):
    random.shuffle(data_list)
    train_size = int(radio * len(data_list))
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    return train_data, test_data

train_data, val_data = split(corpus.data(), train_data_ratio)

#############################
# Build the model
#############################
input_dim = train_data[0][0].size(1)
output_dim = train_data[0][1].size(1)

print('input_dim = {}, output_dim = {}'.format(input_dim, output_dim))

net = model.RNNModel(input_dim, output_dim, hidden_dim, layers_num)
if cuda:
    net.cuda()

criterion = nn.MSELoss()

#############################
# Training code
#############################


def evaluate(data_source):
    net.eval()
    total_loss = 0
    for audio_feature, video_feature in val_data:
        input = Variable(audio_feature, volatile=True)
        target = Variable(video_feature)
        net.hidden = net.init_hidden()
        output = net(input)
        total_loss += criterion(output, target).data
    return total_loss[0] / len(data_source)


def train(epoch, lr):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    total_loss = 0
    start_time = time.time()
    for batch, (audio_feature, video_feature) in enumerate(train_data):
        net.zero_grad()
        net.hidden = net.init_hidden()
        input = Variable(audio_feature)
        target = Variable(video_feature)

        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

        batch += 1
        if batch % log_interval == 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f}'
                  .format(epoch, batch, len(train_data), lr,
                          elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()


best_val_loss = None
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(epoch, lr)
    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'
          .format(epoch, (time.time() - epoch_start_time), val_loss))
    print('-' * 89)

    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
    else:
        lr /= 4.0





















