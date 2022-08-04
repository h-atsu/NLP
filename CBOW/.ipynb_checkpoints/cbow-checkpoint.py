import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from datasets import ptb
import dezero.layers as L
import dezero.functions as F
from dezero.optimizers import Adam, SGD
from dezero import Variable, Model
from dezero import DataLoader
import matplotlib.pyplot as plt
from utils import *
from data import PTB
from tqdm import tqdm
import dezero

window_size = 5
hidden_size = 100
batch_size = 100
max_eopch = 10
eps = 1e-8


trainset = PTB(window_size=window_size)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
corpus = trainset.corpus
vocab_size = len(trainset.word_to_id)


class CBOW(Model):
    def __init__(self, vocab_size, hidden_dim=10):
        super().__init__()
        self.embed_in = L.EmbedID(vocab_size, hidden_dim)
        self.embed_out = L.EmbedID(vocab_size, hidden_dim)
        self.sampler = UnigramSampler(corpus)

    def forward(self, x):
        return self.embed_in(x).sum(axis=1)

    def embed(self, y):
        return self.embed_out(y)


model = CBOW(vocab_size, hidden_size)
optimizer = Adam().setup(model)
sampler = UnigramSampler(corpus)

if os.path.exists(os.path.join(os.path.dirname(__file__), 'ptb_cbow.npz')):
    model.load_weights(os.path.join(os.path.dirname(__file__), 'ptb_cbow.npz'))


info = {}
best_loss = 4000

info['train_loss'] = []
for epoch in tqdm(range(max_eopch)):
    loss_sum = []
    for x, y in tqdm(trainloader, total=trainloader.data_size / trainloader.batch_size, leave=False):
        loss = 0

        # positive example
        pred = model(x)
        embed = model.embed(y)
        prob = F.sigmoid((pred * embed).sum(axis=1))
        loss -= F.log(prob + eps).sum()

        # negative example
        neg_y = sampler.get_negative_sample(y)
        neg_embed = model.embed(neg_y).transpose(1, 0, 2)
        for i in range(len(neg_embed)):
            prob = F.sigmoid((pred * neg_embed[i]).sum(axis=1))
            loss -= F.log(1 - prob + eps).sum()

        # update parameters
        model.cleargrads()
        loss.backward()
        optimizer.update()
        info['train_loss'] += [loss.data]
        loss_sum += [loss.data]
    loss_mean = np.array(loss_sum).mean()
    print(" ======= train loss: {} ======".format(loss_mean))
    if loss_mean < best_loss:
        best_loss = loss_mean
        model.save_weights(os.path.dirname(__file__) + '/ptb_cbow.npz')


plt.plot(info['train_loss'])
