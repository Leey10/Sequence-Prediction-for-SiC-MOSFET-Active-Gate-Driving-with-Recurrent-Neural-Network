import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence

from datasets import *
from model import Encoder, Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


# define hyperparameters
num_class = 22
num_epoch = 400
learning_rate = 5e-4
batch_size = 1024
trans_size = 3
hidden_size = 360
embed_size = 50
num_layers = 3
num_workers = 2
epoch_cnt = 0


loss_avg = 0
all_loss = []
print_every = 10


# instantiates the models
encoder = Encoder(trans_size, hidden_size, num_layers).to(device)
#print('encoder instantiate')
decoder = Decoder(embed_size, hidden_size, num_class, num_layers).to(device)
#print('decoder instantiate')

# loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 5, min_lr = 5e-5)
# load check point
checkpoint = torch.load("GRU_EDRNN.tar")
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
start_epoch = checkpoint['epoch']
trainLoss = checkpoint['train_loss']
validLoss = checkpoint['valid_loss']


print('valid loss=',validLoss[-1])
print('epoch = ', start_epoch)