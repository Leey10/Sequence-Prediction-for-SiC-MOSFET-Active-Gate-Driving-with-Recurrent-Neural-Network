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
#--------------------------------------------------------------------
# --------------------------Data Processing--------------------------------
#--------------------------------------------------------------------
num_class = 22
# read ig sequence
Ig_data=pd.read_csv("../combined_Ig.csv", header = None).to_numpy()
num_data = len(Ig_data)
# read switching results and fectch Esw, didt, dvdt
Esw_pu = 4000e-6
dvdt_mean_pu = -72.65e9
dvdt_max_pu = -82.71e9
didt_pu = 8.3e9
TransPara = pd.read_csv("../combined_trans.csv", header = None).to_numpy()
# per unit values used, direct values also feasible as the input to BN layer
Esw = TransPara[:,4] / Esw_pu
dvdt = TransPara[:,3] / dvdt_max_pu
didt = TransPara[:,2] / didt_pu
Trans_data = np.concatenate(( didt.reshape((num_data,1)), dvdt.reshape((num_data,1)), Esw.reshape((num_data,1)) ), axis = 1)

# make the inputs a single array
full_data = np.array(np.concatenate((Trans_data, Ig_data), 1))
np.random.shuffle(full_data)

# split the data into train and validation data
num_data = len(full_data)
train_size = int(num_data * 0.8)
valid_size = num_data - train_size
train_data, valid_data = torch.utils.data.random_split(full_data, [train_size, valid_size])
train_data = np.array(train_data)
valid_data = np.array(valid_data)

#----------------------------------------------------------------------------------
#-------------------------training process---------------------
#----------------------------------------------------------------------------------
# define hyperparameters
num_epoch = 1000
learning_rate = 1e-3
batch_size = 1024
trans_size = 3
hidden_size = 360
embed_size = 50
num_layers = 3
num_workers = 2
min_loss = 100


loss_avg = 0
all_loss = []
print_every = 10


# instantiates the models
encoder = Encoder(trans_size, hidden_size, num_layers).to(device)
decoder = Decoder(embed_size, hidden_size, num_class, num_layers).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.95, patience = 1, min_lr = 1e-5)


# generate datasets
train_dataset = TrainDataSet(train_data)
valid_dataset = ValidDataSet(valid_data)
training_loader = data.DataLoader(dataset = train_dataset, 
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers = num_workers,
                               collate_fn=collate_fn)
    
validation_loader = data.DataLoader(dataset = valid_dataset, 
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers = num_workers,
                               collate_fn=collate_fn)
trainLoss = []
validLoss = []
    
for epoch in range(num_epoch):
   
    #-------------------------training phase------------------------------
    encoder.train()
    decoder.train()
    train_loss = 0
    
    for (trans, IgSeq, lengths) in training_loader:
        # set mini_batch
        trans = trans.to(device)
        lengths = [lengths[i]-1 for i in range(len(lengths))]
        IgSeq = IgSeq.to(device)
        labels = pack_padded_sequence(IgSeq[:,1:], lengths, batch_first = True)[0]
        # forward, backward and optimize
        h_0 = encoder(trans)
        output = decoder(IgSeq, h_0, lengths )
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss * len(trans) / len(train_dataset)
    #--------------------------end of training phase ---------------------
        
        
    #------------------------Valid phase-------------------------------------
    encoder.eval()
    decoder.eval()
    valid_loss = 0
    
    for (trans, IgSeq, lengths) in validation_loader:
        
        # set mini_batch
        trans = trans.to(device)
        lengths = [lengths[i]-1 for i in range(len(lengths))]
        IgSeq = IgSeq.to(device)
        labels = pack_padded_sequence(IgSeq[:,1:], lengths, batch_first = True)[0]
        
        # forward, backward and optimize
        h_0 = encoder(trans)
        output = decoder(IgSeq, h_0, lengths )
        loss = criterion(output, labels)
       
        valid_loss += loss * len(trans) / len(valid_dataset)
    
    #-------------------------end of valid phase--------------------------
    scheduler.step(valid_loss)
    if epoch % print_every == 0:
        if valid_loss < min_loss :
            min_loss = valid_loss
            checkpoint = {
            'epoch': epoch + 1,
            'train_loss' : trainLoss,
            'valid_loss' : validLoss,
            'min_loss' : min_loss,
            'encoder_state_dict':encoder.state_dict(),
            'decoder_state_dict':decoder.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict()
            }
            torch.save(checkpoint, "model-new.tar")
        trainLoss.append(train_loss )
        validLoss.append(valid_loss )
        print("epoch = {}, train_loss = {}, valid_loss = {}".format(epoch,train_loss,valid_loss))

