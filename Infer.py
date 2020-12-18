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
trans_size = 3
hidden_size = 360
embed_size = 50
num_layers = 3
num_workers = 2
num_class = 22


# instantiates the models
encoder = Encoder(trans_size, hidden_size, num_layers).to(device)
decoder = Decoder(embed_size, hidden_size, num_class, num_layers).to(device)

# load check point
checkpoint = torch.load("GRU_EDRNN.tar")
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
trainLoss = checkpoint['train_loss']
validLoss = checkpoint['valid_loss']
    
encoder.eval()
decoder.eval()

# user defined switching targets for the GRU-EDRNN
Esw_x = 0.28
didt_x = 0.26
dvdt_x = 0.4

# Inferring process
trans = torch.Tensor([[didt_x,dvdt_x,Esw_x]]).to(device)
h_0 = encoder(trans)
output = decoder.sample(h_0)

# print to terminal and output .txt without <end>
print(output)
File = open(r"AGD_Sequence.txt","w")
File.write(str(output[:-1]).strip('[]'))
File.close()


    