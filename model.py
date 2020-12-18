import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, trans_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.trans_size = trans_size
        self.n_layer = num_layers
        
        self.linear = nn.Linear(trans_size , hidden_size)
        self.bn = nn.BatchNorm1d(trans_size)
        self.activation = nn.ReLU()
       
    
    def forward(self, trans_input):
        h_0 = ( self.linear( self.bn(trans_input) ) )
        h_0 = h_0.view(1,-1,self.hidden_size) # change shape for gru h_0 input
        batch = len(trans_input)
        h_0 = torch.cat( (h_0, torch.Tensor( torch.zeros(self.n_layer - 1, batch, self.hidden_size) ).to(device)) , 0 )
        
        return h_0
        
        
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_class, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_class, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first = True, dropout = 0.5)
        self.linear = nn.Linear(hidden_size, num_class)
        self.bn = nn.BatchNorm1d(num_class)
        self.dropout = nn.Dropout(p=0.5)
        self.n_class = num_class
        
    def forward(self, Ig_input, h_0, length):
        embeddings = self.embed(Ig_input)
        pack_input = pack_padded_sequence(embeddings, length, batch_first = True)
        output, h_n = self.gru(pack_input, h_0)
        output = self.dropout(output[0])
        output = self.bn( self.linear(output) )
        return output
        
    # for the inferring: having <start> as the first input, let the decoder circulate
    def sample(self, h_0):
        num_max_length = 100
        hidden = h_0
        sampled_id = []
        Ig = torch.LongTensor([0]).to(device)
        for i in range(num_max_length):
            input = self.embed(Ig)
            input = input.view(1,1,-1)
            output, hidden = self.gru(input, hidden)
            output = self.linear(output.squeeze(1))
            _, predicted = output.max(1)
            sampled_id.append(predicted.item())
            if predicted == self.n_class - 1:
                return sampled_id
            Ig = torch.LongTensor([predicted]).to(device)
        return sampled_id

