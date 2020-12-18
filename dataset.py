import torch
import torch.utils.data as data
import numpy as np

class TrainDataSet(data.Dataset):
    def __init__(self,train_dataset):
        self.len = len(train_dataset)
        self.transPara = train_dataset[:,0:3]
        self.IgSeq = train_dataset[:,3:]
        
    def __getitem__(self, index):
        IgSeq = np.trim_zeros(self.IgSeq[index])
        IgSeq = [int( np.round(Ig/0.1) ) for Ig in IgSeq  ]
        IgSeq.insert(0,0)
        IgSeq.append(21)
        return torch.Tensor(self.transPara[index]), torch.LongTensor(IgSeq)
    
    def __len__(self):
        return self.len

class ValidDataSet(data.Dataset):
    def __init__(self,valid_dataset):
        self.len = len(valid_dataset)
        self.transPara = valid_dataset[:,0:3]
        self.IgSeq = valid_dataset[:,3:]
        
    def __getitem__(self, index):
        IgSeq = np.trim_zeros(self.IgSeq[index])
        IgSeq = [int( np.round(Ig/0.1) ) for Ig in IgSeq  ]
        IgSeq.insert(0,0)
        IgSeq.append(21)
        return torch.Tensor(self.transPara[index]), torch.LongTensor(IgSeq)
    
    def __len__(self):
        return self.len 

# build customized collate_fn that generates sorted sequence batch
# made compatible for pack_padded_sqeuence()
def collate_fn(pack):
    pack.sort( key = lambda x: len(x[1]), reverse  = True )
    trans, IgSeq = zip(*pack)
    
    trans = torch.stack(trans,0)
    
    lengths = [len(Ig) for Ig in IgSeq]
    sequence = torch.zeros( len(IgSeq), max(lengths) ).long()
    for i, seq in enumerate(IgSeq):
        end = lengths[i]
        sequence[i, :end] = seq[:end]
    return torch.Tensor(trans), torch.LongTensor(sequence), torch.LongTensor(lengths)
    
