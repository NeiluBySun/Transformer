import torch
import torch.nn as nn
from torch.nn import functional as F

from math import sin,cos,sqrt

class Transformer():
    def __init__(self,encoder,decoder) -> None:
        pass

class Encoder(nn.Module):
    def __init__(self,number_of_blocks:int,vocab_size:int,embedding_size:int = 64,max_context_size:int = 30) -> None:
        self.embedding_layer = nn.Embedding((vocab_size,embedding_size))
        self.position_layer = self.positions(max_context_size,embedding_size)
        EncoderBlocks = [EncoderBlock() for i in range(number_of_blocks)]
        pass
    
    def forward(self,context):
         embeding = self.embedding_layer(context)
         possed_embeding += self.position_layer
    def positions(self,d,embedding_size) -> torch.TensorType:
        arr = []
        for pos in range(d):
            vector = []
            for i in range(embedding_size):
                vector.append(sin(pos/10000**(2*i/d) if i % 2 ==0 else cos(pos/10000**(2*i/d))))
            arr.append(vector)
        return torch.tensor(arr,requires_grad=False)
class EncoderBlock():
    def __init__(self,context,heads:int,) -> None:
        pass
    
class SelfAttentionHead(nn.Module):
    def __init__(self,embedding_size:int,vectors_size:int) -> None:
        super(SelfAttentionHead, self).__init__()
        self.Wq = torch.rand((embedding_size, vectors_size),requires_grad=True)
        self.Wk = torch.rand((embedding_size,vectors_size),requires_grad=True)
        self.Wv = torch.rand((embedding_size,vectors_size),requires_grad=True)
        self.d_sqrt = sqrt(vectors_size)
        self.softmax = nn.Softmax(1)
    def forward(self,context_embedings):
        Q = context_embedings@self.Wq
        K = context_embedings@self.Wk
        V = context_embedings@self.Wv
        
        return self.softmax(Q@K.T/(self.d_sqrt))@V
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self,heads,embedding_size:int,vectors_size:int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.AttentionHeads = [SelfAttentionHead(embedding_size,vectors_size) for i in range(heads)]
        self.z = torch.tensor([])
        self.Wo = torch.rand((vectors_size*heads,embedding_size),requires_grad=True)
    def forward(self,context_embedings):
        for head in self.AttentionHeads:
            self.z = torch.cat((self.z,head(context_embedings)),dim=1)
        return self.z@self.Wo
    
    
    
class FeedForward(nn.Module):
    def __init__(self,inpu_output_size:int,hidden_layear_size:int) -> None:
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(inpu_output_size, hidden_layear_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layear_size, inpu_output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out   

