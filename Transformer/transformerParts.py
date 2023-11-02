import torch
import torch.nn as nn
from torch.nn import functional as F

from math import sin,cos,sqrt

class Transformer():
    def __init__(self,encoder,decoder) -> None:
        pass

class Encoder(nn.Module):
    def __init__(
        self,
        number_of_blocks:int,
        vocab_size:int,
        embedding_size:int = 64,
        max_context_size:int = 30) -> None:
        
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
class EncoderBlock(nn.Module):
    def __init__(self,heads:int,embedding_size:int) -> None:
        super(EncoderBlock, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(heads,embedding_size,int(sqrt(embedding_size)))
        self.linear_layer1 = nn.Linear(in_features=embedding_size, out_features=embedding_size*2)
        self.linear_layer2 = nn.Linear(in_features=embedding_size*2, out_features=embedding_size)
        self.activation = nn.LeakyReLU()
        
    def forward(self,context:torch.Tensor):
        attention_matrix = self.MultiHeadAttention(context)
        addNorm = self.LayerNorm(context,attention_matrix)
        l1 = self.linear_layer1(addNorm)
        a = self.activation(l1)
        ff = self.linear_layer2(a)
        addNorm = self.LayerNorm(addNorm,ff)
        return addNorm 
         
    def LayerNorm(self,context,z_vectors):
        n = context.size(dim=0)
        mat_sum = context+z_vectors
        deviation = torch.std(mat_sum,dim=1)
        mean = torch.mean(mat_sum,dim=1)
        return (mat_sum - mean.view(n, 1))/deviation.view(n, 1)
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
    def __init__(
        self,
        heads:int,
        embedding_size:int,
        vectors_size:int) :
        
        super(MultiHeadAttention, self).__init__()
        self.AttentionHeads = [SelfAttentionHead(embedding_size,vectors_size) for i in range(heads)]
        self.z = torch.tensor([])
        self.Wo = torch.rand((vectors_size*heads,embedding_size),requires_grad=True)
    def forward(self,context_embedings):
        for head in self.AttentionHeads:
            self.z = torch.cat((self.z,head(context_embedings)),dim=1)
        return self.z@self.Wo
    
    
    
class FeedForward(nn.Module):
    def __init__(
        self,
        inpu_output_size:int,
        hidden_layear_size:int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(inpu_output_size, hidden_layear_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layear_size, inpu_output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out   

