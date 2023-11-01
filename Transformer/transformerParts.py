import torch
import torch.nn as nn
from math import sin,cos

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
    
class SelfAttentionHead():
    def __init__(self,emgbedding_size:int,vectors_size:int) -> None:
        Wq = torch.tensor((),requires_grad=True)
        
        
class MultiHeadAttention():
    def __init__(self,heads,vectors_size:int) -> None:
        pass
    
    
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

