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
    def __init__(self,context:torch.tensor,heads:int,) -> None:
        pass