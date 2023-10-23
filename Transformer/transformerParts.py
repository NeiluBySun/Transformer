import torch
import torch.nn as nn
from math import sin,cos


class Transformer():
    def __init__(self) -> None:
        pass

class Encoder():
    def __init__(self,number_of_blocks:int,vocab_size:int,embeding_size:int = 64,max_context_size:int = 30) -> None:
        self.embeding_layer = nn.Embedding((vocab_size,embeding_size))
        self.position_layer = self.positions(max_context_size,embeding_size)
        EncoderBlocks = [EncoderBlock() for i in range(number_of_blocks)]
        pass
    
    def TransformToEmbedings(self,context:list,pos_encoding:bool = True):
        pass
     
    def positions(self,d,embedding_size):
        arr = []
        for pos in range(d):
            vector = []
            for i in range(embedding_size):
                vector.append(sin(pos/10000**(2*i/d) if i % 2 ==0 else cos(pos/10000**(2*i/d))))
            arr.append(vector)
        return torch.tensor(arr)
class EncoderBlock():
    def __init__(self) -> None:
        pass