import transformerParts
import torch

encoder = transformerParts.Encoder(3,100,20,5,5)

encoder(torch.tensor([1,2,3,4,5]))



