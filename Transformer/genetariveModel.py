import transformerParts
import torch

eb = transformerParts.EncoderBlock(4,10)

print(eb(torch.rand(10,10)).size())

