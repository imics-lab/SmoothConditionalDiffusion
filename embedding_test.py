from support.modules1D_cls_free import SinusoidalPosEmb
import torch

emb = SinusoidalPosEmb(64)
x = torch.zeros((1, 32))
x_emb = emb(x)
print(x_emb)