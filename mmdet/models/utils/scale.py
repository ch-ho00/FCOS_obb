import torch
import torch.nn as nn


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
         return torch.cat([(x[:,:4,:,:]*self.scale).float().exp(), x[:,4,:,:].unsqueeze(1)],1)
#        return (x *self.scale).float().exp()
