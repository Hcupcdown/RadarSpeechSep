import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..conformer import ConformerBlock
from ..Mossformer.conv_modules import *


class RadarNet(nn.Module):


    def __init__(self,
                 in_channels:int,
                 embed_channels:int,
                 out_channels:int=512) -> None:
        super().__init__()

        self.in_conv = Conv2dBlock(in_channels=1,
                                   out_channels=embed_channels, 
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        
        self.resconv2d = nn.Sequential()
        for i in range(5):
            self.resconv2d.add_module(f'resblock{i}',
                                      ResBlock(Conv2dBlock(in_channels=embed_channels,
                                                           out_channels=embed_channels,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1)))
        # end for

        self.out_conv = Conv2dBlock(in_channels=embed_channels,
                                    out_channels=1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        
        self.conv1d = nn.Sequential(
             Conv1dBlock(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1))

        self.resconv1d = nn.Sequential()
        for i in range(5):
            self.resconv1d.add_module(f'resblock{i}',
                                      ResBlock(Conv1dBlock(in_channels=out_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1)))
        # end for

        # self.flatten_conv = nn.Conv1d(
        #     in_channels=hidden,
        #     out_channels=1,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )

    def forward(self, x):

        x = rearrange(x, "b c h w -> (b c) h w")
        x = x.unsqueeze(dim = 1)
        x = self.in_conv(x)
        x = self.resconv2d(x)
        x = self.out_conv(x)

        x = x.squeeze(dim = 1)
        x = self.conv1d(x)
        x = self.resconv1d(x)
        # x = self.flatten_conv(x)
        return x