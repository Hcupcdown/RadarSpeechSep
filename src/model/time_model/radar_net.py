import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .conformer import ConformerBlock
from .conv_modules import *


class Encoder(nn.Module):
    

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride, layer,
                 depth,
                 g1=2,
                 g2=2):
        super().__init__()
        
        self.layer      = layer
        self.depth      = depth
        self.down_conv  = nn.Sequential(nn.Conv1d(in_channels,
                                                  in_channels,
                                                  kernel_size,
                                                  stride),
                                        nn.InstanceNorm1d(in_channels),
                                        nn.ReLU())
        self.conv_block = ResConBlock(in_channels, growth1=g1, growth2=g2)

    def forward(self, x):

        x = self.down_conv(x)        
        x = self.conv_block(x)

        return x

class Decoder(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 layer,
                 depth,
                 g1=2,
                 g2=1/2):
        super().__init__()

        self.layer      = layer
        self.depth      = depth
        self.conv_block = ResConBlock(in_channels, growth1=g1, growth2=g2)
        self.up_conv    = nn.Sequential(nn.ConvTranspose1d(out_channels,
                                                           out_channels,
                                                           kernel_size,
                                                           stride),
                                        nn.InstanceNorm1d(out_channels),
                                        nn.ReLU())
    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.up_conv(x)

        return x
    
class RadarNet(nn.Module):


    def __init__(self,
                 in_channels:int,
                 embed_channels:int,
                 hidden:int,
                 depth:int,
                 device:str="cuda:0") -> None:
        super().__init__()

        self.out_channels = hidden*(2**depth)
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
                         out_channels=self.out_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1))

        self.resconv1d = nn.Sequential()
        for i in range(5):
            self.resconv1d.add_module(f'resblock{i}',
                                      ResBlock(Conv1dBlock(in_channels=self.out_channels,
                                                           out_channels=self.out_channels,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1)))
        # end for

        self.flatten_conv = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):

        x = rearrange(x, "b c h w -> (b c) h w")
        x = x.unsqueeze(dim = 1)
        x = self.in_conv(x)
        x = self.resconv2d(x)
        x = self.out_conv(x)

        x = x.squeeze(dim = 1)
        x = self.conv1d(x)
        x = self.resconv1d(x)
        x = self.flatten_conv(x)
        return x


class RadarSpeechSepNet(nn.Module):


    def __init__(self, 
                 in_channels:int=1, 
                 out_channels:int=1,
                 hidden:int=32,
                 depth:int=4,
                 kernel_size:int=8,
                 stride:int=4,
                 growth:int=2,
                 device:str="cuda:0"):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.device = device
        self.length = None

        self.in_conv = nn.Sequential(nn.Conv1d(in_channels,
                                               hidden,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.ReLU())

        self.out_conv = nn.Conv1d(hidden,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

        self.radar_net = RadarNet(in_channels=513,
                                  embed_channels=32,
                                  hidden=hidden,
                                  depth=depth,
                                  device=device)
        
        
        in_channels = in_channels*hidden
        out_channels = out_channels*growth
        
        encoders = []
        decoders = []
        for layer in range(depth):
            encoders.append(Encoder(in_channels=in_channels,
                                    out_channels=out_channels*hidden,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    layer=layer,
                                    depth=depth))

            decoders.append(Decoder(in_channels=out_channels*hidden,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    layer=layer,
                                    depth=depth))
        
            in_channels  = hidden*(2**(layer+1))
            out_channels *= growth
        # end for

        self.fusion_net = nn.Sequential(nn.Conv1d(in_channels=2*in_channels,
                                                  out_channels=in_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1))
        decoders.reverse()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        for i in range(3):
            setattr(self,
                    f"conformer_{i}",
                    ConformerBlock(dim=in_channels,
                                   ff_mult=4,
                                   ff_dropout=0.1,
                                   conv_dropout=0.1))

    def encode(self, x):
        x = x.unsqueeze(dim = 1)
        length = x.shape[-1]
        self.cal_padding(length)
        x = F.pad(x, (0, self.length - length))
        x = self.in_conv(x)

        skips = []
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)
        
        return x, length, skips
    
    def decode(self, x, length, skips = None):
        for decoder in self.decoders:
            skip = skips.pop()
            skip = skip.repeat_interleave(x.shape[0]//skip.shape[0], dim=0)
            x = decoder(x)
            x  = x + skip

        x = self.out_conv(x) 
        x = x[..., :length]
        
        return x
    
    def forward(self, sound, radar):
        """
        Forward pass of the radar_net model.

        Args:
            sound (torch.Tensor): Sound tensor with shape [b, 1, T].
            radar (torch.Tensor): Radar tensor with shape [b, speaker_num, fre_num, time_len].

        Returns:
            torch.Tensor: Sound tensor with shape [batch_size*speaker_num, 1, T].
        """
        sound, length, skips = self.encode(sound)
        
        # generate radar mask
        radar_mask = self.radar_net(radar)
        sound = sound.repeat_interleave(radar_mask.shape[0]//sound.shape[0], dim=0)
        
        # mask sound
        mask = F.interpolate(radar_mask, size=sound.shape[-1])
        masked_sound = mask * sound
        
        # fusion masked sound and original sound
        sound = self.fusion_net(torch.cat((masked_sound, sound), dim=1))
        
        # conformer
        sound = sound.permute(0, 2, 1).contiguous()
        for i in range(3):
            sound = getattr(self, f"conformer_{i}")(sound)
        sound = sound.permute(0, 2, 1).contiguous()

        # decode
        y = self.decode(sound, length, skips)

        return y
    
    def cal_padding(self, length):
        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        self.length =  int(length)

if __name__=='__main__':
    testnet = RadarNet(in_channels=500,
                       embed_channels=32,
                       hidden=32,
                       depth=4)
    test_tensor = torch.randn(1, 1, 500, 500)
    res = testnet(test_tensor)
    # res = res.unsqueeze(dim=1)
    res = F.interpolate(res,
                        size=(128,),
                        mode='linear',
                        align_corners=False)
    print(res.shape)