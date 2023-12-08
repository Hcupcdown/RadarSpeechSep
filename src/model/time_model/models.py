import math

import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .conformer import ConformerBlock
from .conv_modules import *
from .radar_net import RadarNet


class MaskGate(nn.Module):
    # Original copyright:
    # The copy right is under the MIT license.
    # MANNER (https://github.com/winddori2002/MANNER) / author: winddori2002
    def __init__(self, channels):
        super().__init__()
        
        self.output      = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1),
                                         nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1),
                                         nn.Sigmoid())
        self.mask        = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1),
                                         nn.ReLU())

    def forward(self, x):

        mask = self.output(x) * self.output_gate(x)
        mask = self.mask(mask)
    
        return mask
    
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
        self.down_conv  = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size, stride),
                                        nn.InstanceNorm1d(in_channels),
                                        nn.ReLU())
        self.conv_block = ResConBlock(in_channels, growth1=g1, growth2=g2)

    def forward(self, x):

        x = self.down_conv(x)        
        x = self.conv_block(x)

        return x

class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, layer, depth, g1=2, g2=1/2):
        super().__init__()
        
        self.layer      = layer
        self.depth      = depth
        self.conv_block = ResConBlock(in_channels, growth1=g1, growth2=g2)
        self.up_conv    = nn.Sequential(nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride),
                                        nn.InstanceNorm1d(out_channels),
                                        nn.ReLU())
    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.up_conv(x)

        return x

class MaskFusionNet(nn.Module):

    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 hidden:int,
                 codebook_dim:int,
                 codebook_size:int,
                 depth:int,
                 kernel_size:int=8,
                 stride:int=4,
                 growth:int=2,
                 device:str="cuda:0"):
        super().__init__()
                
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.depth    = depth
        self.device = device
        self.in_conv  = nn.Sequential(nn.Conv1d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU())
        self.clean_in_conv  = nn.Sequential(nn.Conv1d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU())
        self.out_conv = nn.Conv1d(hidden, in_channels, kernel_size=3, stride=1, padding=1)
        self.clean_out_conv = nn.Sequential(nn.Conv1d(hidden, in_channels, kernel_size=3, stride=1, padding=1),
                                            nn.Sigmoid())

        self.radar_net = RadarNet(in_channels=513,
                                  embed_channels=32,
                                  hidden=hidden,
                                  depth=depth,
                                  device=device)
        self.length = None  
        in_channels   = in_channels*hidden
        out_channels  = out_channels*growth
        
        encoders = []
        decoders = []
        clean_encoders = []
        mask_decoders = []
        for layer in range(depth):
            encoders.append(Encoder(in_channels=in_channels,
                                   out_channels=out_channels*hidden,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   layer=layer,
                                   depth=depth))
            clean_encoders.append(Encoder(in_channels=hidden,
                                          out_channels=hidden,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          layer=layer,
                                          depth=depth,
                                          g1=2,
                                          g2=1))
            decoders.append(Decoder(in_channels=out_channels*hidden,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   layer=layer,
                                   depth=depth))
            mask_decoders.append(Decoder(in_channels=1,
                                   out_channels=1,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   layer=layer,
                                   depth=depth,
                                   g2=1))

            in_channels  = hidden*(2**(layer+1))
            out_channels *= growth

        decoders.reverse()
        mask_decoders.reverse()
        self.flag = 0
        self.encoders = nn.ModuleList(encoders)
        self.clean_encoders = nn.ModuleList(clean_encoders)
        self.decoders = nn.ModuleList(decoders)
        self.mask_decoders = nn.ModuleList(mask_decoders)

        for i in range(3):
            setattr(self, f"conformer_{i}", ConformerBlock(dim=in_channels, ff_mult=4, ff_dropout=0.1, conv_dropout=0.1))
        
        self.mask_gate = MaskGate(hidden)

    def encode(self, x, clean):

        length = x.shape[-1]
        self.cal_padding(length)
        x = F.pad(x, (0, self.length - length))
        clean = F.pad(clean, (0,self.length - length))
        x = self.in_conv(x)
        clean = self.clean_in_conv(clean)

        skips = []

        for encoder, cencode in zip(self.encoders, self.clean_encoders):
            skips.append(x)
            x = encoder(x)
            clean = cencode(clean)
        clean = self.clean_out_conv(clean)
        return x, clean, length, skips
    
    def decode(self, x, mask, length, skips = None):

        for decoder, mask_decoder in zip(self.decoders, self.mask_decoders):
            skip = skips.pop()
            mask = mask_decoder(mask)
            x = decoder(x)
            x  = x + skip*mask

        x = self.out_conv(x) 
        x = x[..., :length]
        
        return x
    

    def quantize_mask(self, mask):
        quantized_mask = torch.zeros_like(mask)
        quantized_mask[mask > 0.5] = 1
        quantized_mask = quantized_mask + mask - mask.detach()
        return quantized_mask
    
    def forward(self, sound, clean):
        """
        input X : [B, 1, T]
        output X: [B, 1, T]
        """
        sound, mask, length, skips = self.encode(sound, clean)

        x = sound.permute(0,2,1).contiguous()
        for i in range(3):
            x = getattr(self, f"conformer_{i}")(x) + x
        x = x.permute(0,2,1).contiguous()
        sound = x + sound
        # mask = self.quantize_mask(mask) 
        # mask = F.interpolate(mask, size=(sound.shape[-1],), mode='linear', align_corners=False)
        y = self.decode(sound, mask, length, skips)

        return  y, mask
    
    def get_clean_mask(self, clean):
        length = clean.shape[-1]
        self.cal_padding(length)

        clean = F.pad(clean, (0,self.length - length))
        clean = self.clean_in_conv(clean)

        for cencode in self.clean_encoders:
            clean = cencode(clean)
        clean = self.clean_out_conv(clean)
        # mask = self.quantize_mask(clean)
        return clean


    def encode_by_mask(self, sound):
        length = sound.shape[-1]
        self.cal_padding(length)
        sound = F.pad(sound, (0, self.length - length))
        sound = self.in_conv(sound)
        skips = []

        for encoder in self.encoders:
            skips.append(sound)
            sound = encoder(sound)

        x = sound.permute(0,2,1).contiguous()
        for i in range(3):
            x = getattr(self, f"conformer_{i}")(x) + x
        x = x.permute(0,2,1).contiguous()
        sound = x + sound

        return sound, skips
    

    def cal_padding(self, length):
        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        self.length =  int(length)


class SoundOnlyNet(nn.Module):

    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 hidden:int,
                 codebook_dim:int,
                 codebook_size:int,
                 depth:int,
                 kernel_size:int=8,
                 stride:int=4,
                 growth:int=2,
                 device:str="cuda:0"):
        super().__init__()
                
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.depth    = depth
        self.device = device
        self.in_conv  = nn.Sequential(nn.Conv1d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU())
        self.out_conv = nn.Conv1d(hidden, in_channels, kernel_size=3, stride=1, padding=1)
        self.length = None  
        in_channels   = in_channels*hidden
        out_channels  = out_channels*growth
        
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
        for i in range(3):
            setattr(self, f"conformer_{i}", ConformerBlock(dim=in_channels, ff_mult=4, ff_dropout=0.1, conv_dropout=0.1))
        
        decoders.reverse()
        self.flag = 0
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.mask_gate = MaskGate(hidden)

    def encode(self, x):

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
            x = decoder(x)
            x  = x + skip
        x = self.out_conv(x) 
        x = x[..., :length]
        
        return x
    
    def forward(self, sound):
        """
        input X : [B, 1, T]
        output X: [B, 1, T]
        """
        sound, length, skips = self.encode(sound)
        
        x = sound.permute(0,2,1).contiguous()
        for i in range(3):
            x = getattr(self, f"conformer_{i}")(x) + x
        x = x.permute(0,2,1).contiguous()
        sound = x + sound
        y = self.decode(sound, length, skips)

        return  y
    
    def cal_padding(self, length):
        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        self.length =  int(length)