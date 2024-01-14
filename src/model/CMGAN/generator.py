import ftplib

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from ..conformer import ConformerBlock, CrossConformerBlock


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2))
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)
        self.num_features = num_features

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        x = x[:,:self.num_features,:]
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2))
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x

class Compress(nn.Module):
    def __init__(self, num_channel=64):
        super(Compress, self).__init__()
        self.compress = nn.Conv2d(num_channel, 1, (1, 1))

    def forward(self, x):
        x = self.compress(x)
        x = torch.mean(x, dim=-1, keepdim=True)
        x = torch.mean(x, dim=-2, keepdim=True)
        return x

class RadarNet(nn.Module):
    def __init__(self, in_channel, channels) -> None:
        super(RadarNet, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=2, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        self.TSCB = TSCB(num_channel=channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 1, (1, 1), (1, 1)),
            nn.InstanceNorm2d(1, affine=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        x = self.TSCB(x)
        x = self.out_conv(x)
        x = torch.mean(x, dim=-1, keepdim=True)
        return x

class CrossFusion(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads=4,
                 conv_kernel_size=31,
                 attn_dropout=0.2,
                 ff_dropout=0.2,) -> None:
        super().__init__()
        self.fusion_t = CrossConformerBlock(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
    
    def forward(self, x, context):
        b, c, t, f = x.size()
        x_t = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        context_t = context.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.fusion_t(x_t, context_t) + x_t
        x_fusion = x_t.view(b, f, t, c).permute(0, 3, 2, 1)
        return x_fusion


class TSCNet(nn.Module):
    def __init__(self,
                 num_channel:int=64,
                 num_features:int=201,
                 clean_mask = False
        ):
        super(TSCNet, self).__init__()
        self.clean_mask = clean_mask
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.radar_net = RadarNet(in_channel=1, channels=num_channel//4)

        self.fusion = CrossFusion(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

        self.TSCB = nn.Sequential(TSCB(num_channel=num_channel),
                                  TSCB(num_channel=num_channel))

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x, radar):
        
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)

        radar = radar.permute(0, 1, 3, 2)
        x_in = torch.cat([mag, x], dim=1)
        batch_size = x_in.shape[0]
        out_1 = self.dense_encoder(x_in)
        radar = self.radar_net(radar)
        radar = F.interpolate(radar, size=(out_1.shape[-2], 1), mode="bilinear", align_corners=False)
        
        out_1 = out_1.repeat_interleave(radar.shape[0]//out_1.shape[0],
                                        dim=0)
        
        out_2 = self.fusion(out_1, out_1*radar)
        
        out_2 = self.TSCB(out_2)

        mask = self.mask_decoder(out_2)
        mask = mask[...,:x.shape[-1]]
        mag = mag.repeat_interleave(mask.shape[0]//mag.shape[0],
                                    dim=0)
        out_mag = mask * mag

        complex_out = self.complex_decoder(out_2)
        complex_out = complex_out[...,:x.shape[-1]]
        noisy_phase = noisy_phase.repeat_interleave(complex_out.shape[0]//noisy_phase.shape[0],
                                                    dim=0)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
        final_imag = rearrange(final_imag, "(b c) n h w -> b (c n) h w", b = batch_size)
        final_real = rearrange(final_real, "(b c) n h w -> b (c n) h w", b = batch_size)
        
        return final_real, final_imag

