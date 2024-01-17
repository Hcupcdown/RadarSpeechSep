from .log import Log
from .metric import sisnr
from .separate_loss import SeparateLoss
from .utils import *


def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return real_compress + 1j*imag_compress

def sound2stft(sound, n_fft, hop):
    sound_dim = sound.dim()
    if sound_dim == 3:
         b, c, t = sound.shape
         sound = sound.view(b*c, t)
    elif sound.dim() > 3:
         raise ValueError("sound must be 2D or 3D tensor, but got {}D tensor".format(sound.dim()))
    
    sound_spec = torch.stft(
        sound,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(sound.device),
        onesided=True,
        return_complex=False
    )
    sound_spec = power_compress(sound_spec)
    if sound_dim == 3:
        sound_spec = sound_spec.view(b, c, *sound_spec.shape[1:])
    return sound_spec

def stft2sound(real, imag, n_fft, hop):
    spec_dim = real.dim()
    if spec_dim == 4:
         b, c = real.shape[:2]
         real = real.view(b*c, *real.shape[2:])
         imag = imag.view(b*c, *imag.shape[2:])
    elif real.dim() > 4:
         raise ValueError("sound must be 2D or 3D tensor, but got {}D tensor".format(real.dim()))
    
    spec_uncompress = power_uncompress(real, imag).squeeze(1)
    sound = torch.istft(
        spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(real.device),
        onesided=True,
    )
    if spec_dim == 4:
        sound = sound.view(b, c, -1)
    return sound