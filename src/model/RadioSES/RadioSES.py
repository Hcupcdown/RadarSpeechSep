
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..DPRNN import Decoder, DPRNN_base


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, W=16, N=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        # Components
        # 50% overlap
        self.conv1d_U = nn.Sequential(nn.Conv1d(1, 4*N, kernel_size=W, stride=W // 2, bias=False),
                                      nn.ReLU(),
                                      nn.Conv1d(4*N, N, kernel_size=1, stride=1, bias=False),
                                      nn.ReLU())

    def forward(self, mixture):
        """
        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        mixture_w = self.conv1d_U(mixture)  # [B, N, L]
        return mixture_w

class RadioSES(nn.Module):
    def __init__(self,
                 num_spk:int=2,
                 audio_enc_dim:int=64,
                 radio_enc_dim:int=16,
                 audio_layer:int=2,
                 radio_layer:int=2,
                 fusion_layer:int=4,
                 audio_segment=128,
                 radio_segment=16,
                 win_len=16) -> None:
        super(RadioSES, self).__init__()
        self.num_spk = num_spk
        self.fusion_dim = audio_enc_dim + num_spk * radio_enc_dim
        self.audio_enc_dim = audio_enc_dim

        self.audio_encoder = Encoder(win_len, audio_enc_dim) # [B T]-->[B N L]
        self.radio_encoder = Encoder(win_len, radio_enc_dim) # [B T]-->[B N L]

        self.enc_LN = nn.GroupNorm(1, audio_enc_dim, eps=1e-8) # [B N L]-->[B N L]
        self.audio_dprnn = DPRNN_base(input_dim=audio_enc_dim,
                                      feature_dim=audio_enc_dim,
                                      hidden_dim=audio_enc_dim,
                                      num_spk=1,
                                      layer=audio_layer,
                                      bidirectional=False,
                                      segment_size=audio_segment)

        self.radio_aprnn = DPRNN_base(input_dim=radio_enc_dim,
                                      feature_dim=radio_enc_dim,
                                      hidden_dim=radio_enc_dim,
                                      num_spk=1,
                                      layer=radio_layer,
                                      bidirectional=False,
                                      segment_size=radio_segment)
        
        self.fusion_dprnn = DPRNN_base(input_dim=self.fusion_dim,
                                      feature_dim=self.fusion_dim,
                                      hidden_dim=self.fusion_dim,
                                      num_spk=num_spk,
                                      layer=fusion_layer,
                                      bidirectional=False,
                                      segment_size=audio_segment)

        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.fusion_dim, audio_enc_dim, 1, bias=False)
        self.decoder = Decoder(audio_enc_dim, win_len)

    def forward(self, audio, radio):
        """
        Forward pass of the RadioSES model.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (B, 1, T).
            radio (torch.Tensor): Input radio tensor of shape (B, N, L).

        Returns:
            torch.Tensor: Estimated source tensor of shape (B, nspk, T).
        """
        B, _, _ = audio.size()
        radio = radio.unsqueeze(1)  # [B, L] -> [B, 1, L]
        # Radio
        radio = rearrange(radio, "B N L -> (B N) () L") # B, N, L --> B, L, N
        radio_feature = self.radio_encoder(radio) # B, N, L
        radio_feature = self.radio_aprnn(radio_feature) # B, N, L
        radio_feature = rearrange(radio_feature, "B N L C -> (B N) C L") # B, E, L --> B, L, E

        # audio
        mixture_w = self.audio_encoder(audio)  # B, E, L
        audio_feature = self.enc_LN(mixture_w) # B, E, L
        audio_feature = self.audio_dprnn(audio_feature) # B, E, L
        audio_feature = rearrange(audio_feature, "B N L C -> (B N) C L") # B, E, L --> B, L, E

        # fusion
        radio_feature = F.interpolate(radio_feature,
                                      size=audio_feature.size(-1)) # B, N, L
        
        radio_feature = rearrange(radio_feature, "(B N) C L -> B (N C) L", B = B) # B, N, L

        fusion_feature = torch.cat([audio_feature, radio_feature], dim=1)
        # separate
        score_ = self.fusion_dprnn(fusion_feature)  # B, nspk, T, N
        score_ = score_.view(B*self.num_spk, -1, self.fusion_dim).transpose(1, 2).contiguous()  # B*nspk, N, T
        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]
        score = score.view(B, self.num_spk, self.audio_enc_dim, -1)  # [B*nspk, E, L] -> [B, nspk, E, L]
        est_mask = F.relu(score)

        est_source = self.decoder(mixture_w, est_mask) # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]

        return est_source

if __name__=="__main__":
    test_model = RadioSES()
    audio = torch.randn(2, 1, 8000*3)
    radio = torch.randn(2, 2, 1000*3)
    est_source = test_model(audio, radio)
    print(est_source.shape)
