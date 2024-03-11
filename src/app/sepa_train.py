
import os

import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from einops import rearrange
from tqdm import tqdm

from utils import sound2stft, sound_normal, stft2sound
from utils.metric import eval_composite, get_stoi, sisnr
from utils.separate_loss import SeparateLoss

from .train import Trainer


class FreTrainer(Trainer):


    def __init__(self, model, data, n_fft, hop, loss_weights, args):
        super().__init__(model, data, args)
        self.n_fft = n_fft
        self.hop = hop
        self.loss_weights = loss_weights

    def data2stft(self, data):

        noisy = data["noisy"].to(self.args.device)
        clean = data["clean"].to(self.args.device)

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy = noisy * c[:, None]
        if noisy.dim() == clean.dim():
            clean = clean * c[:, None]
        else:
            clean = clean * c[:, None, None]
        
        """[b, c, t] -> [b, c, 2, n_fft, fft_t]"""
        noisy_spec = sound2stft(noisy, self.n_fft, self.hop)
        clean_spec = sound2stft(clean, self.n_fft, self.hop)
        clean_real = clean_spec[:, :, 0, :, :]
        clean_imag = clean_spec[:, :, 1, :, :]

        return {
            "clean_audio":clean,
            "noisy_audio":noisy,
            "noisy_spec":noisy_spec,
            "clean_real":clean_real,
            "clean_imag":clean_imag,
            "c":c
        }
    
    
    def calculate_loss(self, est_real, est_imag, est_audio, clean_real, clean_imag, clean_audio):

        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        min_len = min(est_audio.size(-1), clean_audio.size(-1))
        time_loss = torch.mean(torch.abs(est_audio[...,:min_len] - clean_audio[...,:min_len]))
        
        snr_loss = torch.mean(sisnr(est_audio[...,:min_len], clean_audio[...,:min_len]))
        loss = - snr_loss

        return {
            "loss":loss,
            "loss_ri":loss_ri,
            "loss_mag":loss_mag,
            "time_loss":time_loss,
            "snr_loss":snr_loss
        }

class FreRadarSepaTrainer(FreTrainer):


    def process_data(self, batch_data):
        sound_data = self.data2stft(batch_data)
        radar_data = batch_data["radar"].to(self.args.device)
        radar_data = rearrange(radar_data, "b c h w -> (b c) h w")
        radar_data = radar_data.unsqueeze(1)
        sound_data["radar"] = radar_data
        return sound_data

    def run_batch(self, batch_data):
        data = self.process_data(batch_data)
        noisy_in = data["noisy_spec"].permute(0, 1, 3, 2)
        radar = data["radar"]
        
        est_real, est_imag = self.model(noisy_in, radar)
        est_real = est_real.permute(0, 1, 3, 2)
        est_imag = est_imag.permute(0, 1, 3, 2)
        est_audio = stft2sound(est_real, est_imag, self.n_fft, self.hop)

        loss = self.calculate_loss(est_real=est_real,
                                   est_imag=est_imag,
                                   est_audio=est_audio,
                                   clean_real=data["clean_real"],
                                   clean_imag=data["clean_imag"],
                                   clean_audio=data["clean_audio"])

        c = data["c"]
        est_audio = est_audio / c[:, None, None]

        return loss, est_audio
    

class TimeTrainer(Trainer):

    @staticmethod
    def eval_composite(est_audio, clean_audio, metrics):
        est_audio = est_audio.squeeze(0)
        clean_audio = clean_audio.squeeze(0)
        temp_metrics = eval_composite(clean_audio, est_audio)
        for key in temp_metrics:
            metrics[key] += temp_metrics[key]
        metrics["stoi"] += get_stoi(clean_audio, est_audio, 8000)


    def save_wav(self, sample_id, **kwargs):
        for key, audio in kwargs.items():
            save_path = f"{sample_id}/{key}"
            audio = audio.squeeze(0)
            audio = audio.to(torch.float)
            os.makedirs(os.path.join("result", save_path), exist_ok=True)
            for i in range(audio.shape[0]):
                temp_audio = audio[i].unsqueeze(0).cpu().detach()
                torchaudio.save(os.path.join("result", save_path,f"{i}.wav"), temp_audio, 8000)
    
    def process_data(self, batch_data):
        noisy = batch_data["mix"].to(self.args.device)
        clean = batch_data["clean"].to(self.args.device)
        noisy, _ = sound_normal(noisy)
        clean, _ = sound_normal(clean)
        return {
            "noisy":noisy,
            "clean":clean
        }

    def calculate_loss(self, est_audio, clean_audio):
        est_audio = rearrange(est_audio, "b c t -> (b c) t")
        clean_audio = rearrange(clean_audio, "b c t -> (b c) t")
        mse_loss = F.mse_loss(est_audio, clean_audio)
        snr_loss = torch.mean(sisnr(est_audio, clean_audio))
        loss = -snr_loss

        return {
            "loss":loss,
            "mse_loss":mse_loss,
            "snr_loss":snr_loss
        }

class TimeRadarSepaTrainer(TimeTrainer):

    def __init__(self, model, data, args):
        super().__init__(model, data, args)

        self.loss_fn = lambda x, y:-sisnr(x,y)

    def process_data(self, batch_data):
        data = super().process_data(batch_data)
        radar = batch_data["radar"].to(self.args.device)
        data["radar"] = radar
        return data
    
    def run_batch(self, batch_data):
        data = self.process_data(batch_data)
        # torchaudio.save("noisy.wav", batch_data["mix"][0][0:1], 8000)
        # for i in range(batch_data["clean"].shape[1]):
        #     torchaudio.save(f"clean{i}.wav", batch_data["clean"][0][i:i+1], 8000)
    
        noisy_in = data["noisy"]
        radar_in = data["radar"]
        est_audio = self.model(noisy_in, radar_in)
        sep_loss = torch.mean(self.loss_fn(est_audio, data["clean"]))
        snr = -sep_loss
        loss = {
            "loss":sep_loss,
            "snr":snr
        }

        return loss, est_audio
    
class TimeSepaTrainer(TimeTrainer):
    def __init__(self, model, data, args):
        super().__init__(model, data, args)
        self.sep_loss = SeparateLoss(mix_num=args.dataset["mix_num"],
                                     device=args.device)
        self.loss_fn = lambda x, y:-sisnr(x,y)

    def run_batch(self, batch_data):
        data = self.process_data(batch_data)
        noisy_in = data["noisy"]

        est_audio = self.model(noisy_in)
        sep_loss = self.sep_loss.cal_seploss(est=est_audio,
                                             clean=data["clean"],
                                             loss_fn = self.loss_fn)
        snr = -sep_loss
        loss = {
            "loss":sep_loss,
            "snr":snr
        }
        return loss, est_audio