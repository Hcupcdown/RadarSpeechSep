import json
import os
import random

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader


class SeparDataset:


    def __init__(self,
                 dataset_dir,
                 mix_num=2,
                 segment=None,
                 sample_rate=16000,
                 dynamic_mix=True,
                 dynamic_speaker_num=False,
                 pad_to_batch=False,
                 radar=False,
                 mix_type="mix_clean"
                ):
        """
        Initialize the Dataset object.

        Args:
            dataset_dir (str): The directory path of the dataset.
            mix_num (int, optional): The number of audio sources to mix. Defaults to 2.
            segment (float, optional): The duration of each audio segment in seconds. Defaults to None.
            sample_rate (int, optional): The sample rate of the audio. Defaults to 16000.
            dynamic_mix (bool, optional): Whether to dynamically mix different audio sources. Defaults to True.
            dynamic_speaker_num (bool, optional): Whether to dynamically change the number of speakers in each mixture. Defaults to False.
            pad_to_batch (bool, optional): Whether to pad the audio segments to the same speaker only work when dynamic_mix is True. Defaults to False.
            radar (bool, optional): Whether to use radar data. Defaults to False.
            mix_type (str, optional): The type of audio mixture. Defaults to "mix_clean".
        """
        
        self.dataset_dir = dataset_dir
        self.sample_rate  = sample_rate
        self.segment = segment
        self.mix_num = mix_num
        self.dynamic_mix = dynamic_mix
        self.dynamic_speaker_num = dynamic_speaker_num
        self.pad_to_batch = pad_to_batch
        self.radar = radar
        self.mix_type = mix_type

        self.srr = 106.67
        if segment is not None:
            self.segment = int(segment  * sample_rate)
            self.radar_segment = int(self.segment / self.srr)

        self.sample_list = os.listdir(os.path.join(self.dataset_dir, self.mix_type))

    def _segment(self, clean, radar = None):

        sound_len = clean.shape[-1]
        
        if sound_len < self.segment:
            clean = F.pad(clean, (0, self.segment - sound_len))
        else:
            offset = random.randint(0, sound_len - self.segment)
            clean = clean[:, offset : offset+self.segment]
        
        if not self.radar:
            return clean
        
        
        if radar.shape[-1] < self.radar_segment:
            radar = F.pad(radar, (0, self.radar_segment - radar.shape[-1]))
        else:
            radar_offset = int(offset/self.srr)
            radar = radar[..., radar_offset : radar_offset + self.radar_segment]

        return clean, radar
    
    def _segment_batch(self, mix, clean, radar = None):
        sound_len = mix.shape[-1]
        if sound_len < self.segment:
            mix = F.pad(mix, (0, self.segment - sound_len))
            clean = F.pad(clean, (0, self.segment - sound_len))
        else:
            offset = random.randint(0, sound_len - self.segment)
            mix = mix[:, offset: offset+self.segment]
            clean = clean[:, offset: offset+self.segment]
        if not self.radar:
            return mix, clean
        if radar.shape[-1] < self.radar_segment:
            radar = F.pad(radar, (0, self.radar_segment - radar.shape[-1]))
        else:
            radar_offset = int(offset/self.srr)
            end_index = min(radar.shape[-1]-1 , radar_offset + self.radar_segment)
            radar = radar[..., radar_offset : end_index]
            if self.radar_segment - radar.shape[-1] > 0:
                radar = F.pad(radar, (0, self.radar_segment - radar.shape[-1]))
        return mix, clean, radar
        
    def _getitem_static(self, index):

        sample_file = self.sample_list[index]
        mix_audio_file = os.path.join(self.dataset_dir, self.mix_type, sample_file)
        mix_audio, _ = torchaudio.load(mix_audio_file)
        clean_out = []
        radar_out = [] if self.radar else None

        for speaker_id in range(self.mix_num):
            clean_audio_file = os.path.join(self.dataset_dir, 
                                            "s{}".format(speaker_id+1), sample_file)

            clean, _ = torchaudio.load(clean_audio_file)

            if self.radar:
                radar_file = os.path.join(self.dataset_dir, 
                                          "s{}_radar".format(speaker_id+1),
                                          sample_file.replace(".wav", ".npy"))
                radar = torch.tensor(np.load(radar_file),
                                     dtype = torch.float32)

                radar_out.append(radar)

            clean_out.append(clean)
        clean_out = torch.cat(clean_out, dim=0)
        
        if self.radar:
            if radar.dim() == 2:
                radar_out = torch.stack(radar_out, dim=0)
            else:
                radar_out = torch.cat(radar_out, dim=0)

        if self.segment is not None:

            return self._segment_batch(mix=mix_audio, clean=clean_out, radar=radar_out)
        
        if self.radar:
            return mix_audio, clean_out, radar_out
        else:
            return mix_audio, clean_out
    
    def _getitem_dynamic(self, index):

        clean_out = []
        radar_out = [] if self.radar else None

        for speaker_id in range(self.mix_num):
            sample_file = random.choice(self.sample_list)
            clean_audio_file = os.path.join(self.dataset_dir, 
                                            "s{}".format(speaker_id+1), sample_file)

            clean, _ = torchaudio.load(clean_audio_file)
            if self.radar:
                radar_file = os.path.join(self.dataset_dir, 
                                          "s{}_radar".format(speaker_id+1),
                                          sample_file.replace(".wav", ".npy"))
                radar = torch.tensor(np.load(radar_file),
                                     dtype = torch.float32)

                radar_out.append(radar)

            clean_out.append(clean)
        
        clean_out = torch.cat(clean_out, dim=0)
        mix_audio = torch.sum(clean_out, dim=0, keepdim=True)
        if self.radar:
            if radar.dim() == 2:
                radar_out = torch.stack(radar_out, dim=0)
            else:
                radar_out = torch.cat(radar_out, dim=0)

        if self.segment is not None:

            return self._segment_batch(mix=mix_audio, clean=clean_out, radar=radar_out)
        
        if self.radar:
            return mix_audio, clean_out, radar_out
        else:
            return mix_audio, clean_out


    def __getitem__(self, index):
        if self.dynamic_mix:
            return self._getitem_dynamic(index)
        else:
            return self._getitem_static(index)

    def __len__(self):
        return len(self.sample_list)


def radar_collate_fn(batch):
    batch = [x for x in zip(*batch)]
    mix, clean, radar = batch

    return {
        "mix":torch.stack(mix,0),
        "clean":torch.stack(clean,0),
        "radar":torch.stack(radar,0)
        }

def sound_collate_fn(batch):
    batch = [x for x in zip(*batch)]
    mix, clean = batch

    return {
        "mix":torch.stack(mix,0),
        "clean":torch.stack(clean,0)
        }

def build_dataloader(args, train = True):

    if args.radar:
        collate_fn = radar_collate_fn
    else:
        collate_fn = sound_collate_fn
        
    val_dataset   = SeparDataset(args.dataset_dir['val'],
                                 sample_rate= args.dataset['sample_rate'],
                                 dynamic_mix=False,
                                 radar=args.radar,
                                 mix_num=args.dataset['mix_num'],
                                 mix_type=args.dataset['mix_type'])
    
    val_loader    = DataLoader(val_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=args.num_worker,
                               collate_fn=collate_fn)
    dataloader = {"val":val_loader}
    if args.action == "train":
        train_dataset = SeparDataset(args.dataset_dir['train'],
                                     **args.dataset)
    
        train_loader  = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_worker,
                                   collate_fn=collate_fn)
        dataloader = {"train":train_loader, "val":val_loader}
    
    return dataloader