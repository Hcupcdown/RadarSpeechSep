import json
import os
import random

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F


class SeparDataset:


    def __init__(self,
                     dataset_dir,
                     mix_num=5,
                     segment=None,
                     sample_rate=16000,
                     dynamic_mix=True):
            """
            Initialize the Dataset object.

            Args:
                dataset_dir (str): The directory path of the dataset.
                mix_num (int, optional): The number of audio clips to mix. Defaults to 5.
                segment (float, optional): The duration of each audio segment in seconds. Defaults to 2.
                sample_rate (int, optional): The sample rate of the audio clips. Defaults to None.
                dynamic_mix (bool, optional): Whether to dynamically generate the data list. Defaults to True.
            """
            
            self.dataset_dir = dataset_dir
            self.sample_rate  = sample_rate
            self.segment = segment
            self.srr = 180.
            if segment is not None:
                self.segment = int(segment  * sample_rate)
                self.radar_segment = int(self.segment // self.srr)

            self.dynamic_mix = dynamic_mix
            if dynamic_mix:
                self._gen_data_list_dynamic()
            else:
                self._gen_data_list_static()
                
            self.mix_num = mix_num
        
    def _gen_data_list_dynamic(self):

        file_list = os.listdir(os.path.join(self.dataset_dir, 'clean'))
        self.clean_list = []
        self.radar_list = []
        
        for i in range(len(file_list)):
            self.clean_list.append(os.path.join(self.dataset_dir, 'clean', file_list[i]))
            self.radar_list.append(os.path.join(self.dataset_dir, 'radar', file_list[i].replace('.wav', '.npy')))

    def _gen_data_list_static(self):

        data_list = os.listdir(self.dataset_dir)
        self.mix_list = []
        self.clean_list = []
        self.radar_list = []
        for sample_name in data_list:
            temp_cleans = []
            temp_radars = []
            self.mix_list.append(os.path.join(self.dataset_dir, sample_name, "noise.wav"))
            cleans = os.listdir(os.path.join(self.dataset_dir, sample_name, "clean"))
            radars = os.listdir(os.path.join(self.dataset_dir, sample_name, "radar"))
            for clean_file, radar_file in zip(cleans, radars):
                temp_cleans.append(os.path.join(self.dataset_dir, sample_name, "clean", clean_file))
                temp_radars.append(os.path.join(self.dataset_dir, sample_name, "radar", radar_file))
            self.clean_list.append(temp_cleans)
            self.radar_list.append(temp_radars)
    
    def _segment(self, clean, radar):
            """
            Segments the clean and radar data based on the specified segment length.
            
            Args:
                clean (torch.Tensor): The clean audio data.
                radar (torch.Tensor): The radar data.
            
            Returns:
                tuple: A tuple containing the segmented clean and radar data.
            """
            if self.segment is None:
                return clean, radar
            sound_len = clean.shape[-1]
            
            #如果长度小于段长度，填充0
            if sound_len < self.segment:
                clean = F.pad(clean, (0, self.segment - sound_len))
                radar = radar
            #否则截取一段
            else:
                offset = random.randint(0, sound_len - self.segment)
                radar_offset = int(offset/self.srr)
                clean = clean[:, offset: offset+self.segment]
                radar = radar[:,radar_offset: radar_offset+self.radar_segment]
            
            if radar.shape[-1] < self.radar_segment:
                radar = F.pad(radar, (0, self.radar_segment - radar.shape[-1]))
            elif radar.shape[-1] > self.radar_segment:
                radar = radar[:,:self.radar_segment]
            return clean, radar
    
    def _getitem_static(self, index):

        clean_files = self.clean_list[index]
        radar_files = self.radar_list[index] 
        clean_out = []
        radar_out = []

        for clean_file, radar_file in zip(clean_files, radar_files):
            clean, _  = torchaudio.load(clean_file)
            radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
            clean, radar = self._segment(clean=clean, radar=radar)
            
            clean_out.append(clean)
            radar_out.append(radar)

        clean_out = torch.cat(clean_out, dim=0)
        radar_out = torch.stack(radar_out, dim=0)
        return torch.sum(clean_out, dim=0), clean_out, radar_out

    def _getitem_dynamic(self, index):

        sample_num = random.randint(2, self.mix_num)
        clean_out = []
        radar_out = []
        for i in range(sample_num):
            index = random.randint(0, len(self.clean_list)-1)
            clean_file = self.clean_list[index]
            radar_file = self.radar_list[index]
            clean, _  = torchaudio.load(clean_file)
            radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
            file_length = clean.shape[-1]
    
            #如果长度小于段长度，填充0
            if file_length < self.segment:
                clean = F.pad(clean, (0, self.segment - file_length))
                radar = radar
            #否则截取一段
            else:
                offset = random.randint(0, file_length - self.segment)
                radar_offset = int(offset/self.srr)
                clean = clean[:, offset: offset+self.segment]
                radar = radar[:,radar_offset: radar_offset+self.radar_segment]
            
            if radar.shape[1] < self.radar_segment:
                radar = F.pad(radar, (0, self.radar_segment - radar.shape[1]))
            elif radar.shape[1] > self.radar_segment:
                radar = radar[:,:self.radar_segment]

            clean_out.append(clean)
            radar_out.append(radar)

        clean_out = torch.cat(clean_out, dim=0)
        radar_out = torch.stack(radar_out, dim=0)

        if sample_num < self.mix_num:
            clean_out = F.pad(clean_out, (0,0,0, self.mix_num - sample_num))
            radar_out = F.pad(radar_out, (0,0,0,0,0, self.mix_num - sample_num))
        
        return torch.sum(clean_out, dim=0), clean_out, radar_out


    def __getitem__(self, index):
        if self.dynamic_mix:
            return self._getitem_dynamic(index)
        else:
            return self._getitem_static(index)

    def __len__(self):
        return len(self.clean_list)


def collate_fn(batch):
    batch = [x for x in zip(*batch)]
    mix, clean, radar = batch

    return {
        "noisy":torch.stack(mix,0),
        "clean":torch.stack(clean,0),
        "radar":torch.stack(radar,0)}