import json
import os
import random

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F


class TrainDataset:


    def __init__(self, dataset_dir, segment=2, sample_rate=None):
        self.dataset_dir = dataset_dir
        self.sample_rate  = sample_rate
        
        self.segment = int(segment  * sample_rate)
        self.srr = 180.
        self.radar_segment = int(self.segment // self.srr)
        self._gen_data_list()        
        self.id_flag = 0
        
    def _gen_data_list(self):
        file_list = os.listdir(os.path.join(self.dataset_dir, 'clean'))
        self.noisy_list = []
        self.clean_list = []
        self.radar_list = []
        self.mask_list = []
        for i in range(len(file_list)):
            self.clean_list.append(os.path.join(self.dataset_dir, 'clean', file_list[i]))
            self.noisy_list.append(os.path.join(self.dataset_dir, 'noise', file_list[i]))
            self.radar_list.append(os.path.join(self.dataset_dir, 'radar', file_list[i].replace('.wav', '.npy')))
            self.mask_list.append(os.path.join(self.dataset_dir, 'mask', file_list[i].replace('.wav', '.npy')))
        
    def __getitem__(self, index):
        clean_file = self.clean_list[index]
        noisy_file = self.noisy_list[index]
        radar_file = self.radar_list[index]
        mask_file = self.mask_list[index]
        clean, _  = torchaudio.load(clean_file)
        noisy, _    = torchaudio.load(noisy_file)
        radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
        mask = torch.tensor(np.load(mask_file), dtype = torch.float32).unsqueeze(0)
        file_length = clean.shape[1]

        #如果长度小于段长度，填充0
        if file_length < self.segment:
            clean_out = F.pad(clean, (0, self.segment - file_length))
            noisy_out = F.pad(noisy, (0, self.segment - file_length))
            mask_out = F.pad(mask, (0, self.segment - file_length))
            radar_out = radar
        #否则截取一段
        else:
            index = random.randint(0, file_length - self.segment)
            radar_index = int(index/self.srr)
            clean_out = clean[:, index: index+self.segment]
            noisy_out = noisy[:, index: index+self.segment]
            mask_out = mask[:, index: index+self.segment]
            radar_out = radar[:,radar_index: radar_index+self.radar_segment]
        
        if radar_out.shape[1] < self.radar_segment:
            radar_out = F.pad(radar_out, (0, self.radar_segment - radar_out.shape[1]))
        elif radar_out.shape[1] > self.radar_segment:
            radar_out = radar_out[:,:self.radar_segment]

        return noisy_out, clean_out, radar_out, mask_out

    def __len__(self):
        return len(self.noisy_list)
    
class TestDataset:


    def __init__(self, dataset_dir, sample_rate=None):

        self.dataset_dir = dataset_dir
        clean, noisy, self.radar_list, self.mask_list = self._gen_data_list()
        self.clean_set = Audioset(clean, sample_rate = sample_rate)
        self.noisy_set = Audioset(noisy, sample_rate = sample_rate)

        assert len(self.clean_set) == len(self.noisy_set)

    def _gen_data_list(self):

        file_list = os.listdir(os.path.join(self.dataset_dir, 'clean'))
        noisy_list = []
        clean_list = []
        radar_list = []
        mask_list = []
        for i in range(len(file_list)):
            clean_list.append(os.path.join(self.dataset_dir, 'clean', file_list[i]))
            noisy_list.append(os.path.join(self.dataset_dir, 'noise', file_list[i]))
            radar_list.append(os.path.join(self.dataset_dir, 'radar', file_list[i].replace('.wav', '.npy')))
            mask_list.append(os.path.join(self.dataset_dir, 'mask', file_list[i].replace('.wav', '.npy')))
        return clean_list, noisy_list, radar_list, mask_list
    

    def __getitem__(self, index):

        radar_path = self.radar_list[index]
        mask_path = self.mask_list[index]

        radar = np.load(radar_path)
        mask = np.load(mask_path)

        radar = torch.tensor(radar, dtype = torch.float32)
        mask = torch.tensor(mask, dtype = torch.float32).unsqueeze(0)

        return self.noisy_set[index], self.clean_set[index], radar, mask

    def __len__(self):

        return len(self.noisy_set)

class Audioset:


    def __init__(self, files=None, sample_rate=None):
        """
        files should be a list [(file, length)]
        """

        self.files        = files
        self.sample_rate  = sample_rate

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):
        
        file = self.files[index]
        out, sr    = torchaudio.load(file)
        if self.sample_rate is not None:
            if sr != self.sample_rate:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                   f"{self.sample_rate}, but got {sr}")
        return out


class SeparTrainDataset:


    def __init__(self,
                     dataset_dir,
                     mix_num=5,
                     segment=2,
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
    
    def _getitem_static(self, index):

        mix_file = self.mix_list[index]
        clean_files = self.clean_list[index]
        radar_files = self.radar_list[index] 
        mix, _  = torchaudio.load(mix_file)
        clean_out = []
        radar_out = []

        for clean_file, radar_file in zip(clean_files, radar_files):
            clean, _  = torchaudio.load(clean_file)
            radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
            clean_out.append(clean)
            radar_out.append(radar)

        clean_out = torch.cat(clean_out, dim=0)
        radar_out = torch.stack(radar_out, dim=0)
        return mix.squeeze(0), clean_out, radar_out
    
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
        sample_num = random.randint(2, self.mix_num)
        clean_out = []
        radar_out = []
        for i in range(sample_num):
            index = random.randint(0, len(self.clean_list)-1)
            clean_file = self.clean_list[index]
            radar_file = self.radar_list[index]
            clean, _  = torchaudio.load(clean_file)
            radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
            file_length = clean.shape[1]
    
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

    def __len__(self):
        return len(self.clean_list)


class SeparTestDataset:

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self._gen_data_list()

    def _gen_data_list(self):
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

    def __getitem__(self, index):
        mix_file = self.mix_list[index]
        clean_files = self.clean_list[index]
        radar_files = self.radar_list[index] 
        mix, _  = torchaudio.load(mix_file)
        clean_out = []
        radar_out = []
        for clean_file, radar_file in zip(clean_files, radar_files):
            clean, _  = torchaudio.load(clean_file)
            radar = torch.tensor(np.load(radar_file), dtype = torch.float32)
            clean_out.append(clean)
            radar_out.append(radar)
        clean_out = torch.cat(clean_out, dim=0)
        radar_out = torch.stack(radar_out, dim=0)
        return mix.squeeze(0), clean_out, radar_out

    def __len__(self):
        return len(self.clean_list)

def collate_fn(batch):
    batch = [x for x in zip(*batch)]
    mix, clean, radar = batch

    return {
        "noisy":torch.stack(mix,0),
        "clean":torch.stack(clean,0),
        "radar":torch.stack(radar,0)}

if __name__=="__main__":
    test_data = SeparTestDataset(r"/home/han_dc/speech_separation/radar_sound_dataset/test")
    noisy, clean, radar = test_data[0]
    print(noisy.shape)
    print(clean.shape)
    print(radar.shape)