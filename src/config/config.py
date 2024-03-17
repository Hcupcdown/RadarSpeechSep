import argparse
import os

import yaml


def get_config():

    parser = argparse.ArgumentParser()

    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    parser.add_argument('--model', type=str, default='RadarMossFormer', help='DPRNN, ConvTasNet, MossFormer, RadioSES, RadarMossFormer')
    # dataset
    parser.add_argument('--train', type=str, default=r'G:\my_radar_sound\mini_data\unseen\2mix\train', help='Train path')
    parser.add_argument('--val', type=str, default=r'G:\my_radar_sound\mini_data\unseen\2mix\test', help='Val path')
    parser.add_argument('--test', type=str, default=r'G:\my_radar_sound\mini_data\unseen\2mix\test', help='Test path')
    parser.add_argument('--sample_rate', type=int, default=8000, help='Sample rate')
    parser.add_argument('--segment', type=int, default=6, help='Segment') # segment signal per 2 seconds
    parser.add_argument('--mix_num', type=int, default=1, help='Mix num')
    parser.add_argument('--dynamic_mix', type=bool, default=False, help='Dynamic mix')
    parser.add_argument('--dynamic_speaker_num', type=bool, default=True, help='Dynamic speaker num')
    parser.add_argument('--pad_to_batch', type=bool, default=True, help='Pad to batch')
    parser.add_argument('--mix_type', type=str, default='s1_noise', help='Mix type') # mix_clean / mix_both

    #basic 
    parser.add_argument('--model_path', type=str, default='log/24-03-04-16-26-23/model/best_val.pth', help='Model path')
    parser.add_argument('--learning_rate', type=float, default=15e-5, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Checkpoint') # If you want to train with pre-trained, or resume set True

    # device 
    parser.add_argument('--device', type=str, default='cuda:0', help='Gpu device')
    parser.add_argument('--env', type=str, default='local', help='Enviornment')
    parser.add_argument('--num_worker', type=int, default=4, help='Num workers')

    parser.add_argument("--val_per_epoch", type=int, default=1, help="")
    arguments = parser.parse_args()

    return arguments