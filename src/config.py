import argparse
import os


def get_config():

    parser = argparse.ArgumentParser()

    parser.add_argument('action', type=str, default='train', help='Action') # train / test

    # dataset
    parser.add_argument('--train', type=str, default=r'F:\LibriMix\Libri2Mix\wav8k\max\train-100', help='Train path')
    parser.add_argument('--val', type=str, default=r'F:\LibriMix\Libri2Mix\wav8k\max\dev', help='Val path')
    parser.add_argument('--test', type=str, default=r'F:\LibriMix\Libri2Mix\wav8k\max\test', help='Test path')
    parser.add_argument('--sample_rate', type=int, default=8000, help='Sample rate')
    parser.add_argument('--segment', type=int, default=4, help='Segment') # segment signal per 2 seconds
    parser.add_argument('--mix_num', type=int, default=2, help='Mix num')
    parser.add_argument('--dynamic_mix', type=bool, default=True, help='Dynamic mix')
    parser.add_argument('--dynamic_speaker_num', type=bool, default=True, help='Dynamic speaker num')
    parser.add_argument('--pad_to_batch', type=bool, default=True, help='Pad to batch')
    parser.add_argument('--radar', type=bool, default=False, help='Radar')
    parser.add_argument('--mix_type', type=str, default='mix_clean', help='Mix type') # mix_clean / mix_both

    #basic 
    parser.add_argument('--model_path', type=str, default='log/23-12-12-17-03-58/model/temp_train.pth', help='Model path')
    parser.add_argument('--learning_rate', type=float, default=15e-5, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=300, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Checkpoint') # If you want to train with pre-trained, or resume set True

    # device 
    parser.add_argument('--device', type=str, default='cuda:0', help='Gpu device')
    parser.add_argument('--env', type=str, default='local', help='Enviornment')
    parser.add_argument('--num_worker', type=int, default=4, help='Num workers')

    parser.add_argument("--val_per_epoch", type=int, default=5, help="")
    arguments = parser.parse_args()

    return arguments
