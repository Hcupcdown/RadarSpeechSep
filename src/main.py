import os
import warnings

from thop import profile
from torchinfo import summary

from app.sepa_test import FreSepaTester
from app.sepa_train import *
from config import *
from data import *
from model import CleanSpeechSepNet, MossFormer, RadarSpeechSepNet, TSCNet
from utils import *


def main():
    
    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))

    seed_init(1234)

    if args.action == 'train':

        data_loader = build_dataloader(args)

        model = MossFormer(speaker_num=2)

        trainer = TimeSepaTrainer(model, data_loader, args)
        trainer.train()
    else:

        model  = MossFormer(speaker_num=2)

        model.load_state_dict(torch.load("log/23-12-13-16-24-16/model/best_val.pth")["state_dict"])

        tester = TimeSepaTest(model, data_loader, args = args)
        print('---Test score---')
        tester.test()

if __name__ == "__main__":
    main()
