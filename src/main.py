import os
import warnings

from thop import profile
from torch.utils.data import DataLoader
from torchinfo import summary

from app.sepa_test import FreSepaTester
from app.sepa_train import *
from config import *
from data.dataset import *
from model import CleanSpeechSepNet, MossFormer, RadarSpeechSepNet, TSCNet
from utils import *


def main():
    
    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))

    seed_init(1234)

    if args.action == 'train':

        train_dataset = SeparDataset(args.dataset['train'],
                                          segment=args.setting['segment'],
                                          sample_rate= args.setting['sample_rate'],
                                          dynamic_mix=False)

        train_loader  = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_worker,
                                   collate_fn=collate_fn)
         
        val_dataset   = SeparDataset(args.dataset['val'],
                                     sample_rate= args.setting['sample_rate'],
                                     dynamic_mix=False)
        
        val_loader    = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_worker, collate_fn=collate_fn)
        data_loader   = {'train':train_loader, 'val':val_loader}
        # model = TSCNet(num_channel=16, num_features=513, clean_mask = False)
        # trainer = FreRadarSepaTrainer(model, data_loader, n_fft=1024, hop=256, loss_weights=[0.1,0.9,0.9,0.2], args = args)
        model = MossFormer(speaker_num=2)
        input = torch.randn(1, 1, 32000)
        Flops, Params = profile(model, inputs=(input, ))

        print(f"Flops: {Flops/1e9}G", f"Params: {Params/1e6}M")

        trainer = TimeSepaTrainer(model, data_loader, args)
        trainer.train()
    else:
        val_dataset   = SeparDataset(args.dataset['val'])
        val_loader    = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_worker, collate_fn=collate_fn)
        model  = TSCNet(num_channel=16, num_features=513, clean_mask = False)
        model.load_state_dict(torch.load("log/23-12-06-15-27-13/model/best_train.pth")["state_dict"])

        tester = FreSepaTester(model, val_loader, n_fft=1024, hop=256, args = args)
        print('---Test score---')
        tester.test()

if __name__ == "__main__":
    main()
