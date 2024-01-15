import os

from app.sepa_test import FreSepaTester
from app.sepa_train import *
from config import *
from data import *
from model import ConvTasNet, MossFormer, RadarMossFormer
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    
    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))

    seed_init(1234)

    if args.action == 'train':

        data_loader = build_dataloader(args)

        # model = RadarMossFormer()
        # trainer = TimeRadarSepaTrainer(model, data_loader, args)
        # model = MossFormer(speaker_num=args.dataset["mix_num"])
        model  = ConvTasNet()
        trainer = TimeSepaTrainer(model, data_loader, args)
        trainer.train()
    else:
        data_loader = build_dataloader(args, train = False)
        model  = RadarMossFormer()
        model.load_state_dict(torch.load("log/24-01-14-21-15-17/model/best_train.pth")["state_dict"])

        tester = TimeRadarSepaTest(model, data_loader, args = args)
        print('---Test score---')
        tester.test()

if __name__ == "__main__":
    main()
