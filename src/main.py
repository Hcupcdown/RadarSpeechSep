import os

from app import build_tester, build_trainer
from config import *
from data import *
from model import bulid_model
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    
    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))
    seed_init(1234)
    data_loader = build_dataloader(args)
    model = bulid_model(args)
    
    if args.action == 'train':
        trainer = build_trainer(args, model, data_loader)
        trainer.train()
    else:
        tester = build_tester(args, model, data_loader)
        tester.test()

if __name__ == "__main__":
    main()
