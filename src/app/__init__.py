import torch

from .sepa_test import TimeRadarSepaTest, TimeSepaTest
from .sepa_train import TimeRadarSepaTrainer, TimeSepaTrainer


def build_trainer(args, model, data):
    timemodels = ['ConvTasNet', 'DPRNN', 'MossFormer']
    if args.model in timemodels:
        return TimeSepaTrainer(model, data, args)
    elif args.model == 'RadarMossFormer' or args.model == 'RadioSES':
        return TimeRadarSepaTrainer(model, data, args)
    else:
        raise NotImplementedError

def build_tester(args, model, data):
    model.load_state_dict(torch.load(args.model_path)["state_dict"])
    timemodels = ['ConvTasNet', 'DPRNN', 'MossFormer']
    if args.model in timemodels:
        return TimeSepaTest(model, data, args)
    elif args.model == 'RadarMossFormer':
        return TimeRadarSepaTest(model, data, args)
    else:
        raise NotImplementedError