from .ConvTasNet import ConvTasNet
from .DPRNN import DPRNN_sep
from .Mossformer import MossFormer
from .RadarMossFormer import RadarMossFormer
from .RadioSES import RadioSES


def bulid_model(args):
    if args.model == 'ConvTasNet':
        model = ConvTasNet(**args.model_config)
    elif args.model == 'DPRNN':
        model = DPRNN_sep(**args.model_config)
    elif args.model == 'MossFormer':
        print("MossFormer")
        model = MossFormer(**args.model_config)
    elif args.model == 'RadarMossFormer':
        model = RadarMossFormer(**args.model_config)
    elif args.model == 'RadioSES':
        model = RadioSES(**args.model_config)
    else:
        raise NotImplementedError
    return model