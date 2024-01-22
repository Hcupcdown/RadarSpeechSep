from .ConvTasNet import ConvTasNet
from .DPRNN import DPRNN_sep
from .FusionNet import RadarMossFormer
from .Mossformer import MossFormer
from .RadioSES import RadioSES


def bulid_model(args):
    if args.model == 'ConvTasNet':
        model = ConvTasNet(C = args.dataset["mix_num"])
    elif args.model == 'DPRNN':
        model = DPRNN_sep(nspk=args.dataset["mix_num"])
    elif args.model == 'MossFormer':
        print("MossFormer")
        model = MossFormer(speaker_num=args.dataset["mix_num"])
    elif args.model == 'RadarMossFormer':
        model = RadarMossFormer()
    elif args.model == 'RadioSES':
        model = RadioSES()
    else:
        raise NotImplementedError
    return model