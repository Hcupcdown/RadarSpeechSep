import os

import yaml

from .config import get_config


def get_args_dict():
    """
    Get your arguments and make dictionary.
    If you add some arguments in the model, you should edit here also.
    """
    speaker_dict = {"ConvTasNet":"C",
                    "DPRNN":"nspk",
                    "MossFormer":"speaker_num",
                    "RadioSES":"num_spk"}
    args = get_config()
    args.dataset_dir = {'train':args.train,'val':args.val,'test':args.test}
    args.dataset = {'sample_rate':args.sample_rate,
                    'segment':args.segment,
                    'mix_num':args.mix_num,
                    'dynamic_mix':args.dynamic_mix,
                    'dynamic_speaker_num':args.dynamic_speaker_num,
                    'pad_to_batch':args.pad_to_batch,
                    'mix_type':args.mix_type}

    args.ex_name = os.getcwd().replace('\\','/').split('/')[-1]
    model_config_path = os.path.join('src', 'config', 'model_config', args.model+'.yml')
    with open(model_config_path) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    args.model_config = model_config
    if args.model in speaker_dict.keys():
        args.model_config[speaker_dict[args.model]] = args.mix_num

    return args