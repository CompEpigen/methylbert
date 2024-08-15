from collections import OrderedDict
from transformers import BertConfig

METHYLBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hanyangii/methylbert_hg19_12l": "https://huggingface.co/hanyangii/methylbert_hg19_12l/raw/main/config.json",
    "hanyangii/methylbert_hg19_8l": "https://huggingface.co/hanyangii/methylbert_hg19_8l/raw/main/config.json",
    "hanyangii/methylbert_hg19_6l": "https://huggingface.co/hanyangii/methylbert_hg19_6l/raw/main/config.json",
    "hanyangii/methylbert_hg19_4l": "https://huggingface.co/hanyangii/methylbert_hg19_4l/raw/main/config.json",
    "hanyangii/methylbert_hg19_2l": "https://huggingface.co/hanyangii/methylbert_hg19_2l/raw/main/config.json"
}

class MethylBERTConfig(BertConfig):
    pretrained_config_archive_map = METHYLBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    loss="bce"

class Config(object):
    def __init__(self, config_dict: dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def get_config(**kwargs):
    '''
    Create a Config object for configuration from input
    '''
    config = OrderedDict(
          [
            ('lr', 1e-4),
            ('beta', (0.9, 0.999)),
            ('weight_decay', 0.01),
            ('warmup_step', 10000),
            ('eps', 1e-6),
            ('with_cuda', True),
            ('log_freq', 10),
            ('eval_freq', 10),
            ('n_hidden', None),
            ("decrease_steps", 200),
            ('eval', False),
            ('amp', False),
            ("gradient_accumulation_steps", 1), 
            ("max_grad_norm", 1.0),
            ("eval", False),
            ("save_freq", None),
            ("loss", "bce")
          ]
        )

    if kwargs is not None:
        for key in config.keys():
            if key in kwargs.keys():
                config[key] = kwargs.pop(key)

    return Config(config)