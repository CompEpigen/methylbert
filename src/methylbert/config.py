from collections import OrderedDict

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
            #("methyl_learning", "cnn"),
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