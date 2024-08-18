from .autoencoder import AutoRec
from .ease import EASE
from .autoencoder_si import AutoRecSi
from .vae import RecVAE

def model_factory(cfg):
    if cfg.model_code == 'autoencoder':
        return AutoRec(cfg)
    elif cfg.model_code == 'ease':
        return EASE()
    elif cfg.model_code == 'autoencoder_si':
        return AutoRecSi(cfg)
    elif cfg.model_code == 'vae':
        return RecVAE(cfg)
    else:
        raise ValueError('Unknown model name: {}'.format(cfg.model_code))