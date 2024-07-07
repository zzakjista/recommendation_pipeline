from .autoencoder import AutoRec
from .ease import EASE
from .autoencoder_si import AutoRecSi
from .vae import RecVAE

def model_factory(args):
    if args.model_code == 'autoencoder':
        return AutoRec(args)
    elif args.model_code == 'ease':
        return EASE()
    elif args.model_code == 'autoencoder_si':
        return AutoRecSi(args)
    elif args.model_code == 'vae':
        return RecVAE(args)
    else:
        raise ValueError('Unknown model name: {}'.format(args.model_code))