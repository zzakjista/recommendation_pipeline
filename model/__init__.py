from .autoencoder import AutoRec
from .ease import EASE
def model_factory(args):
    if args.model == 'autoencoder':
        return AutoRec(args)
    elif args.model == 'ease':
        return EASE()
    else:
        raise ValueError('Unknown model name: {}'.format(args.model))