from .trainer import AERunner, EASERunner

def runner_factory(model, dataloader, args):
    if args.model == 'autoencoder':
        return AERunner(model, dataloader, args)
    elif args.model == 'ease':
        return EASERunner(model, dataloader, args)
    else:
        raise ValueError('Unknown model name: {}'.format(args.model))