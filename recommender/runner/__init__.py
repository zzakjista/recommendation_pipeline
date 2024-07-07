from .trainer import AERunner, EASERunner

def runner_factory(model, dataloader, args):
    if args.model_code == 'autoencoder':
        return AERunner(model, dataloader, args)
    elif args.model_code == 'ease':
        return EASERunner(model, dataloader, args)
    elif args.model_code == 'autoencoder_si':
        return AERunner(model, dataloader, args)
    elif args.model_code == 'vae':
        return AERunner(model, dataloader, args)
    else:
        raise ValueError('Unknown model name: {}'.format(args.model_code))