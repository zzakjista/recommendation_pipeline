from .dataloader import PytorchDataLoader, MatrixDataLoader

def dataloader_factory(dataset, args):
    if args.model_code == 'autoencoder':
        return PytorchDataLoader(dataset, args)
    elif args.model_code == 'ease':
        return MatrixDataLoader(dataset, args)
    elif args.model_code == 'autoencoder_si':
        return PytorchDataLoader(dataset, args)
    elif args.model_code == 'vae':
        return PytorchDataLoader(dataset, args)
    else:
        raise ValueError(f"model_code {args.model_code} is not supported")


    
