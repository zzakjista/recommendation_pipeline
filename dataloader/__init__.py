from .dataloader import PytorchDataLoader, MatrixDataLoader

def dataloader_factory(dataset, args):
    if args.model == 'autoencoder':
        return PytorchDataLoader(dataset, args)
    elif args.model == 'ease':
        return MatrixDataLoader(dataset, args)
    else:
        raise ValueError(f"model {args.model} is not supported")


    
