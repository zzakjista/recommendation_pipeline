from .trainer import AERunner, EASERunner

def runner_factory(model, dataloader, cfg):
    if cfg.model_code == 'autoencoder':
        return AERunner(model, dataloader, cfg)
    elif cfg.model_code == 'ease':
        return EASERunner(model, dataloader, cfg)
    elif cfg.model_code == 'autoencoder_si':
        return AERunner(model, dataloader, cfg)
    elif cfg.model_code == 'vae':
        return AERunner(model, dataloader, cfg)
    else:
        raise ValueError('Unknown model name: {}'.format(cfg.model_code))