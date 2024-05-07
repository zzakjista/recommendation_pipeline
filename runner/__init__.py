from .trainer import AERunner

def runner_factory(model, dataloader, optimizer, criterion, lr, device, dataset, scheduler=None):
    return AERunner(model, dataloader, optimizer, criterion, lr, device, dataset, scheduler=None)