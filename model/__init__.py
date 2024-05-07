from .autoencoder import AutoRec

def model_factory(model_name:str, input_dim:int, hidden_dim:int):
    if model_name == 'AutoRec':
        return AutoRec(input_dim, hidden_dim)
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))