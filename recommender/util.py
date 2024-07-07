import bentoml

def save_bento_model(model, args):
    """
    BentoML을 활용하여 모델을 저장하는 함수
    :input: 
        - model
        - args
        - vocab
    :return:
        - None
    :output:
        - BentoService
    """
    if args.model_code == 'autoencoder':
        model_tag = bentoml.pytorch.save_model(args.model_code, model)
    elif args.model_code == 'ease':
        model_tag = bentoml.picklable.save_model(args.model_code, model)
    else:
        raise ValueError(f'Not supported model_code: {args.model_code}')
    return model_tag
    
    

