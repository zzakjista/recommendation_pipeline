def set_template(args):
    
    args.game_name = 'steam_games'
    args.topk = 10
    args.num_items = 0
    args.num_users = 0
    
    # train hyperparameters #
    args.num_epochs = 1
    args.lr = 0.001
    args.weight_decay = 0.0
    args.optimizer = 'adam'
    args.criterion = 'mse'
    args.device = 'cuda'

    # dataloader # 
    args.batch_size = 256

    # model selection # 
    args.model = 'autoencoder'
    
    
    # autoencoder hyperparameters #
    args.hidden_dim = 64
    args.latent_dim = 64

    # ease hyperparameters # 
    args.reg = [100.0]

    args.hidden_dim = 64
    args.latent_dim = 64