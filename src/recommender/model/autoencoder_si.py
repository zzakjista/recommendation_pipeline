import torch
from torch import nn
import numpy as np

class AutoRecSi(nn.Module):
    def __init__(self, args):
        super(AutoRecSi, self).__init__()
        self.num_users = args.num_users
        self.num_items = args.num_items
        self.genre_dim = args.genre_dim
        self.age_dim = args.age_dim

        self.input_dim = self.num_items + self.genre_dim + self.age_dim
        self.latent_dim = args.latent_dim 
        self.res_latent_dim = args.latent_dim + self.genre_dim + self.age_dim
        self.hidden_dim = args.hidden_dim

        # side info 
        ## user age
        self.genre_embedding = nn.Embedding(args.num_genres, 5)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.res_latent_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

        self.init_weights()

    def encode(self, input):
        latent = self.encoder(input)
        return latent
    
    def decode(self, latent):
        recont_mat = self.decoder(latent)
        return recont_mat

    def forward(self, mat, **kwargs):
        age = torch.tensor(kwargs['age'])
        genre = torch.tensor(kwargs['genre'])
        genre_emb = self.genre_embedding(kwargs['genre'])
        genre_emb = genre_emb.mean(dim=1) 
        input = torch.cat([mat, genre_emb, age], dim=1)
        latent = self.encode(mat)
        res_latent = torch.cat([latent, genre_emb, age], dim=1)
        recont_mat = self.decoder(res_latent)
        return recont_mat

    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0/(fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0/(fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)


