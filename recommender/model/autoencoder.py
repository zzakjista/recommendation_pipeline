from torch import nn
import numpy as np

class AutoRec(nn.Module):
    def __init__(self, args):
        super(AutoRec, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.num_users = args.num_users
        self.num_items = args.num_items

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_dim),
            # nn.Sigmoid(),
            # nn.Linear(self.hidden_dim, self.hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            # nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            # nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.num_items),
        )

        self.init_weights()

    def forward(self, mat):
        latent = self.encoder(mat)
        recont_mat = self.decoder(latent)

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


