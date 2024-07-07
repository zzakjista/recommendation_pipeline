import sys
import argparse 
from recommender.template import set_template


parser = argparse.ArgumentParser(description='arguments for recommend model')
parser.add_argument('--mode', type=str, default=None, choices=['train', 'inference'], help='mode')
# Service # 
parser.add_argument('--data_code', type=str, default=None, choices=['steam_games', 'amazon_games', 'ml-100k'], help='data_code')
parser.add_argument('--metayn', type=bool, default=None, help='use meta or not')
parser.add_argument('--topk', type=int, default=10, help='recommend topk')
parser.add_argument('--num_users', type=int, default=None, help='num_users')
parser.add_argument('--num_items', type=int, default=None, help='num_items')

# Model #
parser.add_argument('--model_code', type=str, default=None, choices=['autoencoder', 'ease'], help='model')
parser.add_argument('--num_epochs', type=int, default=10, help='epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'bpr'], help='criterion')

# Autoencoder #
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden_size')
parser.add_argument('--latent_dim', type=int, default=32, help='latent_size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')


if 'ipykernel' in sys.modules:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()
set_template(args)
