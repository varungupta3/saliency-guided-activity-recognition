import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

# Training settings
parser = argparse.ArgumentParser(description='SALGAN - Pytorch')

parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--test_batchsize', type=int, default=32, help='input batch size for testing (default: 32)')

parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')

parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 0.0003)')
parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate (default: 0.00001)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, help='optimizer momentum (default: 0.5)')
parser.add_argument('--alpha', type=float, default=0.05, help='alpha (default: 0.05)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=4, help='how many batches to wait before logging training status')

parser.add_argument('--plot_saliency', type=str2bool, default=False, help='boolean to plot true and predicted saliency maps (default: False)')
parser.add_argument('--predict_saliency', type=str2bool, default=False, help='boolean to use true saliency or predict saliency maps (default: False)')

