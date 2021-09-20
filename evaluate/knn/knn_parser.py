import configargparse

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def tnse_parse_config(config_path=None, 
                        verbose=False):
        
    parser = configargparse.ArgumentParser(description='PyTorch ImageNet Training',
                                           default_config_files=[config_path])
    parser.add_argument('--train-data', metavar='DIR',
                        help='path to train dataset')
    parser.add_argument('--test-data', metavar='DIR',
                        help='path to test dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4096/8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    # additional configs:
    parser.add_argument('--model-path', default='', type=str,
                        help='path to simsiam model path')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='checkpoint filename')
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    
    args = parser.parse_args()
    
    if verbose:
        print(args)
    return args