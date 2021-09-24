import configargparse

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def knn_parse_config(config_path=None, 
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
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-b', '--batch-size', default=4096/8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')

    # additional configs:
    parser.add_argument('--model-path', default='', type=str,
                        help='path to simsiam model path')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='checkpoint filename')
    parser.add_argument('--k', default=5, type=int,
                        help='number of k for KNN (default: 5)')
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    
    args = parser.parse_args()
    
    if verbose:
        print(args)
    return args