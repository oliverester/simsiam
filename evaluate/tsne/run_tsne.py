#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from evaluate.tsne.tsne_parser import tnse_parse_config
import simsiam.builder
import tracking

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def run_tsne(config_path):
    args = tnse_parse_config(config_path=config_path,
                               verbose=True)
    
    tracking.log_config(args.model_path, config_path)
      
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # Simply call main_worker function
    main_worker(args.gpu, args)


def main_worker(gpu,args):
    
    args.gpu = gpu
    args.device = 'cuda'
    
    torch.cuda.set_device(args.gpu)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    # get model but without predictor (we want the features)
    model = simsiam.builder.SimSiam(models.__dict__[args.arch],
                                    args.dim,
                                    inference=True)

    # load from pre-trained, before DistributedDataParallel constructor

    args.pretrained = os.path.join(args.model_path, args.checkpoint)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder'): # and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                else:
                    print(f"Remove layer {k}")
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == set([])

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(args.pretrained))

    model.cuda(args.gpu)

    print(model)
    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.test_data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    embeddings, labels = get_embeddings(val_loader, model, args)
    visualize_tsne(embeddings=embeddings, 
                   save_to=os.path.join(args.model_path, 'tnse_plot.jpg'), 
                   labels=labels)
    

def visualize_tsne(embeddings, save_to, labels=None, colors_per_class=None):
    
    if colors_per_class is None:
        colors_per_class = {0: [[1, 0.5, 0.5]],
                            1: [[0.5, 1, 1]]}
    if labels is None:
        labels = np.array([0]*len(embeddings))
    
    tsne = TSNE(n_components=2).fit_transform(embeddings)
         
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
        
        # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = colors_per_class[label]

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    print(f"Saving tnse result to {save_to}")
    plt.savefig(save_to)

    # scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def get_embeddings(val_loader, model, args):
   
    print("Determine tsne embeddings...")
    embeddings = torch.zeros(size=(len(val_loader.dataset), args.dim), dtype=torch.float32, device=args.device)
    labels = torch.zeros(size=(len(val_loader.dataset),), dtype=torch.int, device=args.device)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            embeddings[i*int(args.batch_size) : i*int(args.batch_size)+images.size(0)] = output
            labels[i*int(args.batch_size) : i*int(args.batch_size)+images.size(0)] = targets

    return embeddings.cpu().numpy(), labels.cpu().numpy()