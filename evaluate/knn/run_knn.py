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
     
    args.pretrained = os.path.join(args.model_path, args.checkpoint)

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
    """
    KNN classifier establishes a "model" by determine the training-embeddings.
    We also need the true classes for these samples (might also work for semi-supervised)
    Eval-embeddings are then matched via KNN.

    Args:
        gpu ([type]): [description]
        args ([type]): [description]
    """
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
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model.cuda(args.gpu)

    print(model)
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.train_data)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    embeddings, labels = get_embeddings(val_loader, model, args)
    
    knn_model = build_knn_model(embeddings, labels, k=5)
    
     # Data loading code
    testdir = os.path.join(args.test_data)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_embeddings, labels = get_embeddings(test_loader, model, args)
    
    # predict class for test embeddings unsing knnn model
    #accuary = ..
    

def build_knn_model(embeddings, labels, k=5):
    pass


def get_embeddings(val_loader, model, args):
   
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