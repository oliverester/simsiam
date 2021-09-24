#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from simsiam.data_provider import DataProvider
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

from plotnine import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from simsiam.evaluate.knn.knn_parser import knn_parse_config
import simsiam.builder
import tracking


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def run_knn(config_path):
    args = knn_parse_config(config_path=config_path,
                               verbose=True)

    tracking.log_config(args.model_path, config_path)
     
    args.pretrained = os.path.join(args.model_path, args.checkpoint)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # Simply call main_worker function
    main_worker(args.gpu, args)


def main_worker(gpu, args):
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
    data_provider = DataProvider(args=args)
    train_loader, _ = data_provider.get_train_loader(aug=False)
    
    print("Determine train embeddings..")
    embeddings, labels = get_embeddings(train_loader, model, args)
    
    knn_classifier = build_knn_model(embeddings, labels, k=5)
    print("Provide KNN-classifier")
    
     # Data loading code
    val_loader = data_provider.get_val_loader()

    print("Determine validation embeddings..")
    test_embeddings, test_labels = get_embeddings(val_loader, model, args)
    
    # predict class for test embeddings unsing knnn model
    print("Predict validation embeddings..")
    test_pred = knn_classifier.predict(X=test_embeddings)
    
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    print(confusion_matrix(y_true=test_labels, y_pred=test_pred)) 
    print(classification_report(y_true=test_labels, y_pred=test_pred))
    test_pred_prob = knn_classifier.predict_proba(X=test_embeddings)
    print(roc_auc_score(y_true=test_labels, y_score=test_pred_prob[:,1]))
    fpr, tpr, threshold = roc_curve(y_true=test_labels, y_score=test_pred_prob[:,1])
    
    df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    p = ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')
    ggsave(plot = p, filename = os.path.join(args.model_path, "roc_curve.jpg"), path = ".")


def build_knn_model(embeddings, labels, k=5):
    
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X=embeddings, y=labels)
    return knn_classifier


def get_embeddings(val_loader, model, args):
   
    embeddings = torch.zeros(size=(len(val_loader.dataset), args.dim), dtype=torch.float32, device=args.device)
    labels = torch.zeros(size=(len(val_loader.dataset),), dtype=torch.int, device=args.device)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pointer = 0
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            embeddings[pointer  : pointer+images.size(0)] = output
            labels[pointer : pointer+images.size(0)] = targets
            pointer += images.size(0)

    return embeddings.cpu().numpy(), labels.cpu().numpy()