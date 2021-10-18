#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from simsiam.evaluate.eval_utils import get_embeddings
from simsiam.pretrain.load_pretrained_model import load_pretrained_model
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
    model = load_pretrained_model(args)
   
    print(model)
    cudnn.benchmark = True

    # Data loading code    
    data_provider = DataProvider(args=args)
    train_loader, _ = data_provider.get_train_loader(aug=False, sampling=args.train_sampling)
    
    print("Determine train embeddings..")
    embeddings, labels = get_embeddings(train_loader, model, args)
    
    knn_classifier = build_knn_model(embeddings, labels, k=5)
    print("Provide KNN-classifier")
    
     # Data loading code
    val_loader = data_provider.get_val_loader(sampling=args.test_sampling)

    print("Determine validation embeddings..")
    test_embeddings, test_labels = get_embeddings(val_loader, model, args)
    
    # predict class for test embeddings unsing knnn model
    print("Predict validation embeddings..")
    test_pred = knn_classifier.predict(X=test_embeddings)
    
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
    import pandas as pd
    cmtx = pd.DataFrame(
    confusion_matrix(y_true=test_labels, y_pred=test_pred), 
    index=['true:-1', 'true:1'], 
    columns=['pred:-1', 'pred:1'])
    print(cmtx)

    print(classification_report(y_true=test_labels, y_pred=test_pred))
    test_pred_prob = knn_classifier.predict_proba(X=test_embeddings)
    print(f'AUC: {roc_auc_score(y_true=test_labels, y_score=test_pred_prob[:,1])}')
    fpr, tpr, threshold = roc_curve(y_true=test_labels, y_score=test_pred_prob[:,1])
    
    df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    p = ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')
    ggsave(plot = p, filename = os.path.join(args.model_path, "roc_curve.jpg"), path = ".")


def build_knn_model(embeddings, labels, k=5):
    
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X=embeddings, y=labels)
    return knn_classifier
