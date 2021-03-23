#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, Alexnet, Branchy_Alexnet, LB_Net
from utils import get_dataset, average_weights, average_weights_cloud, exp_details



if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    exp_details(args)

    #if args.gpu:
        #torch.cuda.set_device(args.gpu)
    #device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL

    loss_type = 'ori'
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)

    elif args.model == 'alexnet':
        global_model = Alexnet(args=args)

    elif args.model == 'branchy_alexnet':
        loss_type = 'branchy'
        global_model = Branchy_Alexnet(args=args)

    elif args.model == 'LB_Net':
        loss_type = 'branchy'
        global_model = LB_Net(args=args)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    checkpoint_path = os.path.join('./checkpoint', args.model)
    write_path = os.path.join('./result', args.model)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    checkpoint_path = os.path.join(checkpoint_path, 'best_global_model-{dataset}-{epoch}-{exit}.pkl')
    write_path = os.path.join(write_path, '{dataset}-{iid}-{num_user}-{local_bs}-result.txt')

    # copy weights
    #global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy0, train_accuracy1 = [], [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    n_edge = 3
    edge_user = (args.num_users)/n_edge
    edges = []
    m = args.num_users
    #multi = m // 10
    #n_m = [0, 2*multi, m]
    n_m = [0, 3, 8, m]
    p = []
    for i in range(n_edge):
        edges.append(range(n_m[i], n_m[i+1]))
        p.append(float(1.0 * len(edges[i])) / m)

    print("p is ..",p)
    for i in range(n_edge):
        print("i is",i)
        print("edges is..", edges[i])
        for idx in edges[i]:
            print("idx", idx)

    acc0_epochs, acc1_epochs = [], []
    best_acc0, best_acc1 = 0.0, 0.0
    with open(write_path.format(dataset=args.dataset, num_user=args.num_users, iid=args.iid, local_bs=args.local_bs), "a+", encoding="utf-8") as f:
        for epoch in tqdm(range(args.epochs)):

            edge_weights = []
            global_model.train()

            for i in range(n_edge):
                local_weights, local_losses = [], []
                for idx in edges[i]:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                              idxs=user_groups[idx], loss_type=loss_type)
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                edge_tem_weights = average_weights(local_weights)
                edge_weights.append(edge_tem_weights)


            # update global weights
            global_weights = average_weights_cloud(edge_weights, p)

            #global_weights = average_weights(edge_weights)
            global_model.load_state_dict(global_weights)


            print("epoch is", epoch)
            test_acc, test_loss = test_inference(args, global_model, test_dataset, loss_type)

            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc[0]))
            acc0_epochs.append(test_acc[0])
            if test_acc[0] > best_acc0:
                torch.save(global_model, checkpoint_path.format(dataset=args.dataset, epoch=epoch, exit=0))
                best_acc0 = test_acc[0]
            if loss_type == 'branchy':
                print("|---- Test Accuracy (exit 1): {:.2f}%".format(100 * test_acc[1]))
                acc1_epochs.append(test_acc[1])
                if test_acc[1] > best_acc1:
                    torch.save(global_model, checkpoint_path.format(dataset=args.dataset, epoch=epoch, exit=1))
                    best_acc1 = test_acc[1]
            f.write(str(epoch))
            f.write('\n')
            f.write(str(acc0_epochs))
            f.write('\n')
            f.write(str(acc1_epochs))
            f.write('\n')
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
