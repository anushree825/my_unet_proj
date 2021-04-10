# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:40:51 2020

"""

import os

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import lovasz_losses as L
from losses import ComboLoss
import time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(args, loader, model, optimizer, criterion, epoch):
    print("enter train")
    
    losses = []
    iou = []
    
    # for idx in enumerate(len(loader)):
    tepoch = tqdm(range(len(loader)),unit='batch')

    # with tqdm(total=len(loader), unit='batch') as tepoch:
        # print("len(loader):" ,len(loader))
    for idx,(im,mask,label) in zip(tepoch,loader):
        # print('yay')
        tepoch.set_description(f"Epoch (Train) {epoch}")

        im = im.to(device)
        mask = mask.to(device)
        # print("train -- mask:", mask.shape)

        # label = label.to(device).squeeze()
        
        out = model(im)
        # print("train -- out:", out.shape)
        loss = criterion(out,mask)
        loss.backward()
        
        optimizer.zero_grad()
        
        optimizer.step()
        
        losses.append(loss.item())

        with torch.no_grad():
            temp_out = torch.sigmoid(out)
            preds = (temp_out.data > args.threshold).long()
            iou.append(L.iou_binary(preds, mask, per_image=True))
        
        tepoch.set_postfix(batch_loss=loss.item(), batch_iou = iou[-1], idx = idx)

        if(args.debug and idx == 5):break


    print('Epoch (Train) {0} Mean Loss [{1:.8f}], mIOU [{2:.8f}]'.format(epoch, np.mean(losses), np.mean(iou)))

def eval_epoch(args, loader, model, criterion, epoch):
    
    losses = []
    iou = []
    with torch.no_grad():
        tepoch = tqdm(range(len(loader)),unit='batch')

    # with tqdm(total=len(loader), unit='batch') as tepoch:
        # print("len(loader):" ,len(loader))
        for idx,(im,mask,label) in zip(tepoch,loader):

            tepoch.set_description(f"Epoch (Val) {epoch}")
            
            im = im.to(device)
            mask = mask.to(device)
            # print("train -- mask:", mask.shape)

            # label = label.to(device).squeeze()
            
            out = model(im)
            # print("train -- out:", out.shape)
            loss = criterion(out,mask)
            
            losses.append(loss.item())

            out = torch.sigmoid(out)
            preds = (out.data > args.threshold).long()
            iou.append(L.iou_binary(preds, mask, per_image=True))

            tepoch.set_postfix(batch_loss=loss.item(), batch_iou = iou[-1],idx = idx)
            
            if(args.debug and idx == 10):break
    
    print('Epoch (Val) {0} Mean Loss [{1:.8f}], mIOU [{2:.8f}]'.format(epoch, np.mean(losses), np.mean(iou)))
        


def train(args, loaders, model):
    
    to_optim = [{'params':model.parameters(),'lr':args.lr,'weight_decay':args.decay}]
    optimizer = optim.Adam(to_optim)

    # tau = list(map(int,args.tau.split(',')))
    # scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tau, gamma=args.gamma)

    criterion = ComboLoss(**{'weights':{'bce':3, 'dice':1, 'focal':4}})
    model = model.to(device)
    for epoch in range(args.num_epochs):
        
        model.train()
        print('begin train')
        train_epoch(args, loaders['train'], model, optimizer, criterion, epoch)
        
        model.eval()
        eval_epoch(args, loaders['val'], model, criterion, epoch)
        
        # scheduler.step()
        
    if(args.save_model):
        torch.save(model.state_dict(), args.model_dict_path)


# def train(args, loaders, model):

#     ENCODER = 'se_resnext50_32x4d'
#     ENCODER_WEIGHTS = 'imagenet'
#     CLASSES = ['0','1']
#     ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
#     DEVICE = device

#     # create segmentation model with pretrained encoder
#     model = smp.FPN(
#         encoder_name=ENCODER, 
#         encoder_weights=ENCODER_WEIGHTS, 
#         classes=len(CLASSES), 
#         activation=ACTIVATION,
#     )



#     loss = smp.utils.losses.DiceLoss()
#     metrics = [
#         smp.utils.metrics.IoU(threshold=0.5),
#     ]

#     optimizer = torch.optim.Adam([ 
#         dict(params=model.parameters(), lr=args.lr),
#     ])

#     train_epoch = smp.utils.train.TrainEpoch(
#         model, 
#         loss=loss, 
#         metrics=metrics, 
#         optimizer=optimizer,
#         device=DEVICE,
#         verbose=True,
#     )

#     valid_epoch = smp.utils.train.ValidEpoch(
#         model, 
#         loss=loss, 
#         metrics=metrics, 
#         device=DEVICE,
#         verbose=True,
#     )


#     max_score = 0

#     for i in range(0, args.num_epochs):
        
#         print('\nEpoch: {}'.format(i))
#         train_logs = train_epoch.run(loaders['train'])
#         valid_logs = valid_epoch.run(loaders['val'])
        
#         # do something (save model, change lr, etc.)
#         if max_score < valid_logs['iou_score']:
#             max_score = valid_logs['iou_score']
#             # torch.save(model, './best_model.pth')
#             if(args.save_model):
#                 torch.save(model.state_dict(), args.model_dict_path)
#             print('Model saved!')
            
#         if i == 25:
#             optimizer.param_groups[0]['lr'] = args.lr
#             print('Decrease decoder learning rate to 1e-5!')
        
#     # to_optim = [{'params':model.parameters(),'lr':args.lr,'weight_decay':args.decay}]
#     # optimizer = optim.Adam(to_optim)

#     # tau = list(map(int,args.tau.split(',')))
#     # scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tau, gamma=args.gamma)

#     # criterion = ComboLoss(**{'weights':{'bce':3, 'dice':1, 'focal':4}})
    
#     # for epoch in range(args.num_epochs):
        
#     #     model.train()
#     #     print('begin train')
#     #     train_epoch(args, loaders['train'], model, optimizer, criterion, epoch)
        
#     #     model.eval()
#     #     eval_epoch(args, loaders['val'], model, criterion, epoch)
        
#     #     # scheduler.step()
        
#     # if(args.save_model):
#     #     torch.save(model.state_dict(), args.model_dict_path)
        


