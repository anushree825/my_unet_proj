# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:41:07 2020

"""
import os
import argparse

import torch, torch.nn as nn

from unet import ResnetSuperVision
from lung_dataloader import create_dataloaders
from train import train
from save_output import save_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)
# device = 'cpu'

parser = argparse.ArgumentParser()

parser.add_argument('--im_dim', type = int, default = (224,224), help = 'Size of input image')
parser.add_argument('--im_dir', type = str, default = 'siim_jpg/images/', help = 'Path to images directory')
parser.add_argument('--mask_dir', type = str, default = 'siim_jpg/masks/', help = 'Path to segmentation masks directory')
parser.add_argument('--train_csv', type = str, default = 'siim_train_small.csv', help = 'Path to train csv')

# parser.add_argument('--im_dir', type = str, default = 'siim_jpg/images/', help = 'Path to images directory')
# parser.add_argument('--mask_dir', type = str, default = 'siim_jpg/masks/', help = 'Path to segmentation masks directory')
# parser.add_argument('--train_csv', type = str, default = 'siim_train.csv', help = 'Path to train csv')

parser.add_argument('--one_hot', type = bool, default = False, help = 'use one hot labels ?')

parser.add_argument('--batch_sz', type = int, default = 4, help = 'batch size for dataset')
parser.add_argument('--num_workers', type = int, default = 4, help = 'Number of workers for dataloader')
parser.add_argument('--drop_last', type = bool, default = True, help = 'Drop last batch of size different from batch_sz')
parser.add_argument('--val_split', type = float, default = 0.2, help = 'Fraction of data used for validation')
parser.add_argument('--num_classes', type = int, default = 2, help = 'Number of classes in data')

parser.add_argument('--out_channels', type = int, default = 1, help = 'Number of channels of output image')
parser.add_argument('--bilinear', type = bool, default = True, help = 'Should UNet expansion use bilinear interpolation or conv.T')

parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')
parser.add_argument('--decay', type = float, default = 0.0004, help = 'decay rate for adam')
parser.add_argument('--tau', default = '30,35', help = 'milstones for multistepLR')
parser.add_argument('--gamma', type = float, default = 0.3, help = 'gamma for multistepLR')

parser.add_argument('--num_epochs', type = int, default = 10, help = 'number of epochs')
parser.add_argument('--threshold', type = float, default = 0.5, help = 'threshold for segmentation')

parser.add_argument('--save_model', type = bool, default = True, help = 'save model?')
parser.add_argument('--model_dict_path', default = './models/model_default_1.pth', help = 'file in which model parameters are saved')
parser.add_argument('--load_model', type = bool, default = False, help = 'load model parameters?')

parser.add_argument('--save_results', type = bool, default = False, help = 'save output images')

parser.add_argument('--debug', type = bool, default = False, help = 'If debug then code will run will less batches for debug purposes')

args = parser.parse_args()

loaders = create_dataloaders(args)
print("Creation of dataloader complete")

# im = next(iter(loaders['train']))
# print("chk im.shape:",im.shape)
# print(len(loaders['train']))
# for i, data in enumerate(loaders['train']):
#     x_imgs, labels, _ = data
#     print(x_imgs.shape)
# exit()

model = ResnetSuperVision(args.out_channels, backbone_arch='resnet34')
if(torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)
model.to(device)

if(not args.save_results):
    train(args, loaders, model)
    model.load_state_dict(torch.load(args.model_dict_path))
    save_output(args, model, loaders['save'])
else:
    model.load_state_dict(torch.load(args.model_dict_path))
    save_output(args, model, loaders['save'])