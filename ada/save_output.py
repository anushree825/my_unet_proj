# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 07:44:17 2020
"""

import os

import torch
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from skimage.io import imsave

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_output(args, model, loader):
    
    if(not os.path.exists('./results')): 
        os.mkdir('./results')
    
    count = 0

    with torch.no_grad():
        for idx,(im,mask) in enumerate(tqdm(loader)):
            
            im = im.to(device)
            torch.cuda.empty_cache()

            out,_ = model(im)
            # softmax = torch.nn.functional.log_softmax(out, dim=1)
            # out = torch.argmax(softmax, dim=1)

            out = (out.squeeze().data > args.threshold).long()

            # print(out.shape)
            for j in range(out.size(0)):
                    
                    out_im =  out[j].detach().cpu().numpy()
                    
                    imsave('./results/' + str(count) + '.jpg', out_im, cmap = 'gray') # Create a results/ directory before running
                    
                    count+=1
            del im,out


