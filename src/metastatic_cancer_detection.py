import os
import math
import argparse
import shutil

import pandas as pd
import numpy as np
import torch
import torchvision
import PIL
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
import time


from training import training
from dataloader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['ResNet','DenseNet','VGG']:
        print('ERROR!! model_type should be selected in ResNet/DenseNet/VGG')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    else:
        shutil.rmtree(config.model_path)
        os.makedirs(config.model_path)
    config.model_path = os.path.join(config.model_path,config.model_type)
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    else:
        shutil.rmtree(config.result_path)
        os.makedirs(config.result_path)       
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


    train_loader = get_loader(image_path=config.data_path + '/train/', labels_csv_file=config.label_path,image_size=config.image_size, batch_size=config.batch_size, num_workers=config.num_workers, mode='train', augmentation_prob=config.augmentation_prob, apply_transform=True)
    valid_loader = get_loader(image_path=config.data_path + '/valid/', labels_csv_file=config.label_path,image_size=config.image_size,batch_size=config.batch_size,num_workers=config.num_workers,mode='valid',augmentation_prob=0., apply_transform=False)
    test_loader = get_loader(image_path=config.data_path + '/test/', labels_csv_file=config.label_path,image_size=config.image_size,batch_size=config.batch_size,num_workers=config.num_workers, mode='test',augmentation_prob=0., apply_transform=False)

    data_loader = None
    if config.mode == 'train':
        data_loader = {
            "train": train_loader,
            "valid": valid_loader
        }
        
    elif config.mode =='test':
        data_loader = {
            "test": test_loader
        }

    solver = training(config, data_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/datadrive/Metastatic/dataset/')
    parser.add_argument('--label_path', type=str, default='/datadrive/Metastatic/train_labels.csv')
    parser.add_argument('--model_path',type=str, default='/datadrive/Metastatic/models/')
    parser.add_argument('--result_path',type=str, default='/datadrive/Metastatic/results/')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--model_type',type=str,default='ResNet',help='ResNet/DenseNet/VGG')
    parser.add_argument('--image_size',type=int, default=224)
    parser.add_argument('--mode',type=str,default='train',help='train/test')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--use_cuda',type=bool, default=True)

    config = parser.parse_args()
    print(config)
    main(config)