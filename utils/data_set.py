import glob
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import csv

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from .autoaugment import ImageNetPolicy

random.seed(44)

class NishikaDataset(data.Dataset):
    def __init__(self, label_dic, root_dir, transform=None, phase='train'):
        self.label_dic = label_dic
        self.root_dir = root_dir
        self.images = list(label_dic.keys())
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''
        get tensor type preprocessed data and labels
        '''
        filename = self.images[index]
        img = Image.open(os.path.join(self.root_dir, filename))
        img_transformed = self.transform(img, self.phase)

        label = self.label_dic[filename]
        return img_transformed, label

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        }

    def __call__(self, img, phase='train'):

        return self.data_transform[phase](img)

def make_datapath_list(phase="train", rate=0.8):

    rootpath = "./data/"
    target_path = os.path.join(rootpath+phase+'/*.jpg')
    #print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    if phase=='train':
        num = len(path_list)
        random.shuffle(path_list)
        return path_list[:int(num*rate)], path_list[int(num*rate):]

    elif phase=='test':
        return path_list