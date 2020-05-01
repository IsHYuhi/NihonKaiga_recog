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

random.seed(44)

class NishikaDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train', csv_path='None'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.csv_path = csv_path
        self.csv_dic={}
        with open(self.csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                self.csv_dic[row[0]] = row[1]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        get tensor type preprocessed data and labels
        '''

        img_path = self.file_list[index]
        img = Image.open(img_path) #[H][W][C]

        img_transformed = self.transform(img, self.phase)

        label = self.csv_dic[img_path.split('/')[3]]

        return img_transformed, int(label)

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
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

    num = len(path_list)
    random.shuffle(path_list)
    return path_list[:int(num*rate)], path_list[int(num*rate):]