#!/usr/bin/env python
# coding: utf-8

# In[7]:


import random, torch, os, numpy as np
import torch.nn as nn
# import config
from torch.utils.data import Dataset
import copy


# In[8]:


class flowermonetDataset(Dataset):
    def __init__(self, root_monet, root_flower, transform=None):
        self.root_monet = root_monet
        self.root_flower = root_flower
        self.transform = transform

        self.monet_images = os.listdir(root_monet)
        self.flower_images = os.listdir(root_flower)
        self.length_dataset = max(len(self.monet_images), len(self.flower_images)) # 1000, 1500
        self.monet_len = len(self.monet_images)
        self.flower_len = len(self.flower_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        monet_img = self.monet_images[index % self.monet_len]
        flower_img = self.flower_images[index % self.flower_len]

        monet_path = os.path.join(self.root_monet, monet_img)
        flower_path = os.path.join(self.root_flower, flower_img)

        monet_img = np.array(Image.open(monet_path).convert("RGB"))
        flower_img = np.array(Image.open(flower_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=monet_img, image0=flower_img)
            monet_img = augmentations["image"]
            flower_img = augmentations["image0"]

        return monet_img, flower_img

