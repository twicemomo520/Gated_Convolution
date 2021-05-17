from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import cv2
import pickle
import glob
import random


class outdoordata(Dataset):
    def __init__(self, processdata_path, mask_path):
        self.data_path = processdata_path
        self.mask_path = mask_path
        self.RGBimages = glob.glob(os.path.join(self.data_path, '*rgb.png'))
        self.RGBimages.sort()
        self.masks = glob.glob(os.path.join(self.mask_path, '*'))    
        self.masks.sort()
        
    def __getitem__(self, index):
        items = self.transform(index) 
        return items
        
    def __len__(self):
        return len(glob.glob(os.path.join(self.data_path, '*rgb.png')))
    
    def loadimage(self, index):        
        image = cv2.imread(self.RGBimages[index])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(256,256))
        
        return image
    
    def loadmask(self):
        index = random.randint(0, len(self.masks)-1)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    
    def transform(self, index):
        image = self.loadimage(index) 
        mask = self.loadmask()
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        return image, mask
'''
data = outdoordata('./data/outdoor2_data', './data/mask')
plt.imshow(data[0][0].permute(1,2,0).cpu().data.numpy())
plt.imshow(data[0][1].permute(1,2,0).cpu().data.numpy(), 'gray')
'''

'''
import glob
masks = glob.glob('C:/Users/123/Desktop/gated convolution/data/mask/*')
for i in range(len(masks)):
    path = masks[i].split('\\')[0]
    name = masks[i].split('\\')[1].split('.jpg')[0]
    os.rename(os.path.join(path, name + '.jpg'), os.path.join(path, name + '.png'))
''' 