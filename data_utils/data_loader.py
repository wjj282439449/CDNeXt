import glob
import os

from numpy.core.fromnumeric import shape
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from data_utils.augment import *
# import augment as augment
import pandas as pd
from PIL import Image
# import random

class WSIDataset(Dataset):
    def __init__(self, root_dir, mode, taskList, total_fold = 5, valid_fold = 2, miniScale = 1):

        self.root_dir = root_dir
        self.mode = mode
        # modify    
        self.miniScale = miniScale
        self.total_fold = total_fold
        self.valid_fold = valid_fold

        self.all_png_dir_1    = []
        self.all_png_dir_2    = []
        self.all_label_change = []
        for k,v in self.root_dir.items():
            self.all_png_dir_1    += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "T1" + os.sep + '*'))#[0: 555]
            self.all_png_dir_2    += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "T2" + os.sep + '*'))#[0: 555]
            self.all_label_change += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "label" + os.sep + '*'))#[0: 555]
        self.all_png_dir_1_name =  [os.path.splitext(os.path.split(i)[1])[0] for i in self.all_label_change]
        print("T1 patch numbers: ", len(self.all_png_dir_1))
        print("T2 patch numbers: ", len(self.all_png_dir_2))
        print("label patch numbers: ", len(self.all_label_change))
        self.isTrain = False
        self.source_size = (256,256)
        self.randomImgSizeList = [256,256]
        self.randomImgSizeList = self.randomImgSizeList[::-1]
        self.randomImgSize = (256, 256)
        
    def __getitem__(self, index):
        dir        = self.all_png_dir_1_name[index]
        img1       = self.all_png_dir_1[index]
        img2       = self.all_png_dir_2[index]
        labelc     = self.all_label_change[index]

        if self.mode == "train":
            img1       = np.array(Image.open(img1).resize(self.randomImgSize))
            img2       = np.array(Image.open(img2).resize(self.randomImgSize))
            labelc     = np.expand_dims(np.array(Image.open(labelc).resize(self.randomImgSize)), axis=2)
            
            img1       = mirrorPadding2D(img1)
            img2       = mirrorPadding2D(img2)
            labelc     = mirrorPadding2D(labelc)
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)
            labelc =  Image.fromarray(np.squeeze(labelc))

            aug = Augmentation()
            # geometric distortion
            img2_combine, bias_y, bias_x = aug.randomSpaceAugment([img1,img2,labelc], source_size=self.randomImgSize, unoverlap=None)
            # photometric distortion
            img1,img2,labelc = img2_combine
            imgPhotometricDistortion1 = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            ])
            imgPhotometricDistortion2 = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            img1 = imgPhotometricDistortion1(img1)
            img2 = imgPhotometricDistortion2(img2)
            labelc     = torch.FloatTensor(np.array(labelc))/255

        elif self.mode in "validation" or self.mode in "test":
            img1       = Image.open(img1).resize(self.randomImgSize)
            img2       = Image.open(img2).resize(self.randomImgSize)
            imgTransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            labelc     = np.expand_dims(np.array(Image.open(labelc).resize(self.randomImgSize)), axis=2)
            labelc     = torch.FloatTensor(np.squeeze(labelc))/255
            img1       = imgTransforms(img1)
            img2       = imgTransforms(img2)
        label1 = torch.FloatTensor([0])
        label2 = torch.FloatTensor([0])
        return img1, img2, label1, label2, labelc, dir

    def __len__(self):
        return len(self.all_png_dir_1)

if __name__ == "__main__":

    pass