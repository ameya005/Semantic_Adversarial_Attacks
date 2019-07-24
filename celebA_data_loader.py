#####################################
#Simple dataloader for celebA########
#####################################

__author = "Ameya Joshi"
__email = "ameya@iastate.edu"

import torch as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from matplotlib import pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import os
import numpy as np
from matplotlib import image as mpimg

AVAILABLE_ATTR = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]


_SORTED_ATTR = [ 
        "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Eyeglasses",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "No_Beard",
        "Pale_Skin",
        "Young"
        ]

class CelebA_Dataset(Dataset):

    def __init__(self, attrib_path, img_dir, train, attrib_name, transform=None, att_gan=False):
        df = pd.read_csv(attrib_path, skiprows=1)
        self.img_dir  = img_dir
        self.attrib_path = attrib_path
        self.img_names = df['Imgname'].values
        self.attrib_name = attrib_name
        self.att_gan = att_gan
        #if isinstance(self.attrib_name, list):
        #    self.attrib_name.sort()
        #if attrib_name not in AVAILABLE_ATTR:
        #    raise Exception('Use a different attribute from %s' % ','.join(AVAILABLE_ATTR))
        if self.att_gan:
            self.y = df[_SORTED_ATTR].values
        else:
            self.y = df[self.attrib_name].values
        self.transform = transform
        self.train_index = 162770
        self.valid_index = 162770 + 19867
        self.test_index = len(self.y)
        if train == 'train':
            self.img_names = self.img_names[:self.train_index]
            self.y = self.y[:self.train_index]
        elif train == 'valid':
            self.img_names = self.img_names[self.train_index:self.valid_index]
            self.y = self.y[self.train_index:self.valid_index]
        elif train == 'test':
            self.img_names = self.img_names[self.valid_index:]
            self.y = self.y[self.valid_index:]
        
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        if self.transform is not None:
            img = self.transform(img)
        #plt.imshow(img.numpy().transpose((1,2,0)))
        #plt.show()
        label = self.y[index]
        if not self.att_gan:
            if label == -1:
                label = 0
        else:
            mod_label = [np.float32(0.0) if i == -1 in label else np.float32(i) for i in label]
            label = mod_label
        return (img, label)
    
    def __len__(self):
        return self.y.shape[0]
