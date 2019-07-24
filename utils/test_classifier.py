import argparse
import operator
import os
from functools import reduce

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from celebA_data_loader import CelebA_Dataset
from simple_classifier import Classifier
import numpy as np
from tqdm import tqdm 



def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir',
                        help='Path to data directory', required=True)
    parser.add_argument('-a', '--attrib_path',
                        help='Path to attrib file', required=True)
    parser.add_argument('-m', '--model_dir',
                        help='Path to model directory', required=True)
    #parser.add_argument('-lr', help='Learning Rate',
    #                    type=float, required=False, default=0.001)
    parser.add_argument('-bs', '--batch_size', help='Batch size',
                        type=int, required=False, default=100)
    #parser.add_argument('--epochs', help='epochs', type=int,
    #                    required=False, default=10000)
    parser.add_argument('--test_attribute', required=True,
                        help='Attribute to train classifier on')
    return parser

parser = build_parser()
args = parser.parse_args()

def get_data_loader(args, test, shuffle=True):
    custom_transforms = transforms.Compose([transforms.CenterCrop(178),
                                            transforms.Resize((256, 256)),
                                            transforms.Lambda(lambda x: (
                                                2.0*np.asarray(x, dtype=np.float32)/255.0 - 1)), 
                                            transforms.ToTensor()])
    ds = CelebA_Dataset(args.attrib_path, args.data_dir, test,
                        args.test_attribute,transform=custom_transforms)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cl = Classifier(input_size=(256, 256, 3))

cl.load_state_dict(torch.load("/data/work2/AdversarialFaderNetworks/new_class_model/best_model.pth"))

cl.to(device)

cl.eval()
    
test_loader = get_data_loader(args, 'test')

correct = 0
total = 0

with torch.no_grad():
    for i,data in enumerate(test_loader):
        inputs,label = data
        inputs, label = inputs.to(device), label.to(device)
        outputs = cl(inputs)
        _, predicted = torch.max(outputs.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        if i==4:
            break
       

print('Accuracy on test images: %d %%' % (100* correct/total))



