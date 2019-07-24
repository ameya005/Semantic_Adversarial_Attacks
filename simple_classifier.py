##############################################
# Male-female CelebA classifier training
#############################################

import argparse
import operator
import os
from functools import reduce

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from celebA_data_loader import CelebA_Dataset
from utils import get_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def fgsm(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    input = x.clone().detach_().to(device)
    input.requires_grad_()
    target = torch.LongTensor([target]).to(device)

    logits = model(input)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    if targeted:
        out = input - eps * input.grad.sign()
    else:
        out = input + eps * input.grad.sign()

    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)

    return out


def pgd(model, x, target, k, eps, eps_step, targeted=True, clip_min=None,
        clip_max=None):
    x_min = x - eps
    x_max = x + eps

    # generate random point in +-eps box around x
    x = 2. * eps * torch.rand_like(x) - eps

    for i in range(k):
        # FGSM step
        x = fgsm(model, x, target, eps_step, targeted)
        # projection step
        x = torch.max(x_min.to(device), x)
        x = torch.min(x_max.to(device), x)

    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)

    return x


def batched_pgd(model, x_batch, y_batch, k, eps, eps_step, targeted=True,
                clip_min=None, clip_max=None):
    n = x_batch.size()[0]
    xprime_batch_list = []

    for i in range(n):
        x = x_batch[i, ...].unsqueeze(0)
        # print(x.size())
        y = y_batch[i]
        xprime = pgd(
            model, x, y, k, eps, eps_step, targeted, clip_min, clip_max
        )
        xprime_batch_list.append(xprime.squeeze(0))

    xprime_batch_tensor = torch.stack(xprime_batch_list)
    assert x_batch.size() == xprime_batch_tensor.size()

    return xprime_batch_tensor


class Classifier(nn.Module):

    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.inp_size = input_size
        # tmp = self.conv_layers(3).forward(torch.rand(
        #    1, self.inp_size[2], self.inp_size[0], self.inp_size[1]))
        #self.conv_out_shape = tmp.size()
        self.conv_out_shape = (1, 128, 33, 33)
        self.flatten_size = prod(self.conv_out_shape[1:])
        self.conv_layers(3)
        self.fully_connected()

    def conv_layers(self, nc):
        self.l1 = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # return nn.Sequential(self.l1, self.l2, self.l3)

    def fully_connected(self):
        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 2)
        )
        # return nn.Sequential(self.fc1, self.fc2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir',
                        help='Path to data directory', required=True)
    parser.add_argument('-a', '--attrib_path',
                        help='Path to attrib file', required=True)
    parser.add_argument('-m', '--model_dir',
                        help='Path to model directory', required=True)
    parser.add_argument('-lr', help='Learning Rate',
                        type=float, required=False, default=0.001)
    parser.add_argument('-bs', '--batch_size', help='Batch size',
                        type=int, required=False, default=32)
    parser.add_argument('--epochs', help='epochs', type=int,
                        required=False, default=10000)
    parser.add_argument('--train_attribute', required=True,
                        help='Attribute to train classifier on')
    parser.add_argument(
        '--robustify', '-r', help='Flag to train using Madry\'s method', action='store_true')
    return parser


def get_data_loader(args, train, shuffle=True):
    custom_transforms = transforms.Compose([transforms.CenterCrop(178),
                                            transforms.Resize((256, 256)),
                                            transforms.Lambda(lambda x: (
                                                2.0*np.asarray(x, dtype=np.float32)/255.0 - 1)),
                                            transforms.ToTensor()])
    ds = CelebA_Dataset(args.attrib_path, args.data_dir, train,
                        args.train_attribute, transform=custom_transforms)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)


def save_model(args, model, filename):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, filename))


def restore_model(model_file, input_size):
    cl = Classifier(input_size)
    wts = torch.load(model_file)
    cl.load_state_dict(wts)
    return cl


def validate(model, device, ds_loader, criterion):
    model.eval()
    corrects = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in ds_loader:
            inputs, label = inputs.to(device), labels.to(device)
            output = model(inputs)
            val_loss += criterion(output, label).item()
            preds = output.max(1, keepdim=True)[1]
            corrects += preds.eq(label.view_as(preds)).sum().item()
        acc = corrects / float(len(ds_loader.dataset))
    return val_loss, acc


def train(args):
    train_loader = get_data_loader(args, 'train')
    val_loader = get_data_loader(args, 'valid')

    cl = Classifier(input_size=(256, 256, 3))
    # print([x for x in cl.parameters()])
    cl.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn.ParameterList(cl.parameters()), lr=args.lr)
    if args.robustify:
        args.logger.info('Training a robust model')
    best_acc = 0.0
    for epoch in range(args.epochs):
        cl.train()
        count = 0
        for inputs, label in train_loader:
            img1 = inputs[0].detach().numpy().transpose(1, 2, 0)
            inputs, label = inputs.to(device), label.to(device)
            if args.robustify:
                cl.eval()
                inputs = batched_pgd(cl, inputs, label, k=5,
                                     eps=0.2, eps_step=0.05, targeted=False)
                plt.show()
                cl.train()
            optimizer.zero_grad()
            output = cl(inputs)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            count += len(inputs)

        tr_loss, tr_acc = validate(cl, device, train_loader, loss_fn)
        args.logger.info(
            'Training-----Epoch {}: Loss={:.4f}, Accuracy={:.3f}'.format(epoch, tr_loss, tr_acc))

        val_loss, val_acc = validate(cl, device, val_loader, loss_fn)
        args.logger.info(
            'Validation------Epoch {}: Loss={:.4f}, Accuracy={:.3f}'.format(epoch, val_loss, val_acc))

        if val_acc > best_acc:
            save_model(args, cl, 'best_model.pth')
            best_acc = val_acc
    return cl


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.logger = get_logger(args.model_dir)
    args.logger.info('Starting training')
    cl = train(args)


if __name__ == '__main__':
    main()
