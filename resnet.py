"""
Simple Resnet Classifier
"""

import torch
from torch import nn
import os
import numpy as np
import argparse
import torchvision as tv
import tensorboardX as tbX
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

_RES_MODELS = {
    'resnet18': tv.models.resnet18, 'resnet34': tv.models.resnet34, 'resnet50': tv.models.resnet50, 'resnet101': tv.models.resnet101,
    'resnet152': tv.models.resnet152, 'resnext50_32x4d': tv.models.resnext50_32x4d, 'resnext101_32x8d': tv.models.resnext101_32x8d,
    'wide_resnet50_2': tv.models.wide_resnet50_2, 'wide_resnet101_2': tv.models.wide_resnet101_2 , 'mobilenet': tv.models.mobilenet_v2
}


def get_next_run(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    return path


class ResNet(nn.Module):
    """
    Resnet classifier
    """

    def __init__(self, n_classes=3, rtype='resnet34', restore=False, path=None):
        super(ResNet, self).__init__()
        if not restore:
            self.n_classes = n_classes
            self.rtype = rtype
            self.res_model = self._get_base_resnet(self.rtype)(
                pretrained=False, progress=False, num_classes=self.n_classes)
        else:
            if path is None:
                raise Exception('Please provide path to load model')
            self.load_model(path)

    def _get_base_resnet(self, rtype):
        return _RES_MODELS[rtype]

    def forward(self, x):
        return self.res_model(x)

    def save_model(self, path):
        path = os.path.join(path, 'resmodel.pth')
        sd = self.res_model.state_dict()
        output_dict = {'rtype': self.rtype,
                       'n_classes': self.n_classes, 'model_state_dict': sd}
        torch.save(output_dict, path)

    def load_model(self, path):
        model_dict = torch.load(path)
        self.rtype = model_dict['rtype']
        self.n_classes = model_dict['n_classes']
        self.res_model = self._get_base_resnet(self.rtype)(
            pretrained=False, progress=False, num_classes=self.n_classes)
        self.res_model.load_state_dict(model_dict['model_state_dict'])


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_data_loader(path, batch_size=16, mode='train'):
    path = os.path.join(path, mode)
    if mode == 'train':
        transforms = tv.transforms.Compose([
            tv.transforms.Resize(size=(128, 128)),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            #tv.transforms.Normalize(mean=(0,0,0), std=(255., 255., 255.), inplace=True),
        ])
    else:
        transforms = tv.transforms.Compose([tv.transforms.Resize(size=(128,128)), tv.transforms.ToTensor(), ])
                                            #tv.transforms.Normalize(mean=(0,0,0), std=(255.0,255.,255.), inplace=True)])
    ds = ImageFolder(path, transform=transforms)
    if mode == 'train':
        weights = make_weights_for_balanced_classes(
            ds.imgs, len(ds.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights))
        shuffle = True
    else:
        sampler = None
        shuffle = False

    dl_loader = DataLoader(ds, batch_size=batch_size, 
                           sampler=sampler, pin_memory=True)
    return dl_loader


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
    """
    Train a resnet
    """
    args.outdir = get_next_run(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = ResNet(args.nclasses, args.rtype)
    resnet = resnet.to(device)
    optim = torch.optim.Adam(params=resnet.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_dl = get_data_loader(args.dpath, batch_size=args.batchsize, mode='train')
    val_dl = get_data_loader(args.dpath, batch_size=args.batchsize, mode='val')
    test_dl = get_data_loader(args.dpath, batch_size=args.batchsize, mode='test')

    writer = tbX.SummaryWriter(logdir=args.outdir)
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        resnet.train()
        count = 0
        for idx, (imgs, labels) in enumerate(train_dl):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            preds = resnet(imgs)
            loss = loss_fn(preds, labels).mean()
            loss.backward()
            optim.step()
            # if idx == 10:
            #    break

        resnet.eval()
        tr_loss, tr_acc = validate(resnet, device, train_dl, loss_fn)
        val_loss, val_acc = validate(resnet, device, val_dl, loss_fn)
        test_loss, test_acc = validate(resnet, device, test_dl, loss_fn)
        writer.add_scalar('Tr_Loss', tr_loss, epoch)
        writer.add_scalar('Tr_acc', tr_acc, epoch)
        writer.add_scalar('Val_loss', val_loss, epoch)
        writer.add_scalar('Val_acc', val_acc, epoch)
        writer.add_scalar('Test_loss', test_loss, epoch)
        writer.add_scalar('Test_acc', test_acc, epoch)
        print('Tr loss: {:03f}, Tr acc: {:02f}, Val loss: {:03f}, Val acc: {:02f}'.format(
            tr_loss, tr_acc, val_loss, val_acc))
        if val_loss <= best_val_loss:
            resnet.save_model(args.outdir)
            print('Saving model at epoch {}, path {}.'.format(epoch, args.outdir))
            best_val_loss = val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dpath', '-d', help='Datapath to images', required=True)
    parser.add_argument(
        '--outdir', '-o', help='Path to output directory', default='.')
    parser.add_argument('--rtype', '-rt', help='Type of Resnet', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                                                                          'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                                                                          'wide_resnet50_2', 'wide_resnet101_2', 'mobilenet'], default='resnet34')
    parser.add_argument('--nclasses', '-n', help='No. of Classes', default=3, type=int)
    parser.add_argument('--dname', '-dn', help='Name of dataset',
                        choices=['BDD'], default='BDD')
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=16, type=int)
    parser.add_argument('--epochs', '-e', help='No of epochs', default=100000, type=int)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
