"""
Attacking the model using a Fader Network
Note: We are basically searching for the interpolation values
which allow us to break a simple classifier. 
"""

import argparse
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from AttGAN.attgan import Generator
from simple_classifier import Classifier, restore_model
from celebA_data_loader import _SORTED_ATTR, CelebA_Dataset
from utils import get_logger
from losses import nontarget_logit_loss

import copy

class AttEncoderModule(nn.Module):
    """
    Class for constructing the attribute vector
    """

    def __init__(self, alpha_init, attrib_flags, thresh_int, projection_step=False, eps=1.0):
        super(AttEncoderModule, self).__init__()
        self.attr_a = alpha_init.clone()
        self.attr_b = alpha_init.clone()
        self.alpha = []
        self.indices = []
        self.thresh_int = thresh_int
        for i in attrib_flags:
            idx = _SORTED_ATTR.index(i)
            print(idx, alpha_init[idx])
            self.alpha.append(torch.tensor(1.0).requires_grad_(True))
            self.indices.append(idx)
        for i in self.indices:
            self.attr_b[i] = 1 - self.attr_b[i]
        self.eps = torch.tensor(eps)

    def get_optim_params(self):
        return self.alpha

    def forward(self):
        attr_b = (self.attr_b * 2 - 1) * self.thresh_int
        for i, j in zip(self.indices, self.alpha):
            attr_b[i] = (j * 2 -1) * self.thresh_int
        print('ATTR',attr_b)
        # Projection step
        #attr_b = torch.min(attr_b, self.eps)
        #attr_b = torch.max(attr_b, -self.eps)

        attr_b = attr_b.unsqueeze(0)
        return attr_b

class Attacker(nn.Module):
    """
    Defines a attack system which can be optimized.
    Input passes through a pretrained fader network for modification.
    Input -> (Fader network) -> (Target model) -> Classification label

    Since Fader network requires that the attribute vector elements (alpha_i) be converted to (alpha_i, 1-alpha_i),
    we use the Mod alpha class to handle this change while preserving gradients.
    """

    def __init__(self, params, params_gen, input_logits):
        super(Attacker, self).__init__()
        self.params = params
        self.target_model = Classifier(
            (params.img_sz, params.img_sz, params.img_fm))
        self.adv_generator = Generator(params_gen.enc_dim, params_gen.enc_layers, params_gen.enc_norm, params_gen.enc_acti,
                                       params_gen.dec_dim, params_gen.dec_layers, params_gen.dec_norm, params_gen.dec_acti,
                                       params_gen.n_attrs, params_gen.shortcut_layers, params_gen.inject_layers, params_gen.img_size)
        self.eps = params.eps
        self.projection = params.proj_flag
        self.input_logits = torch.tensor(input_logits).requires_grad_(False)
        self.attrib_gen = AttEncoderModule(self.input_logits, params.attk_attribs, params_gen.thres_int, self.projection, self.eps)

    def restore(self, legacy=False):
        self.target_model.load_state_dict(torch.load(self.params.model))
        if legacy:
            old_model_state_dict = torch.load(self.params.fader)
            old_model_state_dict.update(_LEGACY_STATE_DICT_PATCH)
            model_state_d = old_model_state_dict
        else:
            model_state_d = torch.load(self.params.attgan)
        self.adv_generator.load_state_dict(model_state_d['G'])

    def forward(self, x, attrib_vector=None):
        if attrib_vector is None:
            self.attrib_vec = self.attrib_gen()
        else:
            self.attrib_vec = attrib_vector
        l_z = self.adv_generator.encode(x)
        recon = self.adv_generator.decode(l_z, self.attrib_vec)
        cl_label = self.target_model(recon)
        return recon, cl_label

def get_data_loader(args, train, shuffle=True):
    custom_transforms = transforms.Compose([transforms.CenterCrop(178),
                                            transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    ds = CelebA_Dataset(args.attrib_path, args.data_dir, train,
                        args.train_attribute, transform=custom_transforms, att_gan=True)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Path to the model to be attacked. As of now, we will use models trained with the simple_classifier script', required=True)
    parser.add_argument(
        '-f', '--attgan', help='Path to attgan networks. Use one of the many AttGAN networks trained', required=True)
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='out', required=False)
    parser.add_argument('-d', '--data_dir',
                        help='Path to data directory', required=True)
    parser.add_argument('-a', '--attrib_path',
                        help='Path to attrib file', required=True)
    parser.add_argument('-t', '--type', help='Attack type: \n\t att: AttGAN based attack \n\t rn: Random attack',
                        choices=['att', 'rn'], default='fn')
    parser.add_argument('--proj_flag', action='store_true', help='Infinity projection flag')
    parser.add_argument('--eps', help='Epsilon value', default=4.0, type=float)
    parser.add_argument('--attk_attribs', nargs='+', help='Attributes to attack over')
    return parser



def attack_random(img, model, num_samples, device, logger, eps):
    """
    Optimizer based attack.
    Iterates over randomly generated attribute vector values and picks the one with
    the worst loss value.
    """
    MAX_ITER = 500
    step = 0
    model.eval()
    model = model.to(device)
    orig_img = img.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)
    img = img.to(device)
    with torch.no_grad():
        orig_logits = model.target_model(img)
        labels = torch.argmax(orig_logits)
    SUCCESS = 1
    FAILURE = 0

    alphas = []
    for i in range(10):
        alpha_init = np.random.rand(6).astype(np.float32) * 2 * eps - eps
        attrib_vec = model.attrib_gen.attr_a.clone().detach().numpy()
        indices = model.attrib_gen.indices
        for idx, att_idx in enumerate(indices):
            attrib_vec[att_idx] = alpha_init[idx]
            alphas.append(torch.tensor(attrib_vec).to(device).unsqueeze(0))

    with torch.no_grad():
        mod_inputs = [model(img, i) for i in alphas]
        model_outs = [nontarget_logit_loss(i[1], labels) for i in mod_inputs]
        loss_vals = [i.detach().cpu().numpy() for i in model_outs]
        worst_loss = np.argmin(np.asarray(loss_vals))
        worst_pred = mod_inputs[worst_loss][1]
        recon = mod_inputs[worst_loss][0]
        worst_loss_val = loss_vals[worst_loss]
        out_img = np.hstack(
            [orig_img, recon.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)])
        if np.argmax(mod_inputs[worst_loss][1].detach().cpu().numpy()) != labels:
            logger.info(
                'Broken-Step:{}, alpha:{}'.format(step, model.attrib_vec))
            return SUCCESS, out_img, alphas[worst_loss], orig_logits.detach().cpu().numpy(), worst_pred.detach().cpu().numpy()
        else:
            return FAILURE, out_img, alphas[worst_loss], orig_logits.detach().cpu().numpy(), worst_pred.detach().cpu().numpy()

def attack_optim(img, model, attrib_tuple, device, logger):
    """
    Optimizer based attack. 
    Runs a reverse gradient to find the value of alpha vector
    which breaks the target model.
    """
    MAX_ITER = 500
    step = 0
    model.train()
    model = model.to(device)
    orig_img = img.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)
    img = img.to(device)
    orig_logits = model.target_model(img)
    labels = torch.argmax(orig_logits)
    SUCCESS = 1
    FAILURE = 0
    optim = torch.optim.RMSprop(model.attrib_gen.get_optim_params(), lr=0.01, weight_decay=0.01)
    loss = np.inf
    logit_arrays = [orig_logits.detach().cpu().numpy().tolist()]
    prev_loss = 10000.0
    pat_cnt = 0
    while loss != 0.0 and step < MAX_ITER:
        recon, logits = model(img)
        logit_arrays.append(logits.detach().cpu().numpy().tolist())
        pred = torch.argmax(logits)
        logger.debug('Modified label:%s, Orig label:%s', pred, labels)
        loss = nontarget_logit_loss(logits, labels)
        if pred.cpu().detach().numpy() != labels.cpu().detach().numpy():
            out_img = np.hstack(
                [orig_img, recon.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)])
            logger.info(
                'Broken-Step:{}, alpha:{}'.format(step, model.attrib_vec))
            return SUCCESS, out_img, model.attrib_vec.cpu().detach().numpy(), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy()
        out_img = np.hstack(
            [orig_img, recon.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)])
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.info('Step:{}, loss:{}, alpha:{}'.format(
            step, loss.detach().cpu().numpy(), model.attrib_vec.detach().cpu().numpy()[0].tolist()))
        step += 1
        if loss.detach().cpu().numpy() - prev_loss < 0.1:
            pat_cnt += 1
        
    return FAILURE, out_img, model.attrib_vec.cpu().detach().numpy(), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy()

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.logger = get_logger(args.outdir)
    args.gen = torch.load(args.attgan)
    args.train_attribute = _SORTED_ATTR
    args.img_sz = 256
    args.img_fm = 3

    # ATTGAN weights
    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.attgan)))
    with open(os.path.join(par_dir, 'setting.txt'),'r') as f:
        args_gen = json.load(f, object_hook=lambda d:argparse.Namespace(**d))
    args.batch_size = 1
    # Minor bug with storing the model is creating this issue. Anyway, there is not much speedup with cuda
    device = torch.device('cpu')
    # Data Loader
    loader = get_data_loader(args, train='test', shuffle=False)

    cnt = 0
    f = open(os.path.join(args.outdir, 'vals.csv'), 'w')
    for input, label in tqdm(loader, total=len(loader)):
        attacker = Attacker(args, args_gen, label).to(device)
        attacker.restore()
        if cnt > 500:
            break
        if args.type == 'att':
            success, out_img, alpha, orig_logits, logits = attack_optim(
                input, attacker, [(-1.5, 1.75), (-2, 3), (-6, 7)], device, args.logger)
        elif args.type == 'rn':
            success, out_img, alpha, orig_logits, logits = attack_random(
                input, attacker, 10, device, args.logger, args.eps)
        if success:
            plt.imshow(out_img)
            plt.title('alpha:{}'.format(alpha))
            plt.savefig(os.path.join(
                args.outdir, '{}_broken.png'.format(str(cnt))))
        else:
            plt.imshow(out_img)
            plt.title('unbroken_alpha:{}'.format(alpha))
            plt.savefig(os.path.join(
                args.outdir, '{}_unbroken.png'.format(str(cnt))))
        np.save(os.path.join(args.outdir, '{}.npy'.format(str(cnt))), out_img)
        #print(loss.shape)
        out_dict = {'success': success, 'orig_logits': orig_logits[0].tolist(
        ), 'logits': logits[0].tolist(), 'alpha': alpha[0].tolist()}
        outstr = json.dumps(out_dict)
        f.write(outstr+'\n')
        cnt += 1
    f.close()


if __name__ == '__main__':
    main()
