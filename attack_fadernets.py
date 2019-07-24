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

from FaderNetworks.src.model import AutoEncoder
from simple_classifier import Classifier, get_data_loader, restore_model
from utils import get_logger
from losses import attack_cw_l2, nontarget_logit_loss


class AttEncoderModule(nn.Module):
    """
    Class for constructing the attribute vector. In this case, works for 3 attributes.
    """

    def __init__(self, alpha1, alpha2, alpha3, projection_step=False, eps=1.0):
        super(AttEncoderModule, self).__init__()
        self.a1 = torch.tensor(alpha1, requires_grad=True)
        self.a2 = torch.tensor(alpha2, requires_grad=True)
        self.a3 = torch.tensor(alpha3, requires_grad=True)
        self.projection_step = projection_step
        self.eps = torch.tensor(eps)

    def forward(self):
        # projection step
        a1_p = self.a1
        a2_p = self.a2
        a3_p = self.a3
        if self.projection_step:
            a1_p = torch.min(a1_p, self.eps)
            a1_p = torch.max(a1_p, -self.eps)
            a2_p = torch.min(a2_p, self.eps)
            a2_p = torch.max(a2_p, -self.eps)
            a3_p = torch.min(a3_p, self.eps)
            a3_p = torch.max(a3_p, -self.eps)
        a1 = torch.stack([a1_p, torch.tensor(0.)])
        a1_ = torch.stack([torch.tensor(0.), a1_p])
        alpha1 = torch.tensor([1., 0.]) - a1 + a1_
        a2 = torch.stack([a2_p, torch.tensor(0.)])
        a2_ = torch.stack([torch.tensor(0.), a2_p])
        alpha2 = torch.tensor([1., 0.]) - a2 + a2_
        a3 = torch.stack([a3_p, torch.tensor(0.)])
        a3_ = torch.stack([torch.tensor(0.), a3_p])
        alpha3 = torch.tensor([1., 0.]) - a3 + a3_
        alpha_vec = torch.cat([alpha1, alpha2, alpha3])
        # Projection step
        alpha_vec = alpha_vec.unsqueeze(0)
        return alpha_vec


class Attacker(nn.Module):
    """
    Defines a attack system which can be optimized.
    Input passes through a pretrained fader network for modification.
    Input -> (Fader network) -> (Target model) -> Classification label

    Since Fader network requires that the attribute vector elements (alpha_i) be converted to (alpha_i, 2-alpha_i),
    we use the Mod alpha class to handle this change while preserving gradients.
    """

    def __init__(self, params):
        super(Attacker, self).__init__()
        self.params = params
        self.target_model = Classifier(
            (params.img_sz, params.img_sz, params.img_fm))
        self.adv_generator = AutoEncoder(params)
        self.eps = params.eps
        self.projection = params.proj_flag
        self.attrib_gen = AttEncoderModule(1.0, 1.0, 1.0, self.projection, self.eps)

    def restore(self, legacy=False):
        self.target_model.load_state_dict(torch.load(self.params.model))
        if legacy:
            old_model_state_dict = torch.load(self.params.fader)
            old_model_state_dict.update(_LEGACY_STATE_DICT_PATCH)
            model_state_d = old_model_state_dict
        else:
            model_state_d = torch.load(self.params.fader)
        self.adv_generator.load_state_dict(model_state_d, strict=False)

    def forward(self, x, attrib_vector=None):
        self.attrib_vec = self.attrib_gen()
        l_z = self.adv_generator.encode(x)
        recon = self.adv_generator.decode(l_z, self.attrib_vec)[-1]
        #recon = recon / recon.max()
        cl_label = self.target_model(recon)
        return recon, cl_label


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Path to the model to be attacked. As of now, we will use models trained with the simple_classifier script', required=True)
    parser.add_argument(
        '-f', '--fader', help='Path to fader networks. Use one of the many Fader networks trained', required=True)
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='out', required=False)
    parser.add_argument('-d', '--data_dir',
                        help='Path to data directory', required=True)
    parser.add_argument('-a', '--attrib_path',
                        help='Path to attrib file', required=True)
    parser.add_argument('-t', '--type', help='Attack type: \n\t cw: Carlini-Wagner L2 attack \n\t fn: Fader Network based attack \n\t rn: Random attack',
                        choices=['cw', 'fn', 'rn'], default='fn')
    parser.add_argument('--proj_flag', action='store_true',
                        help='Infinity projection flag')
    parser.add_argument('--eps', help='Epsilon value', default=4.0, type=float)
    return parser


def get_abs_val(x):
    with torch.no_grad():
        val = torch.abs(x).detach().cpu().numpy()
    return val


def attack_optim(img, model, attrib_tuple, device, logger, eps):
    """
    Optimizer based attack. 
    Runs a reverse gradient to find the value of alpha vector
    which breaks the target model.
    """
    MAX_ITER = 250
    step = 0
    model.train()
    model = model.to(device)
    orig_img = img.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)
    img = img.to(device)
    orig_logits = model.target_model(img)
    labels = torch.argmax(orig_logits)
    SUCCESS = 1
    FAILURE = 0
    optim = torch.optim.Adam(
        [model.attrib_gen.a1, model.attrib_gen.a2, model.attrib_gen.a3], lr=0.1)
    loss = np.inf
    logit_arrays = [orig_logits.detach().cpu().numpy().tolist()]
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
            return SUCCESS, out_img, model.attrib_vec.cpu().detach().numpy(), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy(), loss.detach().cpu().numpy(), logit_arrays
        out_img = np.hstack(
            [orig_img, recon.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)])
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.info('Step:{}, loss:{}, alpha:{}'.format(
            step, loss.detach().cpu().numpy(), model.attrib_vec.detach().cpu().numpy()[0].tolist()))
        with torch.no_grad():
            a1 = get_abs_val(model.attrib_gen.a1)
            a2 = get_abs_val(model.attrib_gen.a2)
            a3 = get_abs_val(model.attrib_gen.a3)
            print(a1, a2, a3)
            if a1 >= eps and a2 >= eps and a3 >= eps:
                break
        step += 1
    return FAILURE, out_img, model.attrib_vec.cpu().detach().numpy(), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy(), loss.detach().cpu().numpy(), logit_arrays


def attack_random(img, model, num_samples, device, logger, eps):
    """
    Optimizer based attack.
    Iterates over randomly generated attribute vector values and picks the one with
    the worst loss value.
    """
    MAX_ITER = 500
    step = 0
    model.train()
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
        alpha_init = np.random.rand(3) * 2 * eps - eps
        alpha2 = np.zeros((1, 6))
        for j in range(alpha_init.shape[0]):
            alpha2[0, j*2] = 1.0 - alpha_init[j]
            alpha2[0, 2*j+1] = alpha_init[j]
        alphas.append(torch.tensor(alpha2))

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
            return SUCCESS, out_img, alphas[worst_loss], orig_logits.detach().cpu().numpy(), worst_pred.detach().cpu().numpy(), worst_loss_val, []
        else:
            return FAILURE, out_img, alphas[worst_loss], orig_logits.detach().cpu().numpy(), worst_pred.detach().cpu().numpy(), worst_loss_val, []


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.logger = get_logger(args.outdir)
    args.gen = torch.load(args.fader)
    
    par_dir = os.path.dirname(os.path.abspath(args.fader))
    with open(os.path.join(par_dir, 'setting.txt'),'r') as f:
        args_gen = json.load(f)
    args_d = vars(args)
    args_d.update(args_gen)

    args.batch_size = 1
    args.train_attribute = 'Male'
    # Minor bug with storing the model is creating this issue. Anyway, there is not much speedup with cuda
    device = torch.device('cpu')
    # Data Loader
    loader = get_data_loader(args, train='test', shuffle=False)

    cnt = 0
    f = open(os.path.join(args.outdir, 'vals.csv'), 'w')
    for input, label in tqdm(loader, total=len(loader)):
        attacker = Attacker(args).to(device)
        attacker.restore()
        if cnt > 500:
            break
        if args.type == 'fn':
            success, out_img, alpha, orig_logits, logits, loss, logit_array = attack_optim(
                input, attacker, [(-1.5, 1.75), (-2, 3), (-6, 7)], device, args.logger, args.eps)
        elif args.type == 'cw':
            success, out_img, alpha, orig_logits, logits, loss, logit_array = attack_cw_l2(
                input, attacker, 0.5, device, args.logger)
        elif args.type == 'rn':
            success, out_img, alpha, orig_logits, logits, loss, logit_array = attack_random(
                input, attacker, 10, device, args.logger, args.eps)
        args.logger.info('Loss:{}'.format(loss))
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
        out_dict = {'success': success, 'orig_logits': orig_logits[0].tolist(
        ), 'logits': logits[0].tolist(), 'alpha': alpha[0].tolist(), 'loss': loss.tolist(), 'logit_array': logit_array}
        outstr = json.dumps(out_dict)
        f.write(outstr+'\n')
        cnt += 1
    f.close()


if __name__ == '__main__':
    main()
