"""
Attacking the model using a Fader Network
Note: We are basically searching for the interpolation values
which allow us to break a simple classifier. 
"""

import argparse
import logging
import os
from collections import OrderedDict
from datetime import datetime
import json

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from FaderNetworks.src.model import AutoEncoder
from simple_classifier import Classifier, get_data_loader, restore_model
from utils import get_logger
from losses import nontarget_logit_loss

class Params():
    """
       Helper class for parameters for fader networks
    """
    def __init__(self):
        self.img_sz = 256
        self.img_fm = 3
        self.init_fm = 32
        self.max_fm = 512
        self.hid_dim = 512
        self.n_layers = 6
        self.n_attr = 2
        self.n_skip = 0
        self.dec_dropout = 0.0
        self.deconv_method = 'convtranspose'
        self.attr = [('Eyeglasses', 2)]
        self.instance_norm = False


class AttEncoderModule(nn.Module):
    """
    Workaround class for constructing alpha for single attribute fader networks
    """

    def __init__(self, alpha):
        super(AttEncoderModule, self).__init__()
        self.a1 = torch.tensor(alpha, requires_grad=True)

    def forward(self):
        a1 = torch.stack([self.a1, torch.tensor(0.)])
        a1_ = torch.stack([torch.tensor(0.), self.a1])
        alpha1 = torch.tensor([1., 0.]) - a1 + a1_
        return alpha1.unsqueeze(0)


class Attacker(nn.Module):
    """
    Defines a attack system which can be optimized.
    Input passes through a pretrained fader network for modification.
    Input -> (Fader network) -> (Target model) -> Classification label

    Since Fader network requires that the attribute vector elements (alpha_i) be converted to (alpha_i, 1-alpha_i),
    we use the Mod alpha class to handle this change while preserving gradients.
    """

    def __init__(self, params):
        super(Attacker, self).__init__()
        self.params = params
        self.target_model = Classifier(
            (params.img_sz, params.img_sz, params.img_fm))
        self.adv_generator_1 = AutoEncoder(params.f1_params)
        self.adv_generator_2 = AutoEncoder(params.f2_params)
        self.adv_generator_3 = AutoEncoder(params.f3_params)
        self.attrib_1 = AttEncoderModule(0.)
        self.attrib_2 = AttEncoderModule(0.)
        self.attrib_3 = AttEncoderModule(0.)

    def restore(self, legacy=False):
        self.target_model.load_state_dict(torch.load(self.params.model))
        model_state_d_1 = torch.load(self.params.fader1)
        model_state_d_2 = torch.load(self.params.fader2)
        model_state_d_3 = torch.load(self.params.fader3)
        self.adv_generator_1.load_state_dict(model_state_d_1, strict=False)
        self.adv_generator_2.load_state_dict(model_state_d_2, strict=False)
        self.adv_generator_3.load_state_dict(model_state_d_3, strict=False)
    
    def forward(self, x, attrib_vector=None):
        if attrib_vector is None:
            self.attrib_v1 = self.attrib_1()
            self.attrib_v2 = self.attrib_2()
            self.attrib_v3 = self.attrib_3()
        else:
            self.attrib_v1 = attrib_vector[:,:2]
            self.attrib_v2 = attrib_vector[:,2:4]
            self.attrib_v3 = attrib_vector[:,4:]
           # print(attrib_vector.size())
        l_z_1 = self.adv_generator_1.encode(x)
        recon_1 = self.adv_generator_1.decode(l_z_1, self.attrib_v1)[-1]
        l_z_2 = self.adv_generator_2.encode(recon_1)
        recon_2 = self.adv_generator_2.decode(l_z_2, self.attrib_v2)[-1]
        l_z_3 = self.adv_generator_3.encode(recon_2)
        recon_3 = self.adv_generator_3.decode(l_z_3, self.attrib_v3)[-1]
        cl_label = self.target_model(recon_3)
        return recon_3, cl_label

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
        alpha_init = np.random.rand(3).astype(np.float32) * 2 * eps - eps
        alpha2 = np.zeros((1,6), dtype=np.float32)
        for j in range(alpha_init.shape[0]):
            alpha2[0,j*2] = 1.0 - alpha_init[j]
            alpha2[0,2*j+1] = alpha_init[j]
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
            #logger.info(
            #    'Broken-Step:{}, alpha:{}'.format(step, model.attrib_vec))
            return SUCCESS, out_img, alphas[worst_loss], orig_logits.detach().cpu().numpy(), worst_pred.detach().cpu().numpy(), worst_loss_val, []
        else:
            return FAILURE, out_img, alphas[worst_loss], orig_logits.detach().cpu().numpy(), worst_pred.detach().cpu().numpy(), worst_loss_val, []


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Path to the model to be attacked. As of now, we will use models trained with the simple_classifier script', required=True)
    parser.add_argument(
        '-f1', '--fader1', help='Path to fader network 1. Use one of the many Fader networks trained', required=True)
    parser.add_argument(
        '-f2', '--fader2', help='Path to fader network 2. Use one of the many Fader networks trained', required=True)
    parser.add_argument(
        '-f3', '--fader3', help='Path to fader network 3. Use one of the many Fader networks trained', required=True)
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='out', required=False)
    parser.add_argument('-d', '--data_dir',
                        help='Path to data directory', required=True)
    parser.add_argument('-a', '--attrib_path',
                        help='Path to attrib file', required=True)
    parser.add_argument('-t', '--type', help='Attack type: \n\t fn: Fader Network based attack \n\t rn: Random Sampling',
                        choices=['fn', 'rn'], default='fn')
    return parser


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
    optim = torch.optim.Adam(
        [model.attrib_1.a1, model.attrib_2.a1, model.attrib_3.a1], lr=0.1)
    loss = np.inf
    logit_arrays = [orig_logits.detach().cpu().numpy().tolist()]
    while loss != 0.0 and step < MAX_ITER:
        recon, logits = model(img)
        logit_arrays.append(logits.detach().cpu().numpy().tolist())
        pred = torch.argmax(logits)
        logger.debug('Modified label:%s, Orig label:%s', pred, labels)
        loss = nontarget_logit_loss(logits, labels)
        # print('Loss:{}'.format(loss.detach().cpu().numpy()))
        if pred.cpu().detach().numpy() != labels.cpu().detach().numpy():
            out_img = np.hstack(
                [orig_img, recon.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)])
            logger.info(
                'Broken-Step:{}, alpha:{}'.format(step, model.attrib_v1))
            return SUCCESS, out_img, np.asarray([model.attrib_v1.cpu().detach().numpy(), model.attrib_v2.cpu().detach().numpy(), model.attrib_v3.cpu().detach().numpy()]), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy(), loss.detach().cpu().numpy(), logit_arrays
        out_img = np.hstack(
            [orig_img, recon.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)])
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.info('Step:{}, loss:{}, alpha:{}'.format(
            step, loss.detach().cpu().numpy(), model.attrib_v1.detach().cpu().numpy()[0].tolist()))
        step += 1
    return FAILURE, out_img, np.asarray([model.attrib_v1.cpu().detach().numpy(), model.attrib_v2.cpu().detach().numpy(), model.attrib_v3.cpu().detach().numpy()]), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy(), loss.detach().cpu().numpy(), logit_arrays

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.logger = get_logger(args.outdir)
    args.gen1 = torch.load(args.fader1)
    args.gen2 = torch.load(args.fader2)
    args.gen3 = torch.load(args.fader3)
    # Issue: Since Fader Network guys store the entrie model, it assumes certain paths.
    # We try to fix it by saving only the weights (state_dict) using mod_fader_network.py
    # Therefore, we need to reconstruct the model using the parameters.
    args.img_sz = 256
    args.img_fm = 3
    args.init_fm = 32
    args.max_fm = 512
    args.hid_dim = 512
    args.n_layers = 6
    args.n_attr = 2
    args.n_skip = 0
    args.dec_dropout = 0.0
    args.deconv_method = 'convtranspose'
    args.attr = [('Eyeglasses', 2)]
    args.instance_norm = False

    ## Fader 1
    args.f1_params = Params()
    par_dir = os.path.dirname(os.path.abspath(args.fader1))
    with open(os.path.join(par_dir, 'setting.txt'),'r') as f:
        args_gen = json.load(f)
    args_d1 = vars(args.f1_params)
    args_d1.update(args_gen)

    ### Fader 2 (Young)
    args.f2_params = Params()
    par_dir = os.path.dirname(os.path.abspath(args.fader2))
    with open(os.path.join(par_dir, 'setting.txt'),'r') as f:
        args_gen = json.load(f)
    args_d2 = vars(args.f2_params)
    args_d2.update(args_gen)

    ### Fader 3 (Pointy nose)
    args.f3_params = Params()
    par_dir = os.path.dirname(os.path.abspath(args.fader3))
    with open(os.path.join(par_dir, 'setting.txt'),'r') as f:
        args_gen = json.load(f)
    args_d3 = vars(args.f3_params)
    args_d3.update(args_gen)

    args.batch_size = 1
    args.train_attribute = 'Male'
    # Minor bug with storing the model is creating this issue. Anyway, there is not much speedup with cuda
    device = torch.device('cpu')
    # Data Loader
    loader = get_data_loader(args, train='test', shuffle=False)

    cnt = 0
    f = open(os.path.join(args.outdir, 'vals.csv'), 'w')
    full_logits = []
    for input, label in tqdm(loader, total=len(loader)):
        attacker = Attacker(args).to(device)
        attacker.restore()
        if cnt > 500:
            break
        if args.type == 'fn':
            success, out_img, alpha, orig_logits, logits, loss, logit_array = attack_optim(
                input, attacker, [(-1.5, 1.75), (-2, 3), (-6, 7)], device, args.logger)
        elif args.type == 'cw':
            success, out_img, alpha, orig_logits, logits, loss, logit_array = attack_cw_l2(
                input, attacker, 0.5, device, args.logger)
        elif args.type == 'rn':
            success, out_img, alpha, orig_logits, logits, loss, logit_array = attack_random(
                input, attacker, 0.5, device, args.logger, 3.0)

        print('Loss:{}'.format(loss))
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
        print(loss.shape)
        out_dict = {'success': success, 'orig_logits': orig_logits[0].tolist(
        ), 'logits': logits[0].tolist(), 'alpha': alpha[0].tolist(), 'loss': loss.tolist(), 'logit_array':logit_array}
        outstr = json.dumps(out_dict)
        f.write(outstr+'\n')
        cnt += 1
    f.close()


if __name__ == '__main__':
    main()
