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
from losses import nontarget_logit_loss

_LEGACY_STATE_DICT_PATCH = OrderedDict([("enc_layers.1.1.num_batches_tracked", 1), ("enc_layers.2.1.num_batches_tracked", 1),
                                        ("enc_layers.3.1.num_batches_tracked",
                                         1), ("enc_layers.4.1.num_batches_tracked", 1),
                                        ("enc_layers.5.1.num_batches_tracked",
                                         1), ("enc_layers.6.1.num_batches_tracked", 1),
                                        ("dec_layers.0.1.num_batches_tracked",
                                         1), ("dec_layers.1.1.num_batches_tracked", 1),
                                        ("dec_layers.2.1.num_batches_tracked",
                                         1), ("dec_layers.3.1.num_batches_tracked", 1),
                                        ("dec_layers.4.1.num_batches_tracked", 1), ("dec_layers.5.1.num_batches_tracked", 1)])
# torch.nn.Module.dump_patches=True


class AttEncoderModule(nn.Module):
    """
    Workaround class for constructing alpha
    """

    def __init__(self, alpha1):
        super(AttEncoderModule, self).__init__()
        self.a1 = torch.tensor(alpha1, requires_grad=True)
        
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
        self.adv_generator = AutoEncoder(params)
        self.attrib_gen = AttEncoderModule(0.)

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
    parser.add_argument('-t', '--type', help='Attack type: \n\t cw: Carlini-Wagner L2 attack \n\t fn: Fader Network based attack',
                        choices=['cw', 'fn'], default='fn')
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
        [model.attrib_gen.a1], lr=0.1)
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
                'Broken-Step:{}, alpha:{}'.format(step, model.attrib_vec))
            return SUCCESS, out_img, model.attrib_vec.cpu().detach().numpy(), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy(), loss.detach().cpu().numpy(), logit_arrays
        out_img = np.hstack(
            [orig_img, recon.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)])
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.info('Step:{}, loss:{}, alpha:{}'.format(
            step, loss.detach().cpu().numpy(), model.attrib_vec.detach().cpu().numpy()[0].tolist()))
        step += 1
    return FAILURE, out_img, model.attrib_vec.cpu().detach().numpy(), orig_logits.detach().cpu().numpy(), logits.detach().cpu().numpy(), loss.detach().cpu().numpy(), logit_arrays


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
    full_logits = []
    for input, label in tqdm(loader, total=len(loader)):
        attacker = Attacker(args).to(device)
        attacker.restore()
        if cnt > 500:
            break
        if args.type == 'fn':
            success, out_img, alpha, orig_logits, logits, loss, logit_array = attack_optim(
                input, attacker, [(-1.5, 1.75), (-2, 3), (-6, 7)], device, args.logger)
        else:
            print('Please check attack_fadernets for more options')

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
        ), 'logits': logits[0].tolist(), 'alpha': alpha[0].tolist(), 'loss': loss.tolist(), 'logit_array': logit_array}
        outstr = json.dumps(out_dict)
        f.write(outstr+'\n')
        cnt += 1
    f.close()


if __name__ == '__main__':
    main()
