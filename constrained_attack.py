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


class ModAlpha(nn.Module):
    """
    Workaround class for constructing alpha
    """

    def __init__(self, alpha1, alpha2, alpha3):
        super(ModAlpha, self).__init__()
        self.a1 = torch.tensor(alpha1, requires_grad=True)
        self.a2 = torch.tensor(alpha2, requires_grad=True)
        self.a3 = torch.tensor(alpha3, requires_grad=True)

    def forward(self):
        a1 = torch.stack([self.a1, torch.tensor(0.)])
        a1_ = torch.stack([torch.tensor(0.), self.a1])
        alpha1 = torch.tensor([1., 0.]) - a1 + a1_
        a2 = torch.stack([self.a2, torch.tensor(0.)])
        a2_ = torch.stack([torch.tensor(0.), self.a2])
        alpha2 = torch.tensor([1., 0.]) - a2 + a2_
        a3 = torch.stack([self.a3, torch.tensor(0.)])
        a3_ = torch.stack([torch.tensor(0.), self.a3])
        alpha3 = torch.tensor([1., 0.]) - a3 + a3_
        return torch.cat([alpha1, alpha2, alpha3]).unsqueeze(0)


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
        self.attrib_gen = ModAlpha(0., 0., 0.)

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


def attack_linearly(img, model, attrib_tuple, device):
    # Just run a linear search of alpha
    model.eval()
    img = img.to(device)
    attrib = [1-attrib_tuple[0], attrib_tuple[0]]
    attrib_step = 1e-3
    recon_imgs = []
    true_label = torch.argmax(model.target_model(img))
    breaks = []
    pred_values = []
    cnt = 0
    alphas = np.arange(attrib_tuple[0], attrib_tuple[1], attrib_step)
    print(alphas)
    recons = []
    b_recons = []
    for alpha in alphas:
        attrib = torch.tensor([1-alpha, alpha]).unsqueeze(0).to(device)
        recon, pred = model(img, attrib)
        if torch.argmax(pred) != true_label:
            breaks.append(1)
            recons.append(np.hstack([img[0].detach().cpu().numpy().transpose(
                (1, 2, 0)), recon[0].detach().cpu().numpy().transpose((1, 2, 0))]))
        else:
            breaks.append(0)
            b_recons.append(np.hstack([img[0].detach().cpu().numpy().transpose(
                (1, 2, 0)), recon[0].detach().cpu().numpy().transpose((1, 2, 0))]))
        cnt += 1
    if len(recons) != 0:
        recon2 = np.vstack(recons[:5])
    else:
        recon2 = np.vstack(b_recons[-5:])
    recon3 = np.vstack(b_recons[:5])
    return breaks, pred_values, recon2, recon3

def attack_binarily(img, model, attrib_tuple, device, logger):
    """ Currently assuming that the alpha value of the attribute is positive.
    # Binary search only works for a single attribute
    # Binary search for alpha
    """
    model.eval()
    img = img.to(device)
    true_label = torch.argmax(model.target_model(img))
    attrib_left = torch.tensor(
        [1-attrib_tuple[0], attrib_tuple[0]]).unsqueeze(0).to(device)
    attrib_right = torch.tensor(
        [1-attrib_tuple[1], attrib_tuple[1]]).unsqueeze(0).to(device)
    recon_left, pred_recon_left = model(img, attrib_left)
    recon_right, pred_recon_right = model(img, attrib_right)
    pred_recon_left = torch.argmax(pred_recon_left)
    pred_recon_right = torch.argmax(pred_recon_right)

    if pred_recon_left == true_label and pred_recon_right == true_label:
        logger.info('Higher max value required')
        attrib_left = attrib_right
        attrib_right = attrib_right*2
    elif pred_recon_left != true_label and pred_recon_right != true_label:
        logger.info('Lower min value required')
        attrib_right = attrib_left
        attrib_left = attrib_left/2.0
    perm_attrib = attrib_right[0, 1]
    attrib = attrib_left + (attrib_right - attrib_left)/2.0
    cnt = 0
    max_iter = 10000
    while np.abs(attrib_left[0, 1].cpu().numpy() - attrib_right[0, 1].cpu().numpy()) > 1e-3:
        logger.info('Num_iter:{}, alpha:{}'.format(
            cnt, attrib[0, 1].cpu().numpy()))
        if cnt > max_iter:
            break
        recon_attrib, pred_attrib = model(img, attrib)
        label_attrib = torch.argmax(pred_attrib)
        if label_attrib != true_label:
            logger.info('Broken at least once, {}, {}'.format(
                true_label, label_attrib))
            attrib_right = attrib
            if np.abs(attrib[0, 1]) - np.abs(perm_attrib) <= 0:
                perm_attrib = attrib[0, 1]
            #attrib_left = attrib
        else:
            attrib_left = attrib
        attrib = attrib_left + (attrib_right - attrib_left)/2.0
        cnt += 1

    recon_attrib, pred_attrib = model(img, torch.tensor(
        [1-perm_attrib, perm_attrib]).unsqueeze(0).to(device))
    label_attrib = torch.argmax(pred_attrib)
    logger.info('True_label:{}, pred_label:{}'.format(
        true_label, label_attrib))
    return perm_attrib, recon_attrib, true_label, label_attrib


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.logger = get_logger(args.outdir)
    args.gen = torch.load(args.fader)
    # Issue: Since Fader Network guys store the entrie model, it assumes certain paths.
    # We try to fix it by saving only the weights (state_dict) using mod_fader_network.py
    # Therefore, we need to reconstruct the model using the parameters.
    args.img_sz = 256
    args.img_fm = 3
    args.init_fm = 32
    args.max_fm = 512
    args.n_layers = 7
    args.n_attr = 6
    args.n_skip = 0
    args.dec_dropout = 0.0
    args.deconv_method = 'convtranspose'
    args.attr = [('Eyeglasses', 2)]
    args.instance_norm = False

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
        ), 'logits': logits[0].tolist(), 'alpha': alpha[0].tolist(), 'loss': loss.tolist(), 'logit_array': logit_array}
        outstr = json.dumps(out_dict)
        f.write(outstr+'\n')
        cnt += 1
    f.close()


if __name__ == '__main__':
    main()
