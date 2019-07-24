"""
Simple Modder for all the Fader Trained models
"""

import torch
import torchvision
import argparse
from FaderNetworks.src.model import AutoEncoder
from FaderNetworks.src.utils import initialize_exp, bool_flag, attr_flag, check_attr
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="default",
                    help="Experiment name")
parser.add_argument("--img_sz", type=int, default=256,
                    help="Image sizes (images have to be squared)")
parser.add_argument("--img_fm", type=int, default=3,
                    help="Number of feature maps (1 for grayscale, 3 for RGB)")
parser.add_argument("--attr", type=attr_flag, default="Smiling,Male",
                    help="Attributes to classify")
parser.add_argument("--instance_norm", type=bool_flag, default=False,
                    help="Use instance normalization instead of batch normalization")
parser.add_argument("--init_fm", type=int, default=32,
                    help="Number of initial filters in the encoder")
parser.add_argument("--max_fm", type=int, default=512,
                    help="Number maximum of filters in the autoencoder")
parser.add_argument("--n_layers", type=int, default=6,
                    help="Number of layers in the encoder / decoder")
parser.add_argument("--n_skip", type=int, default=0,
                    help="Number of skip connections")
parser.add_argument("--deconv_method", type=str, default="convtranspose",
                    help="Deconvolution method")
parser.add_argument("--hid_dim", type=int, default=512,
                    help="Last hidden layer dimension for discriminator / classifier")
parser.add_argument("--dec_dropout", type=float, default=0.,
                    help="Dropout in the decoder")
parser.add_argument("--lat_dis_dropout", type=float, default=0.3,
                    help="Dropout in the latent discriminator")
parser.add_argument("--n_lat_dis", type=int, default=1,
                    help="Number of latent discriminator training steps")
parser.add_argument("--n_ptc_dis", type=int, default=0,
                    help="Number of patch discriminator training steps")
parser.add_argument("--n_clf_dis", type=int, default=0,
                    help="Number of classifier discriminator training steps")
parser.add_argument("--smooth_label", type=float, default=0.2,
                    help="Smooth label for patch discriminator")
parser.add_argument("--lambda_ae", type=float, default=1,
                    help="Autoencoder loss coefficient")
parser.add_argument("--lambda_lat_dis", type=float, default=0.0001,
                    help="Latent discriminator loss feedback coefficient")
parser.add_argument("--lambda_ptc_dis", type=float, default=0,
                    help="Patch discriminator loss feedback coefficient")
parser.add_argument("--lambda_clf_dis", type=float, default=0,
                    help="Classifier discriminator loss feedback coefficient")
parser.add_argument("--lambda_schedule", type=float, default=500000,
                    help="Progressively increase discriminators' lambdas (0 to disable)")
parser.add_argument("--v_flip", type=bool_flag, default=False,
                    help="Random vertical flip for data augmentation")
parser.add_argument("--h_flip", type=bool_flag, default=True,
                    help="Random horizontal flip for data augmentation")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--ae_optimizer", type=str, default="adam,lr=0.0002",
                    help="Autoencoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0002",
                    help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--clip_grad_norm", type=float, default=5,
                    help="Clip gradient norms (0 to disable)")
parser.add_argument("--n_epochs", type=int, default=1000,
                    help="Total number of epochs")
parser.add_argument("--epoch_size", type=int, default=50000,
                    help="Number of samples per epoch")
parser.add_argument("--ae_reload", type=str, default="",
                    help="Reload a pretrained encoder")
parser.add_argument("--lat_dis_reload", type=str, default="",
                    help="Reload a pretrained latent discriminator")
parser.add_argument("--ptc_dis_reload", type=str, default="",
                    help="Reload a pretrained patch discriminator")
parser.add_argument("--clf_dis_reload", type=str, default="",
                    help="Reload a pretrained classifier discriminator")
parser.add_argument("--eval_clf", type=str, default="",
                    help="Load an external classifier for evaluation")
parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Debug mode (only load a subset of the whole dataset)")
params = parser.parse_args()

params.n_attr = 6

os.system('cd FaderNetworks')

stored_model = torch.load('/data/FaderNetworks/models/eye_smile_young_ae.pth')
stored_state = stored_model.state_dict()

f = open('eye_smile_young_state.pth', 'wb')
torch.save(stored_state, f)
#ae2 = AutoEncoder(params)

#ae2.load_state_dict(stored_state, strict=False)



