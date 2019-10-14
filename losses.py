"""
Carlini-Wagner attacks
"""
import torch
import numpy as np
from torch import nn


def fgsm(model, x, target, eps, targeted=True, device=None, clip_min=None, clip_max=None):
    input = x.clone().detach_().to(device)
    input.requires_grad_()
    target = torch.LongTensor(torch.argmax(target).unsqueeze(0).to(device))
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


def pgd(model, x, target, k, eps, eps_step, targeted=True, device=None, clip_min=None,
        clip_max=None):
    x_min = x - eps
    x_max = x + eps
    if device is None:
        device = torch.device('cpu')
    # generate random point in +-eps box around x
    x = 2. * eps * torch.rand_like(x) - eps
    success = 0
    logits_array = [model(x).detach().cpu().numpy().tolist()]
    for i in range(k):
        # FGSM step
        x = fgsm(model, x, target, eps_step, targeted, device=device)
        logits = model(x)
        logits_array.append(logits.detach().cpu().numpy().tolist())
        # projection step
        x = torch.max(x_min.to(device), x)
        x = torch.min(x_max.to(device), x)
        if torch.argmax(logits) != torch.argmax(target):
            success = 1
            break

    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)

    return x, success, logits, eps_step, logits_array


def attack_cw_l2(img, model, epsilon, device, logger):
    """
    Attack using Carlini-Wagner L2 attack
    """
    MAX_ITER = 500
    step = 0
    model.train()
    target = model.target_model(img.to(device))
    img_mod, success, logits, eps, logits_array = pgd(
        model.target_model, img, target=target, k=500, eps=0.005, eps_step=0.005, targeted=False, device=device)
    out_img = np.hstack([img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0),
                         img_mod.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)])
    loss = nontarget_logit_loss(logits, torch.argmax(target))
    return success, out_img, np.float32([eps]), target.detach().cpu().numpy(), logits.detach().cpu().numpy(), loss.detach().cpu().numpy(), logits_array


def nontarget_logit_loss(logit, label, nclasses):
    # Dummy vector for one-hot label vector. For multi-class, change this to # of classes
    Y_ = torch.zeros(1, nclasses)
    Y_[0, label] = 1.0
    actual_logits = (Y_*logit).sum(1)
    nonactual_logits = ((1-Y_)*logit - Y_*10000).max(1)[0]
    model_loss = torch.clampk(actual_logits - nonactual_logits, min=0.0).sum()
    return model_loss
