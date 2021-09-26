import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def mae_cancer_v0(input, target):
    input_ = input[target != 0]
    target_ = target[target != 0]
    return F.l1_loss(input_, target_) if len(target_) != 0 else 0


# def mse_cancer(input, target):
#     input_ = input[target != 0]
#     target_ = target[target != 0]
#     return F.mse_loss(input_, target_) if len(target_) != 0 else 0


def mse_cancer_v0(input, target):
    input_ = input[target != 0]
    target_ = target[target != 0]
    return F.mse_loss(input_, target_) if len(target_) != 0 else 0


def ceo_cancer_v0(input, target):
    input_ = input[target != 0]
    target_ = target[target != 0]
    if len(target_) == 0:
        return 0
    label_ = torch.tensor([1., 2., 3.]).repeat(len(target_), 1).cuda()
    logit_proposed_ = input_.repeat(3, 1).permute(1, 0)
    logit_proposed_ = torch.abs(logit_proposed_ - label_)
    return F.cross_entropy(-logit_proposed_, target_ - 1)

def mae_cancer(input, target):
    mae_loss = F.l1_loss(input, target, reduction='none')
    select = torch.randint(0, 2, (target.shape[0],)).float().cuda()  * torch.sign(target)
    return (mae_loss*select).mean()


# def mse_cancer(input, target):
#     input_ = input[target != 0]
#     target_ = target[target != 0]
#     return F.mse_loss(input_, target_) if len(target_) != 0 else 0


def mse_cancer(input, target):
    mse_loss = F.mse_loss(input, target, reduction='none')
    # print(mse_loss.shape)
    # print(torch.sign(target).shape)
    # print(torch.sign(target))
    select = torch.randint(0, 2, (target.shape[0],)).float().cuda() * torch.sign(target)
    return (mse_loss*select).mean()


def ceo_cancer(input, target):
    label = torch.tensor([0., 1., 2., 3.]).repeat(len(target), 1).cuda()
    logit_proposed = input.repeat(4, 1).permute(1, 0)
    logit_proposed = torch.abs(logit_proposed - label)
    ceo_loss = F.cross_entropy(-logit_proposed, target, reduction='none')
    # select = (torch.randint(0, 2, (target.shape[0],)).cuda()  * torch.sign(target)).float()
    select = torch.sign(target).float()
    return (ceo_loss*select).mean()

# class CeoCancer:
#     def __init__(self, ):
#         super(CeoCancer, self).__init__()
