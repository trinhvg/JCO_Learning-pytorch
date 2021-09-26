# !/usr/bin/env python
# coding=utf-8
"""
https://github.com/liulihao-cuhk/MTMR-NET
"""
import os
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import torch
import math

def get_loss_mtmr(output_score_1, cat_subtlety_score, gt_score_1, gt_attribute_score_1):
    xcentloss_func_1 = nn.CrossEntropyLoss()
    xcentloss_1 = xcentloss_func_1(output_score_1, gt_score_1)

    # ranking loss
    ranking_loss_sum = 0
    half_size_of_output_score = output_score_1.size()[0] // 2
    for i in range(half_size_of_output_score):
        tmp_output_1 = output_score_1[i]
        tmp_output_2 = output_score_1[i + half_size_of_output_score]
        tmp_gt_score_1 = gt_score_1[i]
        tmp_gt_score_2 = gt_score_1[i + half_size_of_output_score]

        rankingloss_func = nn.MarginRankingLoss()

        if tmp_gt_score_1.item() != tmp_gt_score_2.item():
            target = torch.ones(1) * -1
            ranking_loss_sum += rankingloss_func(tmp_output_1, tmp_output_2, Variable(target.cuda()))
        else:
            target = torch.ones(1)
            ranking_loss_sum += rankingloss_func(tmp_output_1, tmp_output_2, Variable(target.cuda()))

    ranking_loss = ranking_loss_sum / half_size_of_output_score

    # attribute loss
    attribute_mseloss_func_1 = nn.MSELoss()
    attribute_mseloss_1 = attribute_mseloss_func_1(cat_subtlety_score, gt_attribute_score_1.float())

    loss = 1 * xcentloss_1 + 5.0e-1 * ranking_loss + 1.0e-3 * attribute_mseloss_1

    return loss
