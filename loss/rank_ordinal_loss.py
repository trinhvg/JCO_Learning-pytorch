import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
https://github.com/Raschka-research-group/coral-cnn/blob/master/coral-implementation-recipe.ipynb
"""


def label_to_levels(label, num_classes=4):
    levels = [1] * label + [0] * (num_classes - 1 - label)
    levels = torch.tensor(levels, dtype=torch.float32)
    return levels


def labels_to_labels(class_labels, num_classes):
    """
    class_labels = [2, 1, 3]
    """
    levels = []
    for label in class_labels:
        levels_from_label = label_to_levels(int(label), num_classes=num_classes)
        levels.append(levels_from_label)
    return torch.stack(levels).cuda()


def cost_fn(logits, label, num_classes):
    imp = torch.ones(num_classes - 1, dtype=torch.float).cuda()
    levels = labels_to_labels(label, num_classes)
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1] * levels
                       + F.log_softmax(logits, dim=2)[:, :, 0] * (1 - levels)) * imp, dim=1))
    return torch.mean(val)


def loss_fn2(logits, label):
    num_classes = 4
    imp = torch.ones(num_classes - 1, dtype=torch.float)
    levels = labels_to_labels(label)
    val = (-torch.sum((F.logsigmoid(logits) * levels
                       + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                      dim=1))
    return torch.mean(val)
