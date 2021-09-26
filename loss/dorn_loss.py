import torch
import torch.nn as nn

"""
refer to https://github.com/liviniuk/DORN_depth_estimation_Pytorch
"""


class OrdinalLoss(nn.Module):
    """
    Ordinal loss as defined in the paper "DORN for Monocular Depth Estimation".
    refer to https://github.com/liviniuk/DORN_depth_estimation_Pytorch
    """

    def __init__(self):
        super(OrdinalLoss, self).__init__()

    def forward(self, pred_softmax, target_labels):
        """
        :param pred_softmax:    predicted softmax probabilities P
        :param target_labels:   ground truth ordinal labels
        :return:                ordinal loss
        """

        n, c = pred_softmax.size()  # C - number of discrete sub-intervals (= number of channels)
        target_labels = target_labels.int().view(n, 1)

        K = torch.zeros((n, c), dtype=torch.int).cuda()
        for i in range(c):
            K[:, i] = K[:, i] + i * torch.ones(n, dtype=torch.int).cuda()

        mask = (K <= target_labels).detach()

        loss = pred_softmax[mask].clamp(1e-8, 1e8).log().sum() + (1 - pred_softmax[~mask]).clamp(1e-8, 1e8).log().sum()
        loss /= -n
        return loss
