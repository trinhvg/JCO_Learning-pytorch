import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CEOLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
    """
    def __init__(self, num_classes=4):
        super(CEOLoss, self).__init__()
        self.num_classes = num_classes
        self.level = torch.arange(self.num_classes)

    def forward(self, x, y):
        """"
        Args:
            x (tensor): Regression/ordinal output, size (B), type: float
            y (tensor): Ground truth,  size (B), type: int/long

        Returns:
            CEOLoss: Cross-Entropy Ordinal loss
        """
        levels = self.level.repeat(len(y), 1).cuda()
        logit = x.repeat(self.num_classes, 1).permute(1, 0)
        logit = torch.abs(logit - levels)
        return F.cross_entropy(-logit, y, reduction='mean')



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=None, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class SoftLabelOrdinalLoss(nn.Module):
    def __init__(self, alpha=1.):
        super(SoftLabelOrdinalLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        """Validates model name.

        Args:
            x (Tensor): [0, 1, 2, 3]
            y (Tensor): [0, 1, 2, 3]

        Returns:
            loss: scalar
        """
        # y /= 3
        # y /= 2
        x = torch.sigmoid(x)
        soft_loss = -(1 - y) * torch.log(1 - x) - self.alpha * y * torch.log(x)
        return torch.mean(soft_loss)



def label_to_levels(label, num_classes=4):
    levels = [1] * label + [0] * (num_classes - 1 - label)
    levels = torch.tensor(levels, dtype=torch.float32)
    return levels


def labels_to_labels(class_labels, num_classes =4):
    """
    class_labels = [2, 1, 3]
    """
    levels = []
    for label in class_labels:
        levels_from_label = label_to_levels(int(label), num_classes=num_classes)
        levels.append(levels_from_label)
    return torch.stack(levels).cuda()


def cost_fn(logits, label):
    num_classes = 3 #Note
    imp = torch.ones(num_classes - 1, dtype=torch.float).cuda()
    levels = labels_to_labels(label, num_classes)
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1] * levels
                       + F.log_softmax(logits, dim=2)[:, :, 0] * (1 - levels)) * imp, dim=1))
    return torch.mean(val)


def loss_fn2(logits, label):
    num_classes = 3 #Note
    imp = torch.ones(num_classes - 1, dtype=torch.float)
    levels = labels_to_labels(label)
    val = (-torch.sum((F.logsigmoid(logits) * levels
                       + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                      dim=1))
    return torch.mean(val)


class FocalOrdinalLoss(nn.Module):
    def __init__(self, alpha=0.75, pooling=False, num_classes=4):
        super(FocalOrdinalLoss, self).__init__()
        self.alpha = alpha
        self.pooling = pooling
        self.num_classes = num_classes

    def forward(self, x, y):
        # convert one-hot y to ordinal y
        levels = labels_to_labels(y, num_classes=self.num_classes)
        q, _ = torch.max(levels*(1-x)**2 + (1-levels)*x**2, dim=1)
        if self.pooling:
            q = q.unsqueeze(0)
            q = q.unsqueeze(0)
            q = nn.MaxPool1d(3, 1, padding=1)(q)
        x = torch.sigmoid(x)
        # compute the loss
        f_loss = q*torch.sum(-self.alpha*levels*torch.log(x) - (1-self.alpha)*(1-levels)*torch.log(1-x))
        return torch.mean(f_loss)





def count_pred(x):
    N = x.shape[0]
    x = x.cuda() > 0.5
    pred = torch.zeros(N).long().cuda()
    pred = pred.view(N, 1)
    for i in range(x.shape[1]):
        pred_i = x[:, :i+1].prod(1)*x[:, :i+1].sum(1)
        pred = torch.cat([pred, pred_i.view(N, 1)], dim =1)
    return pred.max(1)[0]


# #
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# def test():
#     # x = torch.Tensor([[0.7, 0.5, 0.6], [0.5, 0.8, 0.2], [0.8, 0.6, 0.1], [0.1, 0.5, 0.6]])
#     # y = torch.Tensor([1., 2., 3., 0.])
#     # x = x.to("cuda")
#     # y = y.to("cuda")
#     # FocalOrdinalLoss()(x, y)
#     # count_pred(x)
#
#
#     x = torch.Tensor([0.7, 2., 0.6, 1.])
#     y = torch.Tensor([1, 2, 3, 0])
#     x = x.to("cuda")
#     y = y.to("cuda")
#     CEOLoss(4)(x, y)
#     count_pred(x)
#
# test()
#
# #
# #
# #











