from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.).type_as(x)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
    
def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice


def BCE(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce

def jaccard_loss(logits, target, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1).type_as(target)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).type_as(target)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

class Ohem(nn.Module):
    # Based on https://arxiv.org/pdf/1812.05802.pdf
    def __init__(self, fraction=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.fraction = fraction

    def forward(self, y_pred, y_true):
        batch_size = y_true.size(0)
        losses = self.loss(y_pred, y_true).view(batch_size, -1)

        positive_mask = (y_true > 0).view(batch_size, -1)
        Cp = torch.sum(positive_mask, dim=1)
        Cn = torch.sum(~positive_mask, dim=1)
        Chn = torch.max((Cn / 4).clamp_min(5), 2 * Cp)

        loss, num_samples = 0, 0
        for i in range(batch_size):
            positive_losses = losses[i, positive_mask[i]]
            negative_losses = losses[i, ~positive_mask[i]]
            num_negatives = int(Chn[i])
            hard_negative_losses, _ = negative_losses.sort(descending=True)[:num_negatives]
            loss = positive_losses.sum() + hard_negative_losses.sum() + loss
            num_samples += positive_losses.size(0)
            num_samples += hard_negative_losses.size(0)
        loss /= float(num_samples)

        return loss

class CORAL(nn.Module):
    # Adapted to image segmentation based on https://github.com/Raschka-research-group/coral-cnn
    def __init__(self):
        super(CORAL, self).__init__()
        self.levels = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)

    def forward(self, y_pred, y_true):
        device = y_pred.device
        levels = self.levels[y_true].to(device)
        logpt = F.logsigmoid(y_pred)
        loss = torch.sum(logpt * levels + (logpt - y_pred) * (1 - levels), dim=1)
        return -torch.mean(loss)
    

def get_loss(args, device):
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal":
        args.alpha = torch.tensor(list(map(float, args.alpha.split(',')))).to(device)
        criterion = FocalLoss(args.alpha, args.gamma)
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'bce_dice':
        criterion = BCEDiceLoss
    elif args.loss == 'jaccard':
        criterion = jaccard_loss
    elif args.loss == 'ohem':
        criterion = Ohem()
    elif args.loss == 'coral':
        criterion = CORAL()
    else:
        print("Loss not found")
        return
    
    return criterion