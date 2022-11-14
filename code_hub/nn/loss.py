# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/24 9:27 上午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
code: https://github.com/shuxinyin/NLP-Loss-Pytorch/tree/master/unbalanced_loss
ref: https://mp.weixin.qq.com/s?src=11&timestamp=1648215069&ver=3698&signature=UuS45m84cZ-ixLRPKinIU-75vuiYFKpcRafdoQZoJNQ6kOznOTsafzLgOAXzo0K0wIzzT9d41YbnVytfBcqKsgky-*jm4pg2Cqkttae5LGLfS4duHxYccCFbwhNrLEQK&new=1
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal_Loss = -1 * alpha * (1-pt)log(pt)
    """

    def __init(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.reduction = reduction
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        # clamp: yi = min(max(xi, min_value), max_value)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)
        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()  # ?is necessary
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


class MultiFocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if self.alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        assert self.alpha.shape[0] == num_class

    def forward(self, logit, target):
        """

        :param logit: batch_size * 10
        :param target: exp: tensor([1, 3, 9, 6]) batch_size=4, num_class=10, batch_size * 1
        :return:
        """
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()
            prob = prob.view(-1, prob.size(-1))
        ori_shp = target.shape
        target = target.view(-1, 1) # 等价于target.unsqueeze(dim=1)
        # tensor([[7],
        #         [1],
        #         [5],
        #         [5]])-> target.view(-1, 1) example
        # 定义：从原tensor中获取指定dim和指定index的数据
        # 输入Index的shape等于输出value的shape
        # 输入index的索引值仅替换该index中对应dim的index值
        # 最终输出为替换index后在原tensor中的值
        #>>> t = torch.tensor([[1, 2], [3, 4]])
        # >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
        #  tensor([[ 1,  1],
        #         [ 4,  3]])
        prob = prob.gather(1, target).view(-1) + self.smooth
        logpt = torch.log(prob)

        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss


class WBCWithLogitLoss(nn.Module):
    """
    Weighted Binary cross entropy
    WBCE(p, t) = -beta * t * log(p) - (1-t)*log(1-p)
    to decrease the number of false negatives, set beta > 1
    to decrease the number of false positives, set beta < 1
    """
    def __init__(self, weight=1.0, ignore_index=None, reduction='mean'):
        super(WBCWithLogitLoss, self).__init__()
        assert reduction in ['mean', 'none', 'sum']
        self.ignore_index = ignore_index
        self.weight = float(weight)
        self.reduction = reduction
        self.smooth = 0.01

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0]

        if self.ignore_index is not None:
            # ignore_index, ignore_index处均置为0
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = output.view(batch_size, -1)

        output = torch.sigmoid(output)
        # avoid 'nan' loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0-eps)
        # soft label
        target = torch.clamp(target, min=self.smooth, max=1.0-self.smooth)
        # loss = self.bce(output, target)
        loss = -self.weight * target.mul(torch.log(output)) - ((1.0-target)).mul(torch.log(1.0-output))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class BinaryDiceLoss(nn.Module):
    """
    dice coefficient == F1 = 2TP/(2TP+FN+FP)，因此dice loss本质上是直接优化F1的
    F1是离散的，对于单个样本，可以改为连续式 dice = 2p * y/(p + y)
    dice loss = 1 - (2py+eps)/(p+y + eps)
    """
    def __init__(self, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1 # suggest set a large number when target area is large, like '10/100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_size = False
        if 'batch_loss' in kwargs.keys():
            self.batch_size = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0]
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_size:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs()+target.abs(), dim=1) + self.smooth

        loss = 1 - (num/den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('unexpected reduction')


class DiceLoss(nn.Module):
    """
    Args:
        weight: an array of shape [num_classes, ]
        ignore_index: specifies a target value that is ignored and does not contribute to the input graident
        output: a tensor of shape [N, C, *]
        target: a tensor of same shape with output
        other args pass to BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError('ignore_index type error')

    def forward(self, output, target):
        assert output.shape == target.shape
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        output = F.softmax(output, dim=1)
        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(output[:i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1]
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss


class BinaryDSCLoss(nn.Module):
    """
    Creates a criterion that optimizes a multi-class self-adjusting Dice Loss
    Dice Loss for Data-imbalanced NLP tasks - 香侬科技 Paper
    Args:
        alpha(float): a factor to push down the weight of easy examples
        gamma(float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction(string): specifies the reduction to apply to the output
    Shape:
        logits: (N, C) where N is the batch size and C is the number of classes
        targets: N where each values is in [0, C-1]
    """

    def __init__(self, alpha: float=1.0, smooth: float=1.0, reduction: str='mean')->None:
        super().__init__()
        self.apha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.gather(probs, dim=1, index=targets.unsqueze(1))
        targets = targets.unsqueze(dim=1)
        pos_mask = (targets == 1).float() # fixme
        neg_mask = (targets == 0).float()
        pos_weight = pos_mask * ((1 - probs) ** self.alpha) * probs
        pos_loss = 1 - (2 * pos_weight + self.smooth) / (pos_weight + 1 + self.smooth)

        neg_weight = neg_mask * ((1 - probs) ** self.alpha) * probs
        neg_loss = 1 - (2 * neg_weight + self.smooth) / (neg_weight + self.smooth)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


class MultiDSCLoss(nn.Module):
    """
    Creates a criterion that optimizes a multi-class self-adjusting dice loss
    """
    def __init__(self, alpha: float=1.0, smooth: float=1.0, reduction: str='mean'):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none' or self.reduction is None:
            return loss
        else:
            raise NotImplementedError("reduction error")

class GHM_loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        """
        :param bins: split to n bins
        :param alpha: hyper-parameter
        """
        super(GHM_loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])

class GHMC_Loss(GHM_loss):
    """
    GHM_loss for classification
    softmax backward 第一层 梯度为pi -yi，模长尾|pi - yi|
    核心idea: 我们确实不应该过多关注易分样本，但是特别难分的样本(outliers, 离群点)也不应该关注哈，这些离群点的梯度模长d要比一般
    的样本大很多，如果模型被迫去关注这些样本，而有可能降低模型的准确度！况且，这样的样本数量也很多！
    实现方式：核心在于梯度密度计算，把梯度模长范围划分成10个区域，这里要求必须经过sigmoid计算，这样梯度模长的范围就限制
    在0-1之间
    梯度密度GD（g)的物理含义是：单位梯度模长g部分的样本个数
    * 注意，不管FocalLoss还是GHM其实都是对不同样本赋予不同的权重，所以该代码前面的计算的都是样本权重，最后计算GHM Loss
    就是调用了Pytorch自带的binary_cross_entropy_with_logits，将样本权重填进去
    """

    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_loss):
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu

        N = x.size(0) * x.size(1)
        return (loss.weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)