import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.losses.utils import weighted_loss, get_class_weight

from mmseg.models.losses import OhemCrossEntropy
from mmseg.registry import MODELS


@MODELS.register_module()
class OHEMCrossEntropyLoss(OhemCrossEntropy):

    def forward(self, score, target, **kwargs):
        return super().forward(score=score, target=target)


@MODELS.register_module()
class MMDLoss(nn.Module):
    '''Maximum Mean Discrepancies(MMD) loss.'''

    def __init__(self,
                 kernel_mul=2.0,
                 kernel_num=5,
                 loss_weight=1.0,
                 loss_name='loss_mmd'):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul

        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)

        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul**(kernel_num // 2)
        bandwidth_list = [
            bandwidth * (kernel_mul**i) for i in range(kernel_num)
        ]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source,
                                       target,
                                       kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX) * self.loss_weight
        return loss


@MODELS.register_module()
class MyDiceLoss(nn.Module):

    def __init__(self,
                 eps=1e-3,
                 reduction='mean',
                 naive_dice=False,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_name_ = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            one_hot_target = target.unsqueeze(-1).long()
        else:
            pred = F.softmax(pred, dim=1)
            num_classes = pred.shape[1]
            one_hot_target = F.one_hot(torch.clamp(target.long(), 0,
                                                   num_classes - 1),
                                       num_classes=num_classes)

        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(pred,
                                            one_hot_target,
                                            valid_mask=valid_mask,
                                            eps=self.eps,
                                            naive_dice=self.naive_dice,
                                            class_weight=class_weight,
                                            ignore_index=self.ignore_index,
                                            reduction=reduction,
                                            avg_factor=avg_factor)
        return loss

    @property
    def loss_name(self):
        return self.loss_name_


@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              eps=1e-3,
              naive_dice=False,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(pred[:, i],
                                         target[..., i],
                                         valid_mask=valid_mask,
                                         eps=eps,
                                         naive_dice=naive_dice)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred,
                     target,
                     valid_mask,
                     eps,
                     naive_dice=False,
                     batch=True,
                     **kwargs):
    assert pred.shape[0] == target.shape[0]
    if not batch:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    target = target * valid_mask
    num = torch.sum(pred * target)

    if naive_dice:
        d = (2 * num + eps) / (torch.sum(pred) + torch.sum(target) + eps)
    else:
        den = torch.sum(pred * pred) + torch.sum(target * target)
        d = (2 * num) / (den + eps)

    return 1 - d
