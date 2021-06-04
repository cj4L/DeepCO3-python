from torch import nn
import torch.nn.functional as F
import torch


# class Weighed_Bce_Loss(nn.Module):
#     def __init__(self):
#         super(Weighed_Bce_Loss, self).__init__()
#
#     def forward(self, x, label):
#         x = x.view(-1, 1, x.shape[1], x.shape[2])
#         label = label.view(-1, 1, label.shape[1], label.shape[2])
#         label_t = (label == 1).float()
#         label_f = (label == 0).float()
#         p = torch.sum(label_t) / (torch.sum(label_t) + torch.sum(label_f))
#         w = torch.zeros_like(label)
#         w[label == 1] = p
#         w[label == 0] = 1 - p
#         loss = F.binary_cross_entropy(x, label, weight=w)
#         return loss


# class Cls_Loss(nn.Module):
#     def __init__(self):
#         super(Cls_Loss, self).__init__()
#
#     def forward(self, x, label):
#         loss = F.binary_cross_entropy(x, label)
#         return loss
#
# class S_Loss(nn.Module):
#     def __init__(self):
#         super(S_Loss, self).__init__()
#
#     def forward(self, x, label):
#         loss = F.smooth_l1_loss(x, label)
#         return loss


class saliency_loss(nn.Module):
    def __init__(self):
        super(saliency_loss, self).__init__()

    def forward(self, prediction, label):
        threshold = torch.mean(label, dim=[2,3], keepdim=True)
        binary_label = label > threshold
        ratio = torch.mean(binary_label.float(), dim=[2,3], keepdim=True)

        postive = torch.ones_like(label) * ratio
        negative = torch.ones_like(label) * (1 - ratio)

        w = torch.zeros_like(label)
        w[binary_label == 1] = negative[binary_label == 1]
        w[binary_label == 0] = postive[binary_label == 0]

        loss = torch.sum(w * (prediction - label)**2) / prediction.size()[2] / prediction.size()[3]

        return loss

class affinity_loss(nn.Module):
    def __init__(self):
        super(affinity_loss, self).__init__()

    def forward(self, pixel_affinity, sal_affinity, sal_diff):
        loss = torch.mean(pixel_affinity * (1 - sal_affinity)) + 4 * torch.mean(sal_diff * sal_affinity)
        return loss

class co_peak_loss(nn.Module):
    def __init__(self):
        super(co_peak_loss, self).__init__()

    def forward(self, co_peak_value):
        a = -1 * co_peak_value
        b = torch.max(torch.zeros_like(co_peak_value), a)
        t = b + torch.log(torch.exp(-b) + torch.exp(a - b))
        loss = torch.mean(t)
        return loss





class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_saliency = saliency_loss()
        self.loss_affinity = affinity_loss()
        self.loss_co_peak = co_peak_loss()

    def forward(self, x, label, pixel_affinity, sal_affinity, sal_diff, co_peak_value):
        loss_sa = self.loss_saliency(x, label)
        loss_af = self.loss_affinity(pixel_affinity, sal_affinity, sal_diff) * 0.1
        loss_co = self.loss_co_peak(co_peak_value) * 0.1


        return loss_sa, loss_af, loss_co

if __name__ == '__main__':
    s = Loss()
    p = torch.randn(6,1,14,14)
    label = torch.randn(6, 1, 14, 14)
    loss = s(p, label)

    print('a')