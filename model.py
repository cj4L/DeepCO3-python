from types import MethodType
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import scipy.ndimage.filters as filters
import numpy as np
from peak_backprop import pr_conv2d

# vgg choice
base = {'vgg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

# vgg16
def vgg16(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class Model(nn.Module):
    def __init__(self, base):
        super(Model, self).__init__()
        self.base = base
        self.conv6 = nn.Sequential(nn.Conv2d(512, 4096, 7, 1, 3), nn.ReLU())
        self.conv1_1 = nn.Sequential(nn.Conv2d(4096, 1, 1))

    def lpnorm(self, x):
        return x / (torch.sum(x, dim=1, keepdim=True) + 0.01) ** 0.5

    def correlate_cross(self, x):
        n, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        return torch.mm(x, x.transpose(1, 0))

    def sal_diff_cross(self, x):
        n, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        return (x - x.transpose(1, 0)) ** 2

    def correlate(self, x):
        n, c, h, w = x.size()
        pair = n // 2
        x1 = x[:pair]
        x2 = x[pair:]
        x1 = x1.reshape(pair, c, -1)
        x2 = x2.reshape(pair, c, -1)
        return torch.bmm(x1.permute(0, 2, 1), x2).reshape(pair, h, w, h, w)

    def co_peak_gen(self, x):
        pair, h, w, h, w = x.size()
        res = []
        for i in range(pair):
            tmpx = x[i]
            tmpx_np = tmpx.detach().cpu().numpy()
            co_peak_np = filters.maximum_filter(tmpx_np, size=(3, 3, 3, 3))
            co_peak = (tmpx_np == co_peak_np) & (tmpx_np >= np.median(tmpx_np))
            positivate_data = torch.mean(tmpx[np.where(co_peak)])
            res.append(positivate_data)
        return torch.stack(res)

    def plane_peak_gen(self, x):
        x = torch.squeeze(x)
        x_np = x.detach().cpu().numpy()
        plane_peak_np = filters.maximum_filter(x_np, size=(3, 3))
        plane_peak_np = (x_np == plane_peak_np) & (x_np >= np.median(x_np))
        peak_list = torch.nonzero(torch.tensor(plane_peak_np))
        return torch.tensor(plane_peak_np), peak_list

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward




    def forward(self, x, visual=False):
        x = self.base(x)
        feat = self.conv6(x)
        feat_norm = self.lpnorm(feat)
        x = self.conv1_1(feat)
        small_salmap = nn.Sigmoid()(x)
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        salmap = nn.Sigmoid()(x)


        if visual:
            plane_peak, peak_list = self.plane_peak_gen(small_salmap)
            return salmap, plane_peak, peak_list, small_salmap

        pixel_affinity = self.correlate_cross(feat_norm)
        sal_affinity = self.correlate_cross(small_salmap)
        sal_diff = self.sal_diff_cross(small_salmap)

        weight_feat_norm = feat_norm * small_salmap
        fourD_tensor = self.correlate(weight_feat_norm)
        co_peak_value = self.co_peak_gen(fourD_tensor)


        return salmap, pixel_affinity, sal_affinity, sal_diff, co_peak_value


# build the whole network
def build_model():
    return Model(vgg16(base['vgg']))

# weight init
def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torch.backends import cudnn
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    cudnn.benchmark = True
    vgg_path = '/home/chenjin/weights/vgg16_feat.pth'
    device = torch.device('cuda:0')

    net = build_model()
    # tmp = cut_weights(vgg_path)

    net.base.load_state_dict(torch.load(vgg_path))
    net = net.to(device)
    img = torch.randn(1, 3, 448, 448)
    # net = net.to(device)
    img = img.to(device)
    mask = net(img, True)
    print('a')