"""Exact definitions extracted from Train3_YOLO26s_SCConv_768.ipynb.
Do not substitute another SCConv implementation. See tests/test_model.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Bottleneck  # noqa: F401 - keep notebook import context
SCONV_CONFIG = dict(group_num=4, gate_threshold=.5, alpha=.5, squeeze_ratio=2, group_size=2, group_kernel_size=3)

def nearest_valid_group_num(c_num, requested):
    valid = [group for group in range(1, c_num + 1) if c_num % group == 0]
    return min(valid, key=lambda group: (abs(group - requested), -group))

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num=4, eps=1e-5):
        super().__init__()
        self.group_num = nearest_valid_group_num(c_num, group_num)
        if self.group_num != group_num:
            print(f'GroupBatchnorm2d adjusted group_num {group_num} -> {self.group_num} for C={c_num}')
        self.weight = nn.Parameter(torch.ones(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch, self.group_num, -1)
        mean = grouped.mean(2, keepdim=True)
        variance = grouped.var(2, keepdim=True, unbiased=False)
        normalized = ((grouped - mean) / torch.sqrt(variance + self.eps)).reshape(batch, channels, height, width)
        return normalized * self.weight + self.bias

class SRU(nn.Module):
    def __init__(self, oup_channels, group_num=4, gate_threshold=0.5):
        super().__init__()
        if oup_channels % 2:
            raise ValueError('SRU cross reconstruction requires even channels.')
        self.gn = GroupBatchnorm2d(oup_channels, group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        gamma = self.gn.weight
        gamma_weight = (gamma / (gamma.sum() + torch.finfo(gamma.dtype).eps)).view(1, -1, 1, 1)
        weights = self.sigmoid(gn_x * gamma_weight)
        informative = torch.where(weights > self.gate_threshold, torch.ones_like(weights), weights)
        non_informative = torch.where(weights > self.gate_threshold, torch.zeros_like(weights), weights)
        return self.reconstruct(informative * x, non_informative * x)

    def reconstruct(self, x1, x2):
        half = x1.shape[1] // 2
        x11, x12 = torch.split(x1, [half, half], 1)
        x21, x22 = torch.split(x2, [half, half], 1)
        return torch.cat([x11 + x22, x12 + x21], 1)

class CRU(nn.Module):
    def __init__(self, op_channel, alpha=0.5, squeeze_ratio=2, group_size=2, group_kernel_size=3):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel
        self.up_squeeze = self.up_channel // squeeze_ratio
        self.low_squeeze = self.low_channel // squeeze_ratio
        if min(self.up_squeeze, self.low_squeeze) < 1 or self.up_squeeze % group_size or op_channel % group_size:
            raise ValueError('CRU channel configuration is incompatible.')
        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_squeeze, 1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_squeeze, 1, bias=False)
        self.GWC = nn.Conv2d(self.up_squeeze, op_channel, group_kernel_size, 1, group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(self.up_squeeze, op_channel, 1, bias=False)
        self.PWC2 = nn.Conv2d(self.low_squeeze, op_channel - self.low_squeeze, 1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        upper, lower = torch.split(x, [self.up_channel, self.low_channel], 1)
        upper, lower = self.squeeze1(upper), self.squeeze2(lower)
        upper_branch = self.GWC(upper) + self.PWC1(upper)
        lower_branch = torch.cat([self.PWC2(lower), lower], 1)
        fused = torch.cat([upper_branch, lower_branch], 1)
        fused = F.softmax(self.advavg(fused), dim=1) * fused
        first, second = torch.split(fused, fused.shape[1] // 2, 1)
        return first + second

class ScConv(nn.Module):
    def __init__(self, op_channel, group_num=4, gate_threshold=0.5, alpha=0.5, squeeze_ratio=2, group_size=2, group_kernel_size=3):
        super().__init__()
        self.SRU = SRU(op_channel, group_num, gate_threshold)
        self.CRU = CRU(op_channel, alpha, squeeze_ratio, group_size, group_kernel_size)

    def forward(self, x):
        return self.CRU(self.SRU(x))

def wrapped_channels(wrapper):
    raw = getattr(wrapper, 'conv', wrapper)
    return int(raw.in_channels), int(raw.out_channels)

def wrapped_stride(wrapper):
    stride = getattr(getattr(wrapper, 'conv', wrapper), 'stride', 1)
    return (stride, stride) if isinstance(stride, int) else tuple(stride)

class Bottleneck_SCConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.cv1 = original.cv1
        self.add = bool(getattr(original, 'add', False))
        input_channels, output_channels = wrapped_channels(original.cv2)
        self.scconv_applied = (
            wrapped_stride(original.cv2) == (1, 1)
            and input_channels == output_channels
            and input_channels % 2 == 0
        )
        self.cv2 = ScConv(input_channels, **SCONV_CONFIG) if self.scconv_applied else original.cv2

    def forward(self, x):
        output = self.cv2(self.cv1(x))
        return x + output if self.add else output

def replace_bottlenecks(module, stats):
    if module.__class__.__name__ == 'Bottleneck':
        replacement = Bottleneck_SCConv(module)
        stats['seen'] += 1
        stats['scconv'] += int(replacement.scconv_applied)
        stats['kept'] += int(not replacement.scconv_applied)
        return replacement
    for name, child in list(module.named_children()):
        replacement = replace_bottlenecks(child, stats)
        if replacement is not child:
            setattr(module, name, replacement)
    return module

class C3k2_SCConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.c, self.cv1, self.cv2 = original.c, original.cv1, original.cv2
        self.stats = {'seen': 0, 'scconv': 0, 'kept': 0}
        self.m = replace_bottlenecks(original.m, self.stats)

    def forward(self, x):
        values = list(self.cv1(x).chunk(2, 1))
        values.extend(block(values[-1]) for block in self.m)
        return self.cv2(torch.cat(values, 1))

    def forward_split(self, x):
        values = list(self.cv1(x).split((self.c, self.c), 1))
        values.extend(block(values[-1]) for block in self.m)
        return self.cv2(torch.cat(values, 1))
