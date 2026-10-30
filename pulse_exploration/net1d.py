"""Net1D, the 1-D ResNet with squeeze-and-excitation that AI-PPG Age runs on (pulse_exploration/ppg_age.py).

Re-implemented from ``net1d.py`` in the Hugging Face repository Ngks03/PPG-VascularAge (MIT; Net1D by
Shenda Hong, 2020) so that no downloaded code is executed: only the weights are downloaded, and they
are loaded with ``torch.load(weights_only=True)``. Module and parameter names are kept identical, so
the published ``state_dict`` loads with ``strict=True``. Dropped from the original: the unused
``transformers`` import and the verbose printing. In eval mode dropout is off and batch norm uses its
running statistics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MyConv1dPadSame(nn.Module):
    """Conv1d with TensorFlow-style SAME padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        out = (n + self.stride - 1) // self.stride
        p = max(0, (out - 1) * self.stride + self.kernel_size - n)
        return self.conv(F.pad(x, (p // 2, p - p // 2), "constant", 0))


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = max(0, self.kernel_size - 1)
        return self.max_pool(F.pad(x, (p // 2, p - p // 2), "constant", 0))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    """conv1 -> convk -> conv1, squeeze-and-excitation, residual (zero-padded channels)."""

    def __init__(self, in_channels: int, out_channels: int, ratio: float, kernel_size: int, stride: int, groups: int,
                 downsample: bool, is_first_block: bool = False, use_bn: bool = True, use_do: bool = True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block, self.use_bn, self.use_do = is_first_block, use_bn, use_do
        mid = int(out_channels * ratio)
        self.bn1, self.activation1, self.do1 = nn.BatchNorm1d(in_channels), Swish(), nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, mid, 1, 1)
        self.bn2, self.activation2, self.do2 = nn.BatchNorm1d(mid), Swish(), nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(mid, mid, kernel_size, self.stride, groups=groups)
        self.bn3, self.activation3, self.do3 = nn.BatchNorm1d(mid), Swish(), nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(mid, out_channels, 1, 1)
        r = 2
        self.se_fc1 = nn.Linear(out_channels, out_channels // r)
        self.se_fc2 = nn.Linear(out_channels // r, out_channels)
        self.se_activation = Swish()
        if downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity, out = x, x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        for bn, act, do, conv in ((self.bn2, self.activation2, self.do2, self.conv2),
                                  (self.bn3, self.activation3, self.do3, self.conv3)):
            if self.use_bn:
                out = bn(out)
            out = act(out)
            if self.use_do:
                out = do(out)
            out = conv(out)
        se = torch.sigmoid(self.se_fc2(self.se_activation(self.se_fc1(out.mean(-1)))))
        out = torch.einsum("abc,ab->abc", out, se)
        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity.transpose(-1, -2), (ch1, ch2), "constant", 0).transpose(-1, -2)
        return out + identity


class BasicStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: float, kernel_size: int, stride: int, groups: int,
                 i_stage: int, m_blocks: int, use_bn: bool = True, use_do: bool = True):
        super().__init__()
        self.block_list = nn.ModuleList(
            BasicBlock(in_channels if i == 0 else out_channels, out_channels, ratio, kernel_size,
                       stride if i == 0 else 1, groups, downsample=i == 0, is_first_block=i_stage == 0 and i == 0,
                       use_bn=use_bn, use_do=use_do)
            for i in range(m_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        return x


class Net1D(nn.Module):
    """(n, in_channels, length) -> (n, n_classes); global average pooling before the dense layer."""

    def __init__(self, in_channels: int, base_filters: int, ratio: float, filter_list: list[int],
                 m_blocks_list: list[int], kernel_size: int, stride: int, groups_width: int, n_classes: int = 1,
                 use_bn: bool = True, use_do: bool = True, verbose: bool = False):
        super().__init__()
        self.use_bn = use_bn
        self.first_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size, stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()
        stages, ch = [], base_filters
        for i, (out, m) in enumerate(zip(filter_list, m_blocks_list)):
            stages.append(BasicStage(ch, out, ratio, kernel_size, stride, out // groups_width, i, m, use_bn, use_do))
            ch = out
        self.stage_list = nn.ModuleList(stages)
        self.dense = nn.Linear(ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
        for stage in self.stage_list:
            out = stage(out)
        return self.dense(out.mean(-1))
