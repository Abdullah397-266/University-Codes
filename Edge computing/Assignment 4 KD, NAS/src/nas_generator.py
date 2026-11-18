"""
Q2 - automated architecture generator: turns an ArchConfig into a runnable PyTorch network.

    net = build_network(config, num_classes=10)

Layout:  stem 3x3 conv  ->  stage 1 ... stage N  ->  1x1 head conv  ->  global avg pool  ->  dropout  ->  linear
Stage i (i >= 2) starts with a stride-2 block. Residual connections are added automatically whenever a block
keeps both the shape and the stride (stride 1 and in_channels == out_channels).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nas_search_space import ArchConfig


def make_act(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "relu6": nn.ReLU6, "hswish": nn.Hardswish, "silu": nn.SiLU,
            "leaky_relu": lambda inplace=True: nn.LeakyReLU(0.1, inplace=inplace)}[name](inplace=True)


def conv_bn(in_ch, out_ch, k, stride=1, groups=1, act: str | None = "relu") -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(make_act(act))
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Standard KxK convolution block."""

    def __init__(self, in_ch, out_ch, k, stride, act, **_):
        super().__init__()
        self.body = conv_bn(in_ch, out_ch, k, stride, act=act)
        self.res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class DWSepBlock(nn.Module):
    """Depthwise separable convolution: KxK depthwise (groups = channels) followed by 1x1 pointwise."""

    def __init__(self, in_ch, out_ch, k, stride, act, **_):
        super().__init__()
        self.body = nn.Sequential(conv_bn(in_ch, in_ch, k, stride, groups=in_ch, act=act),
                                  conv_bn(in_ch, out_ch, 1, act=act))
        self.res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class MBConvBlock(nn.Module):
    """Inverted residual (MobileNetV2-style): 1x1 expand -> KxK depthwise -> 1x1 linear projection."""

    def __init__(self, in_ch, out_ch, k, stride, act, expand=4):
        super().__init__()
        mid = in_ch * expand
        self.body = nn.Sequential(conv_bn(in_ch, mid, 1, act=act), conv_bn(mid, mid, k, stride, groups=mid, act=act),
                                  conv_bn(mid, out_ch, 1, act=None))
        self.res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


BLOCK_TYPES = {"conv": ConvBlock, "dwsep": DWSepBlock, "mbconv": MBConvBlock}


class NASNet(nn.Module):
    def __init__(self, config: ArchConfig, num_classes: int = 10):
        super().__init__()
        self.config = config
        first_act = config.stages[0].activation
        layers = [conv_bn(3, config.stem_channels, 3, 1, act=first_act)]
        in_ch = config.stem_channels
        for i, st in enumerate(config.stages):
            for j in range(st.depth):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BLOCK_TYPES[st.block](in_ch, st.channels, st.kernel, stride, st.activation,
                                                    expand=st.expand))
                in_ch = st.channels
        layers.append(conv_bn(in_ch, config.head_channels, 1, act=config.stages[-1].activation))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(config.dropout),
                                        nn.Linear(config.head_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_network(config: ArchConfig | dict, num_classes: int = 10) -> NASNet:
    if isinstance(config, dict):
        config = ArchConfig.from_dict(config)
    return NASNet(config, num_classes)


if __name__ == "__main__":
    import random

    from common import count_macs, count_params
    from nas_search_space import sample_config
    rng = random.Random(1)
    for _ in range(5):
        cfg = sample_config(rng)
        net = build_network(cfg).eval()
        out = net(torch.randn(2, 3, cfg.resolution, cfg.resolution))
        print(f"{tuple(out.shape)} params {count_params(net) / 1e3:7.1f} K  MACs "
              f"{count_macs(net, (1, 3, cfg.resolution, cfg.resolution)) / 1e6:6.1f} M | {cfg.summary()}")
