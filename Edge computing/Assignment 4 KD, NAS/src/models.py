"""
Teacher and student architectures for CIFAR-100 (32x32 inputs).

Every model implements
    forward(x)                -> logits
    forward_with_features(x)  -> (logits, feature_map)
where feature_map is the last convolutional feature map before global pooling. These maps are what the
feature-based distillation in kd.py aligns. `feature_channels` gives the channel count of that map.

CIFAR adaptation: ImageNet networks downsample 32x by the last stage, which would shrink a 32x32 image to 1x1.
Each network below removes the first two stride-2 operations so the final feature map is 4x4 (overall
stride 8) for all four models, which keeps teacher and student feature maps spatially aligned.

* Teacher  : ResNet-50 (torchvision bottleneck implementation, 3x3 stride-1 stem, no max-pool).
* Student 1: MobileNetV2 (torchvision implementation, stride of stem and first stride-2 block set to 1).
* Student 2: SqueezeNet 1.1 - implemented here, with BatchNorm in every Fire module so it trains reliably
             from scratch, and without the final ReLU on the logits (a ReLU there makes all logits >= 0,
             which distorts the temperature-softened distribution used in distillation).
* Student 3: EfficientNet-Lite0 - implemented here following the Lite recipe (no squeeze-and-excitation,
             ReLU6 instead of Swish, fixed 32-ch stem and 1280-ch head, B0 stage layout).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


# =========================================================================== teacher
class ResNet50CIFAR(nn.Module):
    feature_channels = 2048

    def __init__(self, num_classes: int = 100):
        super().__init__()
        net = torchvision.models.resnet50(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def features(self, x):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.layer2(n.layer1(n.maxpool(x)))
        return n.layer4(n.layer3(x))                      # (B, 2048, 4, 4)

    def forward_with_features(self, x):
        f = self.features(x)
        logits = self.net.fc(torch.flatten(self.net.avgpool(f), 1))
        return logits, f

    def forward(self, x):
        return self.forward_with_features(x)[0]


# =========================================================================== student 1: MobileNetV2
class MobileNetV2CIFAR(nn.Module):
    feature_channels = 1280

    def __init__(self, num_classes: int = 100, width_mult: float = 1.0):
        super().__init__()
        net = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes, width_mult=width_mult)
        net.features[0][0].stride = (1, 1)               # stem: stride 2 -> 1
        net.features[2].conv[1][0].stride = (1, 1)       # first stride-2 inverted residual -> stride 1
        self.net = net
        self.feature_channels = net.last_channel

    def forward_with_features(self, x):
        f = self.net.features(x)                         # (B, 1280, 4, 4)
        logits = self.net.classifier(torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1))
        return logits, f

    def forward(self, x):
        return self.forward_with_features(x)[0]


# =========================================================================== student 2: SqueezeNet 1.1
class Fire(nn.Module):
    """Fire module: 1x1 squeeze, then parallel 1x1 and 3x3 expand branches concatenated."""

    def __init__(self, in_ch: int, squeeze: int, expand1: int, expand3: int):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_ch, squeeze, 1, bias=False), nn.BatchNorm2d(squeeze),
                                     nn.ReLU(inplace=True))
        self.expand1 = nn.Sequential(nn.Conv2d(squeeze, expand1, 1, bias=False), nn.BatchNorm2d(expand1))
        self.expand3 = nn.Sequential(nn.Conv2d(squeeze, expand3, 3, padding=1, bias=False), nn.BatchNorm2d(expand3))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        return self.act(torch.cat([self.expand1(x), self.expand3(x)], 1))


class SqueezeNetCIFAR(nn.Module):
    feature_channels = 512

    def __init__(self, num_classes: int = 100, dropout: float = 0.3):
        super().__init__()
        pool = lambda: nn.MaxPool2d(3, 2, ceil_mode=True)   # noqa: E731
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 32x32
            Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), pool(),                                 # 16x16
            Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), pool(),                            # 8x8
            Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256), pool(),                                                     # 4x4
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
                                        nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward_with_features(self, x):
        f = self.features(x)                             # (B, 512, 4, 4)
        return self.classifier(f), f

    def forward(self, x):
        return self.forward_with_features(x)[0]


# =========================================================================== student 3: EfficientNet-Lite0
class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, groups=1, act=True):
        super().__init__(nn.Conv2d(in_ch, out_ch, k, s, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True) if act else nn.Identity())


class MBConv(nn.Module):
    """Mobile inverted bottleneck without squeeze-and-excitation (the 'Lite' variant)."""

    def __init__(self, in_ch, out_ch, expand, k, stride, drop_path=0.0):
        super().__init__()
        mid = in_ch * expand
        layers = [] if expand == 1 else [ConvBNAct(in_ch, mid, 1)]
        layers += [ConvBNAct(mid, mid, k, stride, groups=mid), ConvBNAct(mid, out_ch, 1, act=False)]
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and in_ch == out_ch
        self.drop_path = drop_path

    def forward(self, x):
        out = self.block(x)
        if not self.use_res:
            return out
        if self.training and self.drop_path > 0:          # stochastic depth
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            out = out * keep / (1 - self.drop_path)
        return x + out


class EfficientNetLite0CIFAR(nn.Module):
    feature_channels = 1280
    # (expand ratio, kernel, stride, out channels, repeats) - EfficientNet-B0 layout.
    # CIFAR change: the first stride-2 stage (24 ch) uses stride 1 (stem also stride 1).
    STAGES = [(1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
              (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]

    def __init__(self, num_classes: int = 100, dropout: float = 0.2, drop_path: float = 0.1):
        super().__init__()
        layers = [ConvBNAct(3, 32, 3, 1)]               # Lite: stem fixed at 32 channels
        in_ch, total = 32, sum(s[4] for s in self.STAGES)
        idx = 0
        for e, k, s, c, n in self.STAGES:
            for i in range(n):
                layers.append(MBConv(in_ch, c, e, k, s if i == 0 else 1, drop_path * idx / total))
                in_ch, idx = c, idx + 1
        layers.append(ConvBNAct(in_ch, 1280, 1))         # Lite: head fixed at 1280 channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward_with_features(self, x):
        f = self.features(x)                             # (B, 1280, 4, 4)
        return self.classifier(torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1)), f

    def forward(self, x):
        return self.forward_with_features(x)[0]


# =========================================================================== registry
REGISTRY = {
    "resnet50": ResNet50CIFAR,
    "mobilenetv2": MobileNetV2CIFAR,
    "squeezenet": SqueezeNetCIFAR,
    "efficientnet_lite0": EfficientNetLite0CIFAR,
}


def build_model(arch: str, num_classes: int = 100) -> nn.Module:
    if arch not in REGISTRY:
        raise ValueError(f"unknown architecture '{arch}', choose from {list(REGISTRY)}")
    return REGISTRY[arch](num_classes=num_classes)


if __name__ == "__main__":
    # Quick self-check: shapes, parameter counts and feature map sizes of all four networks.
    from common import count_macs, count_params
    x = torch.randn(2, 3, 32, 32)
    for name in REGISTRY:
        m = build_model(name).eval()
        logits, feat = m.forward_with_features(x)
        print(f"{name:20s} logits {tuple(logits.shape)} features {tuple(feat.shape)} "
              f"params {count_params(m) / 1e6:6.2f} M  MACs {count_macs(m) / 1e6:8.1f} M")
