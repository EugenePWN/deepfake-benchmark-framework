"""
Xception Architecture for DeepFake Detection
Paper: "Xception: Deep Learning with Depthwise Separable Convolutions" (Chollet, 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution: depthwise + pointwise."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Block(nn.Module):
    """Xception Middle/Exit flow block with residual connection."""

    def __init__(self, in_channels, out_channels, reps,
                 stride=1, start_with_relu=True, grow_first=True):
        super().__init__()

        # Residual shortcut (1x1 conv if channels or stride differ)
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None

        layers = []
        filters = in_channels

        if grow_first:
            # First sep conv expands channels
            layers += [
                nn.ReLU(inplace=False) if start_with_relu else nn.Identity(),
                SeparableConv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ]
            filters = out_channels

        for _ in range(reps - 1):
            layers += [
                nn.ReLU(inplace=False),
                SeparableConv2d(filters, filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(filters)
            ]

        if not grow_first:
            layers += [
                nn.ReLU(inplace=False),
                SeparableConv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ]

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride=stride, padding=1))

        # Remove leading Identity if present
        self.rep = nn.Sequential(*[l for l in layers if not isinstance(l, nn.Identity)])

    def forward(self, x):
        residual = self.skip(x) if self.skip is not None else x
        return self.rep(x) + residual


class Xception(nn.Module):
    """
    Full Xception network adapted for binary deepfake classification.

    Args:
        num_classes: number of output classes (default 2 for real/fake)
        dropout_rate: dropout before classifier head
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()

        # ── Entry Flow ────────────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        self.block1 = Block(64,  128, reps=2, stride=2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True,  grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=2, start_with_relu=True,  grow_first=True)

        # ── Middle Flow (8 identical blocks) ─────────────────────────────────
        self.block4  = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)

        # ── Exit Flow ─────────────────────────────────────────────────────────
        self.block12 = Block(728, 1024, reps=2, stride=2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(2048)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout         = nn.Dropout(p=dropout_rate)
        self.fc              = nn.Linear(2048, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 2048-d feature vector (before classifier)."""
        # Entry Flow
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle Flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # Exit Flow
        x = self.block12(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        return self.fc(x)


def build_xception(num_classes: int = 2, dropout_rate: float = 0.5,
                   pretrained_path: str | None = None) -> Xception:
    """
    Factory function.

    Args:
        num_classes:     2 for real/fake binary classification
        dropout_rate:    dropout before the FC head (0.5 recommended for small datasets)
        pretrained_path: optional path to a saved state_dict
    """
    model = Xception(num_classes=num_classes, dropout_rate=dropout_rate)

    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"[Xception] Loaded weights from {pretrained_path}")

    return model


if __name__ == "__main__":
    model = build_xception(num_classes=2)
    x = torch.randn(2, 3, 299, 299)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # (2, 2)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")