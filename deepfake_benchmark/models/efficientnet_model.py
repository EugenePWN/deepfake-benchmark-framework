"""
EfficientNet-B4 для детекции дипфейков.

Используем pretrained EfficientNet-B4 из torchvision с заменой головы классификатора.
Стратегия обучения: сначала обучаем только голову (freeze backbone),
затем размораживаем и fine-tune весь backbone с меньшим lr.

Почему EfficientNet-B4:
  - Лучший ImageNet pretrained среди B0-B7 по соотношению точность/скорость
  - Compound scaling (глубина + ширина + разрешение) → лучше улавливает
    мелкие артефакты face-swap на разных масштабах
  - 19M параметров vs 22M у Xception → быстрее обучается
  - Входное разрешение 380×380 → больше деталей чем у Xception (299×299)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNetB4Detector(nn.Module):
    """
    EfficientNet-B4 детектор дипфейков.

    Архитектура:
      [EfficientNet-B4 backbone (pretrained ImageNet)]
        → AdaptiveAvgPool → Flatten
        → Dropout(dropout_rate)
        → Linear(1792 → 512) → SiLU → Dropout(dropout_rate * 0.5)
        → Linear(512 → num_classes)

    Поддерживает двухфазное обучение:
      Фаза 1: backbone заморожен, обучается только голова
      Фаза 2: разморожен весь backbone, fine-tune с меньшим lr

    Args:
        num_classes:   число классов (2 для real/fake)
        dropout_rate:  dropout перед головой классификатора
        pretrained:    использовать ImageNet weights (рекомендуется True)
        freeze_backbone: заморозить backbone при инициализации (для Фазы 1)
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        # Убираем оригинальный classifier — оставляем только features
        self.features = backbone.features
        self.avgpool  = backbone.avgpool   # AdaptiveAvgPool2d((1,1))

        # EfficientNet-B4 выдаёт 1792 features после avgpool
        in_features = 1792

        # Двухслойная голова с SiLU (нативная активация EfficientNet)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

        self._init_classifier()

        if freeze_backbone:
            self.freeze_backbone()

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Freeze / Unfreeze
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Заморозить backbone — обучается только голова (Фаза 1)."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None) -> None:
        """
        Разморозить backbone (Фаза 2).

        Args:
            unfreeze_last_n_blocks: если None — размораживаем весь backbone.
                Если int — размораживаем только последние N блоков.
                Полезно при малом датасете (< 5000 изображений).
        """
        if unfreeze_last_n_blocks is None:
            for p in self.features.parameters():
                p.requires_grad = True
        else:
            # features — это Sequential из блоков EfficientNet
            blocks = list(self.features.children())
            # Сначала всё замораживаем
            for p in self.features.parameters():
                p.requires_grad = False
            # Размораживаем последние N блоков
            for block in blocks[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        """
        Возвращает param_groups для AdamW с разными lr для backbone и головы.
        Используется в Фазе 2 (fine-tune).

        Пример:
            optimizer = AdamW(model.get_param_groups(lr_backbone=1e-5, lr_head=1e-4))
        """
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        head_params     = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает 1792-мерный вектор признаков (до классификатора)."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_efficientnet_b4(
    num_classes: int = 2,
    dropout_rate: float = 0.4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    weights_path: Optional[str] = None,
) -> EfficientNetB4Detector:
    """
    Фабричная функция.

    Args:
        num_classes:     2 для real/fake
        dropout_rate:    dropout перед FC (0.4 рекомендуется для B4)
        pretrained:      ImageNet weights (всегда True если нет причин)
        freeze_backbone: True для старта с Фазы 1 (обучение только головы)
        weights_path:    путь к сохранённому state_dict
    """
    model = EfficientNetB4Detector(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"[EfficientNet-B4] Loaded weights from {weights_path}")

    total     = model.total_params()
    trainable = model.trainable_params()
    print(f"[EfficientNet-B4] Total params: {total:,} | Trainable: {trainable:,}")
    return model


if __name__ == "__main__":
    # Smoke test
    model = build_efficientnet_b4(pretrained=False)
    x = torch.randn(2, 3, 380, 380)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # (2, 2)
    feats = model.extract_features(x)
    print(f"Features: {feats.shape}")  # (2, 1792)