"""
F3Net — Frequency-aware Forgery Feature Network для детекции дипфейков.

Оригинальная статья:
  "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues"
  Qingshan Liu et al., ECCV 2020.

Ключевая идея:
  CNN детекторы (Xception, EfficientNet) работают в пространственной области
  и пропускают артефакты, которые видны только в частотной области:
    - Периодические артефакты от upsampling/conv в генераторе
    - Артефакты JPEG-сжатия на границе маски
    - Различия в high-frequency текстурах кожи (real vs generated)

  F3Net явно извлекает и объединяет:
    1. FAD (Frequency-Aware Decomposition)  — разложение на частотные компоненты
    2. LFS (Local Frequency Statistics)     — локальные частотные статистики через DCT
    3. Fusion                               — объединение пространственных и частотных признаков

Адаптация для этой реализации:
  Используем Xception как backbone (как в оригинальной статье),
  добавляем FAD-head и LFS-branch, объединяем через Cross-Attention Fusion.
  Это более практичная имплементация без зависимостей кроме torch.

Входное разрешение: 299×299 (совместимо с Xception backbone)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# DCT / Частотные утилиты
# ──────────────────────────────────────────────────────────────────────────────

def _dct_filters(n: int, device: torch.device) -> torch.Tensor:
    """
    Создаёт 2D DCT-II фильтры размером n×n.
    Возвращает тензор формы (n*n, 1, n, n) — для nn.Conv2d как фиксированные веса.
    """
    # 1D DCT базис
    k = torch.arange(n, dtype=torch.float32, device=device)
    i = torch.arange(n, dtype=torch.float32, device=device)
    # DCT-II: cos(pi/n * (i + 0.5) * k)
    basis_1d = torch.cos(math.pi / n * (i.unsqueeze(1) + 0.5) * k.unsqueeze(0))  # (n, n)

    # Нормализация
    basis_1d[0] *= 1.0 / math.sqrt(n)
    basis_1d[1:] *= math.sqrt(2.0 / n)

    # 2D DCT = внешнее произведение 1D базисов
    filters = torch.einsum("ij,kl->ikjl", basis_1d, basis_1d)  # (n, n, n, n)
    filters = filters.reshape(n * n, 1, n, n)
    return filters


class DCTLayer(nn.Module):
    """
    Применяет DCT к патчам изображения без обучаемых параметров.
    Используется для извлечения Local Frequency Statistics (LFS).

    Args:
        patch_size: размер патча для DCT (8 — стандарт JPEG, 16 — более крупные паттерны)
        n_components: сколько низкочастотных компонент оставлять (None = все patch_size²)
    """

    def __init__(self, patch_size: int = 8, n_components: Optional[int] = None) -> None:
        super().__init__()
        self.patch_size  = patch_size
        self.n_comp      = n_components or patch_size * patch_size

        # Регистрируем как buffer — не обучается, но сохраняется в state_dict
        filters = _dct_filters(patch_size, device=torch.device("cpu"))
        self.register_buffer("dct_filters", filters)  # (P², 1, P, P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Возвращает: (B, C * n_comp, H // P, W // P)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        # Паддинг если H/W не делится на P
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Применяем DCT к каждому каналу отдельно
        out_channels = []
        for c in range(C):
            xc = x[:, c:c+1, :, :]  # (B, 1, H', W')
            # Grouped conv: каждый фильтр = одна DCT-компонента
            dct_c = F.conv2d(
                xc,
                self.dct_filters[:self.n_comp],  # (n_comp, 1, P, P)
                stride=P,
                padding=0,
            )  # (B, n_comp, H'//P, W'//P)
            out_channels.append(dct_c)

        return torch.cat(out_channels, dim=1)  # (B, C*n_comp, H//P, W//P)


# ──────────────────────────────────────────────────────────────────────────────
# FAD — Frequency-Aware Decomposition
# ──────────────────────────────────────────────────────────────────────────────

class FADHead(nn.Module):
    """
    Frequency-Aware Decomposition Head.

    Разбивает входное изображение на частотные полосы через FFT
    и обрабатывает их свёрточными ветками.

    Три полосы:
      low  — 0–20% максимальной частоты (общая структура)
      mid  — 20–50%                      (текстуры)
      high — 50–100%                     (мелкие детали, артефакты)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()

        # Три независимые ветки для трёх частотных полос
        def _branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
            )

        self.low_branch  = _branch()
        self.mid_branch  = _branch()
        self.high_branch = _branch()

        # Слияние трёх полос
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    @staticmethod
    def _frequency_mask(
        h: int, w: int, low: float, high: float, device: torch.device
    ) -> torch.Tensor:
        """Создаёт радиальную маску в частотном домене."""
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=device).float() - cy
        x = torch.arange(w, device=device).float() - cx
        radius = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
        max_r  = math.sqrt(cy ** 2 + cx ** 2)
        mask   = (radius >= low * max_r) & (radius < high * max_r)
        return mask.float()

    def _decompose(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FFT → применяем маски → iFFT.
        Возвращает три компоненты: (low, mid, high).
        """
        B, C, H, W = x.shape
        dev = x.device

        # 2D FFT
        X = torch.fft.fft2(x, norm="ortho")
        X = torch.fft.fftshift(X, dim=(-2, -1))

        # Маски частотных полос
        m_low  = self._frequency_mask(H, W, 0.00, 0.20, dev)
        m_mid  = self._frequency_mask(H, W, 0.20, 0.50, dev)
        m_high = self._frequency_mask(H, W, 0.50, 1.01, dev)

        def _recon(mask: torch.Tensor) -> torch.Tensor:
            filtered = X * mask.unsqueeze(0).unsqueeze(0)
            filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
            return torch.fft.ifft2(filtered, norm="ortho").real

        return _recon(m_low), _recon(m_mid), _recon(m_high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, mid, high = self._decompose(x)
        f_low  = self.low_branch(low)
        f_mid  = self.mid_branch(mid)
        f_high = self.high_branch(high)
        return self.fusion(torch.cat([f_low, f_mid, f_high], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# LFS — Local Frequency Statistics Branch
# ──────────────────────────────────────────────────────────────────────────────

class LFSBranch(nn.Module):
    """
    Local Frequency Statistics Branch.

    Извлекает локальные частотные статистики через DCT патчей,
    затем обрабатывает через CNN.

    Для детекции дипфейков особенно важны высокочастотные компоненты DCT:
    face-swap генераторы оставляют характерные паттерны в DCT-домене,
    которые видны даже после JPEG-сжатия.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        patch_size: int = 8,
        n_dct_components: int = 32,
    ) -> None:
        super().__init__()
        self.dct = DCTLayer(patch_size=patch_size, n_components=n_dct_components)

        dct_out_channels = in_channels * n_dct_components
        self.conv = nn.Sequential(
            nn.Conv2d(dct_out_channels, out_channels * 2, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DCT: (B, C*n_dct, H//P, W//P)
        freq = self.dct(x)
        return self.conv(freq)


# ──────────────────────────────────────────────────────────────────────────────
# Xception backbone (упрощённый, только Entry+Middle flow для embedding)
# ──────────────────────────────────────────────────────────────────────────────

class _SepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=False)


class _XBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _SepConv(in_ch, out_ch)
        self.conv2 = _SepConv(out_ch, out_ch)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else None
        self.pool  = nn.MaxPool2d(3, stride=stride, padding=1) if stride > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x) if self.skip else x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.pool:
            out = self.pool(out)
        return F.relu(out + residual, inplace=False)


class XceptionBackbone(nn.Module):
    """
    Облегчённый Xception backbone для F3Net.
    Entry Flow → первые 4 Middle Flow блока → выдаёт 728-канальные признаки.
    """

    def __init__(self) -> None:
        super().__init__()
        # Entry Flow
        self.entry = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            _XBlock(64,  128, stride=2),
            _XBlock(128, 256, stride=2),
            _XBlock(256, 728, stride=2),
        )
        # Middle Flow (4 блока)
        self.middle = nn.Sequential(
            *[_XBlock(728, 728) for _ in range(4)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.middle(self.entry(x))


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Fusion
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Объединяет пространственные признаки backbone с частотными признаками
    FAD и LFS через механизм кросс-внимания.

    spatial (728ch) × freq_combined (128ch) → fused (728ch)
    """

    def __init__(self, spatial_ch: int = 728, freq_ch: int = 128) -> None:
        super().__init__()
        # Проецируем частотные признаки в то же пространство что и spatial
        self.freq_proj = nn.Conv2d(freq_ch, spatial_ch, 1, bias=False)
        # Attention map: spatial × freq → веса
        self.attn = nn.Sequential(
            nn.Conv2d(spatial_ch * 2, spatial_ch, 1, bias=False),
            nn.BatchNorm2d(spatial_ch),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(spatial_ch, spatial_ch, 1, bias=False),
            nn.BatchNorm2d(spatial_ch),
            nn.ReLU(inplace=False),
        )

    def forward(
        self, spatial: torch.Tensor, freq: torch.Tensor
    ) -> torch.Tensor:
        # Приводим частотные признаки к размеру spatial
        freq_up = F.interpolate(
            self.freq_proj(freq),
            size=spatial.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        # Attention: где пространственные и частотные признаки согласуются
        attn_map = self.attn(torch.cat([spatial, freq_up], dim=1))
        fused = spatial + attn_map * freq_up
        return self.out_conv(fused)


# ──────────────────────────────────────────────────────────────────────────────
# F3Net
# ──────────────────────────────────────────────────────────────────────────────

class F3Net(nn.Module):
    """
    F3Net: Frequency-aware Forgery Feature Network.

    Три параллельных потока:
      1. Spatial stream  — Xception backbone на RGB-изображении
      2. FAD stream      — FFT-декомпозиция на частотные полосы
      3. LFS stream      — локальные DCT-статистики патчей

    Все три объединяются через Cross-Attention Fusion,
    затем классифицируются финальной головой.

    Args:
        num_classes:      число классов (2)
        dropout_rate:     dropout перед классификатором
        fad_out_ch:       число каналов FAD-ветки
        lfs_out_ch:       число каналов LFS-ветки
        dct_patch_size:   размер патча для DCT (8 = JPEG-совместимый)
        n_dct_components: сколько DCT-компонент использовать
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        fad_out_ch: int = 64,
        lfs_out_ch: int = 64,
        dct_patch_size: int = 8,
        n_dct_components: int = 32,
    ) -> None:
        super().__init__()

        # 1. Spatial stream
        self.spatial_backbone = XceptionBackbone()   # → (B, 728, H/32, W/32)

        # 2. FAD stream
        self.fad = FADHead(in_channels=3, out_channels=fad_out_ch)

        # 3. LFS stream
        self.lfs = LFSBranch(
            in_channels=3,
            out_channels=lfs_out_ch,
            patch_size=dct_patch_size,
            n_dct_components=n_dct_components,
        )

        freq_ch = fad_out_ch + lfs_out_ch  # 128

        # Fusion
        self.fusion = CrossAttentionFusion(spatial_ch=728, freq_ch=freq_ch)

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(728, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Пространственные признаки
        spatial = self.spatial_backbone(x)           # (B, 728, ~9, ~9)

        # Частотные признаки
        fad_feat = self.fad(x)                        # (B, 64, H, W)
        lfs_feat = self.lfs(x)                        # (B, 64, H//P, W//P)

        # Приводим LFS к размеру FAD
        lfs_feat = F.interpolate(
            lfs_feat, size=fad_feat.shape[2:],
            mode="bilinear", align_corners=False,
        )
        freq = torch.cat([fad_feat, lfs_feat], dim=1)  # (B, 128, H, W)

        # Cross-Attention Fusion
        fused = self.fusion(spatial, freq)             # (B, 728, ~9, ~9)

        return self.head(fused)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """728-мерный вектор признаков после fusion (до классификатора)."""
        spatial  = self.spatial_backbone(x)
        fad_feat = self.fad(x)
        lfs_feat = F.interpolate(
            self.lfs(x), size=fad_feat.shape[2:],
            mode="bilinear", align_corners=False,
        )
        freq  = torch.cat([fad_feat, lfs_feat], dim=1)
        fused = self.fusion(spatial, freq)
        return torch.flatten(nn.functional.adaptive_avg_pool2d(fused, (1, 1)), 1)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_f3net(
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    weights_path: Optional[str] = None,
) -> F3Net:
    """Фабричная функция для F3Net."""
    model = F3Net(num_classes=num_classes, dropout_rate=dropout_rate)

    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"[F3Net] Loaded weights from {weights_path}")

    total = model.total_params()
    print(f"[F3Net] Total params: {total:,}")
    return model


if __name__ == "__main__":
    model = build_f3net()
    x = torch.randn(2, 3, 299, 299)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # (2, 2)