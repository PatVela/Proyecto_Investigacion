"""Redes ECG 1D: ResNet-34 inspirada en Hannun y CNN convencional equivalente.

La implementación usa tensores PyTorch con forma ``(batch, channels, time)``.
Para CINC2020-12, la entrada esperada es ``(batch, 12, 5000)`` y la salida son
``num_classes=12`` logits independientes.  La sigmoid se aplica fuera del modelo
para evaluación/inferencia multilabel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_SUBSAMPLE_LENGTHS = [1, 2] * 8


def _num_filters_at(block_index: int, num_start_filters: int, increase_channels_at: int = 4) -> int:
    """Duplica filtros cada ``increase_channels_at`` bloques."""
    return int(2 ** (block_index // increase_channels_at)) * int(num_start_filters)


class ConvBNReLU(nn.Module):
    """Conv1d con padding tipo 'same' aproximado para kernel impar."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """Bloque residual 1D con dos convoluciones."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        # En señales con longitudes impares y stride > 1 puede haber diferencia
        # de 1 muestra por redondeos de Conv1d. Se recorta al mínimo común.
        if out.shape[-1] != shortcut.shape[-1]:
            min_len = min(out.shape[-1], shortcut.shape[-1])
            out = out[..., :min_len]
            shortcut = shortcut[..., :min_len]
        return out + shortcut


class PlainBlock(nn.Module):
    """Bloque convolucional sin conexión residual, equivalente en profundidad."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class ECGBackbone(nn.Module):
    """Backbone común para ResNet y CNN convencional."""

    def __init__(
        self,
        block_cls,
        num_leads: int = 12,
        conv_filter_length: int = 7,
        conv_num_filters_start: int = 32,
        conv_subsample_lengths: list[int] | None = None,
        conv_dropout: float = 0.2,
        increase_channels_at: int = 4,
    ):
        super().__init__()
        if conv_subsample_lengths is None:
            conv_subsample_lengths = DEFAULT_SUBSAMPLE_LENGTHS
        self.conv_subsample_lengths = list(conv_subsample_lengths)

        self.first = ConvBNReLU(
            num_leads,
            conv_num_filters_start,
            kernel_size=conv_filter_length,
            stride=1,
            dropout=0.0,
        )

        blocks = []
        in_filters = conv_num_filters_start
        for block_index, stride in enumerate(self.conv_subsample_lengths):
            out_filters = _num_filters_at(block_index, conv_num_filters_start, increase_channels_at)
            blocks.append(
                block_cls(
                    in_filters,
                    out_filters,
                    kernel_size=conv_filter_length,
                    stride=int(stride),
                    dropout=conv_dropout,
                )
            )
            in_filters = out_filters

        self.blocks = nn.Sequential(*blocks)
        self.final_channels = in_filters
        self.final_bn = nn.BatchNorm1d(in_filters)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        x = self.final_relu(self.final_bn(x))
        return x


class ECGClassifier(nn.Module):
    """Clasificador ECG multilabel.

    Si ``is_regular_conv=False`` usa bloques residuales. Si es ``True`` usa una
    CNN convencional con el mismo calendario de filtros, convoluciones y
    downsampling, pero sin shortcuts.
    """

    def __init__(
        self,
        num_leads: int = 12,
        num_classes: int = 12,
        input_length: int = 5000,
        conv_filter_length: int = 7,
        conv_num_filters_start: int = 32,
        conv_subsample_lengths: list[int] | None = None,
        conv_num_skip: int = 2,
        conv_dropout: float = 0.2,
        is_regular_conv: bool = False,
        **unused,
    ):
        super().__init__()
        if conv_num_skip != 2:
            raise ValueError("Esta implementación usa conv_num_skip=2 para ResNet-34/CNN equivalente.")
        self.num_leads = int(num_leads)
        self.num_classes = int(num_classes)
        self.input_length = int(input_length)
        self.is_regular_conv = bool(is_regular_conv)
        block_cls = PlainBlock if self.is_regular_conv else ResidualBlock
        self.backbone = ECGBackbone(
            block_cls=block_cls,
            num_leads=self.num_leads,
            conv_filter_length=int(conv_filter_length),
            conv_num_filters_start=int(conv_num_filters_start),
            conv_subsample_lengths=conv_subsample_lengths,
            conv_dropout=float(conv_dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.backbone.final_channels, self.num_classes)

    @property
    def model_type(self) -> str:
        return "CNN convencional equivalente" if self.is_regular_conv else "ResNet-34 1D tipo Hannun"

    def logits(self, x):
        h = self.backbone(x)
        h = self.pool(h).squeeze(-1)
        return self.classifier(h)

    def forward(self, x):
        # Devuelve logits; usar torch.sigmoid(logits) fuera del modelo.
        return self.logits(x)


# Alias explícitos para documentación/tests.
class ECGResNet34(ECGClassifier):
    def __init__(self, **kwargs):
        kwargs["is_regular_conv"] = False
        super().__init__(**kwargs)


class ECGRegularCNN(ECGClassifier):
    def __init__(self, **kwargs):
        kwargs["is_regular_conv"] = True
        super().__init__(**kwargs)


def build_network(**params) -> ECGClassifier:
    """Construye ResNet o CNN convencional según ``is_regular_conv``."""
    return ECGClassifier(**params)


def ECG_model(config) -> ECGClassifier:
    """Compatibilidad con scripts previos que pasan un objeto de configuración."""
    params = vars(config).copy() if hasattr(config, "__dict__") else dict(config)
    return build_network(**params)
