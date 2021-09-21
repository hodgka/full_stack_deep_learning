from typing import Any, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
NUM_BLOCKS=2


class Conv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, downsample: bool = False) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(self, input_channels: int, output_channels: int, downsample: bool = False) -> None:
        super().__init__()
        # stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        input = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = input + x
        return x


class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)

        self.conv1 = ConvBlock(input_dims[0], conv_dim)
        self.conv_blocks = [
            ConvBlock(conv_dim, conv_dim, downsample=False) for _ in range(1, args.num_blocks)
        ]
        self.conv2 = Conv(conv_dim, conv_dim, downsample=True)
        self.dropout = nn.Dropout(0.25)

        # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
        # The 2x2 max-pool divides the input size by 2. Flattening squares it.
        conv_output_size = IMAGE_SIZE // 2

        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.conv1(x)
        for block in self.conv_blocks:
            x = block(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument('--num_blocks', type=int, default=NUM_BLOCKS)
        return parser