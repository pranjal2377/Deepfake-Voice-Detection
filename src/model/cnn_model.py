"""
CNN Model Stub — Phase 1 skeleton.
Full implementation in Phase 3.
"""

import torch
import torch.nn as nn
from src.utils.config import N_MELS


class DeepfakeCNN(nn.Module):
    """
    CNN model for deepfake voice detection.
    Takes a mel-spectrogram as input and outputs a probability.
    Full architecture will be built in Phase 3.
    """

    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.n_mels = n_mels

        # ---- Conv Block 1 ----
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ---- Conv Block 2 ----
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ---- Conv Block 3 ----
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, time_frames) mel-spectrogram tensor
        Returns:
            (batch, 1) deepfake probability
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x
