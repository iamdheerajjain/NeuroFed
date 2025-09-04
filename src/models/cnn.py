from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
	def __init__(self, num_classes: int = 3) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),

			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
		)
		self.classifier = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(128, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(128, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.classifier(x)
		return x


def build_model(num_classes: int) -> nn.Module:
	return SmallCNN(num_classes=num_classes)
