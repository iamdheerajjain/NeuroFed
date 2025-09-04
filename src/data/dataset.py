from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from src.data.transforms import get_medical_transforms


@dataclass
class SplitDatasets:
	train: Dataset
	val: Dataset


class ImageFolderDataset(Dataset):
	def __init__(
		self,
		root: str,
		image_size: int,
		class_names: List[str],
		transform: Optional[Callable] = None,
		is_training: bool = True,
	) -> None:
		self.root = root
		self.class_names = class_names
		self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
		self.image_size = image_size
		self.is_training = is_training
		
		# Use medical transforms if no custom transform provided
		if transform is None:
			self.transform = get_medical_transforms(image_size, is_training)
		else:
			self.transform = transform
			
		self.samples: List[Tuple[str, int]] = []
		self._load_from_folders()

	def _load_from_folders(self) -> None:
		for cls in self.class_names:
			cls_dir = os.path.join(self.root, cls)
			if not os.path.isdir(cls_dir):
				continue
			for fname in os.listdir(cls_dir):
				if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
					continue
				self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		path, label = self.samples[idx]
		with Image.open(path) as img:
			if img.mode != "RGB":
				img = img.convert("RGB")
			x = self.transform(img)
		y = torch.tensor(label, dtype=torch.long)
		return x, y


def create_dataloaders(
	root: str,
	image_size: int,
	class_names: List[str],
	batch_size: int,
	num_workers: int,
	val_split: float,
	seed: int,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
	
	# Create training dataset with augmentation
	train_dataset = ImageFolderDataset(
		root=root,
		image_size=image_size,
		class_names=class_names,
		is_training=True,
	)
	
	# Create validation dataset without augmentation
	val_dataset = ImageFolderDataset(
		root=root,
		image_size=image_size,
		class_names=class_names,
		is_training=False,
	)

	val_size = int(len(train_dataset) * val_split)
	train_size = len(train_dataset) - val_size
	generator = torch.Generator().manual_seed(seed)
	train_ds, val_ds = random_split(train_dataset, [train_size, val_size], generator=generator)

	# Compute class counts within the training subset for weighting
	class_counts = torch.zeros(len(class_names), dtype=torch.long)
	for idx in train_ds.indices:
		_, label = train_dataset.samples[idx]
		class_counts[label] += 1

	# Avoid division by zero
	class_counts = torch.clamp(class_counts, min=1)
	class_weights = 1.0 / class_counts.float()

	# Sample weights per item in the training subset
	sample_weights = []
	for idx in train_ds.indices:
		_, label = train_dataset.samples[idx]
		sample_weights.append(class_weights[label].item())
	sample_weights = torch.tensor(sample_weights, dtype=torch.double)

	train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

	train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_loader, val_loader, class_weights
