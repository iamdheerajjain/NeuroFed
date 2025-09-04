from typing import Dict, Tuple, Optional
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainState:
	best_val_loss: float = float("inf")
	best_path: str = ""
	no_improve_epochs: int = 0


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
	pred = logits.argmax(dim=1)
	correct = (pred == targets).float().sum().item()
	return correct / max(1, targets.size(0))


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
	model.train()
	running_loss = 0.0
	running_acc = 0.0
	criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
	for images, targets in tqdm(loader, desc="train", leave=False):
		images = images.to(device)
		targets = targets.to(device)
		optimizer.zero_grad(set_to_none=True)
		logits = model(images)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()
		running_loss += loss.item() * targets.size(0)
		running_acc += accuracy_from_logits(logits, targets) * targets.size(0)
	n = len(loader.dataset)
	return running_loss / max(1, n), running_acc / max(1, n)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
	model.eval()
	running_loss = 0.0
	running_acc = 0.0
	criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
	with torch.no_grad():
		for images, targets in tqdm(loader, desc="val", leave=False):
			images = images.to(device)
			targets = targets.to(device)
			logits = model(images)
			loss = criterion(logits, targets)
			running_loss += loss.item() * targets.size(0)
			running_acc += accuracy_from_logits(logits, targets) * targets.size(0)
	n = len(loader.dataset)
	return running_loss / max(1, n), running_acc / max(1, n)


def maybe_save_checkpoint(state: TrainState, model: nn.Module, val_loss: float, out_dir: str, epoch: int) -> TrainState:
	if val_loss + 1e-9 < state.best_val_loss:
		state.best_val_loss = val_loss
		state.no_improve_epochs = 0
		os.makedirs(out_dir, exist_ok=True)
		path = os.path.join(out_dir, f"best_epoch_{epoch}.pt")
		torch.save(model.state_dict(), path)
		state.best_path = path
	else:
		state.no_improve_epochs += 1
	return state
