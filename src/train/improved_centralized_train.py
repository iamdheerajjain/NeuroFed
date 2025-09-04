import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim

from src.config import load_config, ensure_dir
from src.data.dataset import create_dataloaders
from src.data.transforms import get_medical_transforms
from src.models.cnn import build_model as build_cnn
from src.models.improved_cnn import build_model as build_improved_cnn
from src.utils.training import TrainState, train_one_epoch, evaluate, maybe_save_checkpoint


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/optimized.yaml")
	args = parser.parse_args()

	cfg = load_config(args.config)
	set_seed(cfg.train.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Use improved transforms with augmentation
	train_loader, val_loader, class_weights = create_dataloaders(
		root=cfg.train.data_root,
		image_size=cfg.train.image_size,
		class_names=cfg.train.class_names,
		batch_size=cfg.train.batch_size,
		num_workers=cfg.train.num_workers,
		val_split=cfg.train.val_split,
		seed=cfg.train.seed,
	)
	
	# Use pre-trained model if specified
	if hasattr(cfg.train, 'use_pretrained') and cfg.train.use_pretrained:
		print("Using pre-trained ResNet50 model")
		model = build_improved_cnn(num_classes=cfg.train.num_classes).to(device)
	else:
		print("Using custom CNN model")
		model = build_cnn(num_classes=cfg.train.num_classes).to(device)

	# Print model summary
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")

	optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

	# Save class weights on device for use in training/eval
	class_weights = class_weights.to(device)
	
	# Add learning rate scheduler
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.5, patience=5, verbose=True
	)

	state = TrainState()
	ensure_dir(cfg.train.checkpoint_dir)

	print(f"Starting training for {cfg.train.epochs} epochs...")
	for epoch in range(1, cfg.train.epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, class_weights)
		val_loss, val_acc = evaluate(model, val_loader, device, class_weights)
		
		# Update learning rate
		scheduler.step(val_loss)
		
		state = maybe_save_checkpoint(state, model, val_loss, cfg.train.checkpoint_dir, epoch)
		print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
		
		if state.no_improve_epochs >= cfg.train.early_stopping_patience:
			print("Early stopping triggered.")
			break

	print(f"Best checkpoint: {state.best_path}")


if __name__ == "__main__":
	main()
