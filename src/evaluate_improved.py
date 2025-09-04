import argparse
import os
import torch
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.models.cnn import build_model as build_cnn
from src.models.improved_cnn import build_model as build_improved_cnn
from src.utils.training import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/optimized.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_epoch_64.pt")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pre-trained ResNet model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create validation dataloader
    _, val_loader, class_weights = create_dataloaders(
        root=cfg.train.data_root,
        image_size=cfg.train.image_size,
        class_names=cfg.train.class_names,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        val_split=cfg.train.val_split,
        seed=cfg.train.seed,
    )

    # Load model based on checkpoint type
    if args.use_pretrained or "best_epoch_64.pt" in args.checkpoint:
        print("Loading pre-trained ResNet50 model")
        model = build_improved_cnn(num_classes=cfg.train.num_classes).to(device)
    else:
        print("Loading custom CNN model")
        model = build_cnn(num_classes=cfg.train.num_classes).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Evaluate
    val_loss, val_acc = evaluate(model, val_loader, device, class_weights.to(device))
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Detailed evaluation with predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=cfg.train.class_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)


if __name__ == "__main__":
    main()
