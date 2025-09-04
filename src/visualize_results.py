import argparse
import os
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.models.cnn import build_model as build_cnn
from src.models.improved_cnn import build_model as build_improved_cnn
from src.utils.training import evaluate
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd


def create_confusion_matrix_plot(cm, class_names, save_path="results/confusion_matrix.png"):
    """Create a beautiful confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def create_performance_metrics_plot(results, save_path="results/performance_metrics.png"):
    """Create a bar chart of performance metrics"""
    metrics = ['Precision', 'Recall', 'F1-Score']
    stroke_scores = [results['Stroke']['precision'], results['Stroke']['recall'], results['Stroke']['f1-score']]
    normal_scores = [results['Normal']['precision'], results['Normal']['recall'], results['Normal']['f1-score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, stroke_scores, width, label='Stroke', color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar(x + width/2, normal_scores, width, label='Normal', color='#4ecdc4', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance by Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance metrics saved to: {save_path}")


def create_roc_curve_plot(y_true, y_scores, class_names, save_path="results/roc_curve.png"):
    """Create ROC curve visualization"""
    plt.figure(figsize=(10, 8))
    
    # For binary classification, we only need one ROC curve
    if len(class_names) == 2:
        # Use the positive class (Stroke) probabilities
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='#ff6b6b', lw=2,
                label=f'Stroke Detection (AUC = {roc_auc:.3f})')
    else:
        # Multi-class case
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Calculate ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        for i, color in enumerate(colors[:len(class_names)]):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def create_summary_dashboard(results, save_path="results/summary_dashboard.png"):
    """Create a comprehensive summary dashboard"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall Accuracy
    accuracy = results['accuracy']
    ax1.pie([accuracy, 1-accuracy], labels=[f'Correct\n{accuracy:.1%}', f'Incorrect\n{(1-accuracy):.1%}'], 
            colors=['#4ecdc4', '#ff6b6b'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    
    # 2. Class Distribution
    class_counts = [results['Stroke']['support'], results['Normal']['support']]
    ax2.bar(['Stroke', 'Normal'], class_counts, color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
    ax2.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    for i, v in enumerate(class_counts):
        ax2.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 3. Performance Comparison
    metrics = ['Precision', 'Recall', 'F1-Score']
    stroke_scores = [results['Stroke']['precision'], results['Stroke']['recall'], results['Stroke']['f1-score']]
    normal_scores = [results['Normal']['precision'], results['Normal']['recall'], results['Normal']['f1-score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, stroke_scores, width, label='Stroke', color='#ff6b6b', alpha=0.8)
    ax3.bar(x + width/2, normal_scores, width, label='Normal', color='#4ecdc4', alpha=0.8)
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Performance by Class', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # 4. Key Statistics
    stats_text = f"""
    Model Performance Summary
    
    Overall Accuracy: {accuracy:.1%}
    
    Stroke Detection:
    • Precision: {results['Stroke']['precision']:.1%}
    • Recall: {results['Stroke']['recall']:.1%}
    • F1-Score: {results['Stroke']['f1-score']:.1%}
    
    Normal Detection:
    • Precision: {results['Normal']['precision']:.1%}
    • Recall: {results['Normal']['recall']:.1%}
    • F1-Score: {results['Normal']['f1-score']:.1%}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Key Statistics', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Brain Stroke Detection Model - Performance Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary dashboard saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/optimized.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_epoch_64.pt")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pre-trained ResNet model")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save visualizations")
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

    # Evaluate and collect predictions
    val_loss, val_acc = evaluate(model, val_loader, device, class_weights.to(device))
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Collect predictions and probabilities
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Generate classification report
    report = classification_report(all_targets, all_preds, target_names=cfg.train.class_names, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Create visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    create_confusion_matrix_plot(cm, cfg.train.class_names, 
                                os.path.join(args.output_dir, "confusion_matrix.png"))
    
    # 2. Performance Metrics
    create_performance_metrics_plot(report, 
                                   os.path.join(args.output_dir, "performance_metrics.png"))
    
    # 3. ROC Curve
    create_roc_curve_plot(all_targets, all_probs, cfg.train.class_names,
                          os.path.join(args.output_dir, "roc_curve.png"))
    
    # 4. Summary Dashboard
    create_summary_dashboard(report, 
                            os.path.join(args.output_dir, "summary_dashboard.png"))
    
    print(f"\nAll visualizations saved to: {args.output_dir}/")
    print("="*50)


if __name__ == "__main__":
    main()
