import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from src.config import load_config
from src.models.improved_cnn import build_model as build_improved_cnn
from src.data.transforms import get_medical_transforms


def apply_test_time_augmentation(image_path, model, device, config, num_augmentations=5):
    """Apply test-time augmentation to improve prediction reliability"""
    
    predictions = []
    confidences = []
    
    # Load original image
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Original image prediction
        transform = get_medical_transforms(config.train.image_size, is_training=False)
        original_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(original_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
            confidence = probs.max().item()
        
        predictions.append(pred_class)
        confidences.append(confidence)
        
        # Apply augmentations
        augmentations = [
            lambda x: x.rotate(5),  # Slight rotation
            lambda x: x.rotate(-5),  # Counter rotation
            lambda x: ImageEnhance.Brightness(x).enhance(1.1),  # Brightness
            lambda x: ImageEnhance.Contrast(x).enhance(1.1),  # Contrast
            lambda x: x.filter(ImageFilter.SMOOTH),  # Smoothing
        ]
        
        for i, aug_func in enumerate(augmentations[:num_augmentations-1]):
            try:
                # Apply augmentation
                aug_img = aug_func(img)
                
                # Predict on augmented image
                aug_tensor = transform(aug_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(aug_tensor)
                    probs = F.softmax(outputs, dim=1)
                    pred_class = outputs.argmax(dim=1).item()
                    confidence = probs.max().item()
                
                predictions.append(pred_class)
                confidences.append(confidence)
                
            except Exception as e:
                print(f"Augmentation {i+1} failed: {e}")
                continue
    
    return predictions, confidences


def ensemble_predict(image_path, checkpoint_path, config_path, use_pretrained=True):
    """Ensemble prediction using test-time augmentation"""
    
    print(f"🔍 Ensemble Analysis: {os.path.basename(image_path)}")
    
    # Load model and config
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_improved_cnn(num_classes=cfg.train.num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Get predictions with augmentation
    predictions, confidences = apply_test_time_augmentation(
        image_path, model, device, cfg, num_augmentations=5
    )
    
    # Calculate ensemble statistics
    class_names = cfg.train.class_names
    
    # Most common prediction
    from collections import Counter
    pred_counter = Counter(predictions)
    most_common_pred = pred_counter.most_common(1)[0][0]
    prediction_confidence = pred_counter[most_common_pred] / len(predictions)
    
    # Average confidence
    avg_confidence = np.mean(confidences)
    
    # Prediction consistency
    consistency = pred_counter[most_common_pred] / len(predictions)
    
    predicted_label = class_names[most_common_pred]
    
    # Print results
    print("\n" + "="*60)
    print("🧠 ENSEMBLE PREDICTION RESULTS")
    print("="*60)
    print(f"📸 Image: {os.path.basename(image_path)}")
    print(f"🎯 Final Prediction: {predicted_label}")
    print(f"📊 Average Confidence: {avg_confidence:.2%}")
    print(f"🔄 Prediction Consistency: {consistency:.2%}")
    
    print(f"\n📈 Individual Predictions:")
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        pred_name = class_names[pred]
        print(f"  Augmentation {i+1}: {pred_name} ({conf:.2%})")
    
    print(f"\n📊 Prediction Distribution:")
    for pred_class, count in pred_counter.items():
        pred_name = class_names[pred_class]
        percentage = count / len(predictions) * 100
        print(f"  {pred_name}: {count}/{len(predictions)} ({percentage:.1f}%)")
    
    # Interpretation
    print(f"\n💡 Ensemble Interpretation:")
    if consistency > 0.8:
        print("  ✅ HIGH CONSISTENCY: Model is very confident")
    elif consistency > 0.6:
        print("  🟡 MODERATE CONSISTENCY: Model is reasonably confident")
    else:
        print("  🔴 LOW CONSISTENCY: Model is uncertain")
    
    if predicted_label == "Stroke":
        if avg_confidence > 0.8:
            print("  ⚠️  HIGH CONFIDENCE: Strong indication of brain stroke")
        else:
            print("  ⚠️  MODERATE CONFIDENCE: Possible stroke detected")
    else:
        if avg_confidence > 0.8:
            print("  ✅ HIGH CONFIDENCE: Normal brain scan")
        else:
            print("  ✅ MODERATE CONFIDENCE: Likely normal brain scan")
    
    print("="*60)
    
    return predicted_label, avg_confidence, consistency


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction with test-time augmentation")
    parser.add_argument("image_path", help="Path to the brain image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_epoch_64.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/optimized.yaml",
                       help="Path to configuration file")
    parser.add_argument("--use_pretrained", action="store_true", default=True,
                       help="Use pre-trained ResNet model")
    args = parser.parse_args()
    
    try:
        ensemble_predict(args.image_path, args.checkpoint, args.config, args.use_pretrained)
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
