import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from src.config import load_config
from src.models.cnn import build_model as build_cnn
from src.models.improved_cnn import build_model as build_improved_cnn
from src.data.transforms import get_medical_transforms


def _infer_class_mapping(model, device, data_root, transform, candidate_class_names, max_per_class: int = 8):
    """Infer output-index→class-name mapping using a few images per class.

    If folders under data_root match candidate_class_names, we compute mean logits
    per output neuron for each folder and map each neuron to the class with
    the highest mean response. Falls back to candidate_class_names on failure.
    """
    try:
        import os
        from PIL import Image
        import torch
        model.eval()
        samples_by_class = {}
        for cls in candidate_class_names:
            cls_dir = os.path.join(data_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            # Collect a few images
            imgs = []
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    imgs.append(os.path.join(cls_dir, fname))
                if len(imgs) >= max_per_class:
                    break
            if imgs:
                samples_by_class[cls] = imgs

        if len(samples_by_class) < 2:
            return candidate_class_names

        import torch
        class_to_mean_logits = {}
        with torch.no_grad():
            for cls, paths in samples_by_class.items():
                logits_list = []
                for p in paths:
                    with Image.open(p) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        x = transform(img).unsqueeze(0).to(device)
                    logits = model(x)
                    logits_list.append(logits.detach().cpu())
                if logits_list:
                    mean_logits = torch.cat(logits_list, dim=0).mean(dim=0)  # [C]
                    class_to_mean_logits[cls] = mean_logits

        if not class_to_mean_logits:
            return candidate_class_names

        # Determine for each output index which class it best corresponds to
        num_outputs = next(iter(class_to_mean_logits.values())).numel()
        index_to_class = [None] * num_outputs
        used = set()
        for out_idx in range(num_outputs):
            best_cls = None
            best_val = float('-inf')
            for cls, vec in class_to_mean_logits.items():
                val = vec[out_idx].item()
                if val > best_val and cls not in used:
                    best_val = val
                    best_cls = cls
            if best_cls is None:
                return candidate_class_names
            index_to_class[out_idx] = best_cls
            used.add(best_cls)

        # If any None remain or duplicates exist, fallback
        if any(c is None for c in index_to_class) or len(set(index_to_class)) != len(index_to_class):
            return candidate_class_names

        return index_to_class
    except Exception:
        return candidate_class_names


def predict_image(image_path, checkpoint_path, config_path, use_pretrained=True):
    """Predict whether an image shows a brain stroke or normal brain"""
    
    # Load configuration
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if use_pretrained:
        print("Loading pre-trained ResNet50 model...")
        model = build_improved_cnn(num_classes=cfg.train.num_classes).to(device)
    else:
        print("Loading custom CNN model...")
        model = build_cnn(num_classes=cfg.train.num_classes).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Apply the same transforms used during training
        transform = get_medical_transforms(cfg.train.image_size, is_training=False)
        input_tensor = transform(img).unsqueeze(0).to(device)

    # Try to infer correct class mapping based on training data folders
    inferred_class_names = _infer_class_mapping(
        model=model,
        device=device,
        data_root=cfg.train.data_root,
        transform=get_medical_transforms(cfg.train.image_size, is_training=False),
        candidate_class_names=cfg.train.class_names,
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities.max().item()
    
    # Get class names (possibly corrected)
    class_names = inferred_class_names
    predicted_label = class_names[predicted_class]
    
    # Print results
    print("\n" + "="*50)
    print("🧠 BRAIN STROKE DETECTION RESULTS")
    print("="*50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Model: {'ResNet50' if use_pretrained else 'Custom CNN'}")
    
    # Detailed probabilities
    print("\n📊 Detailed Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
        print(f"  {class_name}: {prob.item():.2%}")
    
    # Interpretation
    print("\n💡 Interpretation:")
    if predicted_label == "Stroke":
        if confidence > 0.9:
            print("  ⚠️  HIGH CONFIDENCE: Strong indication of brain stroke")
        elif confidence > 0.7:
            print("  ⚠️  MODERATE CONFIDENCE: Likely brain stroke - medical attention recommended")
        else:
            print("  ⚠️  LOW CONFIDENCE: Possible stroke - further medical evaluation needed")
    else:
        if confidence > 0.9:
            print("  ✅ HIGH CONFIDENCE: Normal brain scan")
        elif confidence > 0.7:
            print("  ✅ MODERATE CONFIDENCE: Likely normal brain scan")
        else:
            print("  ⚠️  LOW CONFIDENCE: Uncertain - medical review recommended")
    
    print("="*50)
    
    return predicted_label, confidence, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict brain stroke from image")
    parser.add_argument("image_path", help="Path to the brain image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_epoch_62.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/optimized.yaml",
                       help="Path to configuration file")
    parser.add_argument("--use_pretrained", action="store_true", default=True,
                       help="Use pre-trained ResNet model")
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        print("Please train the model first or specify a valid checkpoint path.")
        return
    
    try:
        predict_image(args.image_path, args.checkpoint, args.config, args.use_pretrained)
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        print("Please ensure the image is a valid brain scan image.")


if __name__ == "__main__":
    main()
