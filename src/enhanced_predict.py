import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from src.config import load_config
from src.models.cnn import build_model as build_cnn
from src.models.improved_cnn import build_model as build_improved_cnn
from src.data.transforms import get_medical_transforms


def _infer_class_mapping(model, device, data_root, transform, candidate_class_names, max_per_class: int = 8):
    """Infer output-index→class-name mapping using a few images per class.

    If folders under data_root match candidate_class_names, compute mean logits
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
                    mean_logits = torch.cat(logits_list, dim=0).mean(dim=0)
                    class_to_mean_logits[cls] = mean_logits

        if not class_to_mean_logits:
            return candidate_class_names

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

        if any(c is None for c in index_to_class) or len(set(index_to_class)) != len(index_to_class):
            return candidate_class_names

        return index_to_class
    except Exception:
        return candidate_class_names


def preprocess_image_for_prediction(image_path, target_size=224):
    """Enhanced image preprocessing for better generalization"""
    
    # Load image
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Try PIL first
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                original_size = img.size
                print(f"Original image size: {original_size}")
                
                # Apply medical transforms
                transform = get_medical_transforms(target_size, is_training=False)
                tensor = transform(img)
                return tensor, original_size
        except Exception as e:
            print(f"PIL failed: {e}")
    
    # Fallback to OpenCV
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = (img.shape[1], img.shape[0])
        print(f"Original image size: {original_size}")
        
        # Resize
        img = cv2.resize(img, (target_size, target_size))
        
        # Convert to PIL and apply transforms
        pil_img = Image.fromarray(img)
        transform = get_medical_transforms(target_size, is_training=False)
        tensor = transform(pil_img)
        return tensor, original_size
        
    except Exception as e:
        print(f"OpenCV failed: {e}")
        raise ValueError(f"Could not process image: {image_path}")


def predict_image_enhanced(image_path, checkpoint_path, config_path, use_pretrained=True):
    """Enhanced prediction with better error handling and debugging"""
    
    print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
    print(f"📁 Full path: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None, None, None
    
    # Check file size
    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
    print(f"📏 File size: {file_size:.2f} MB")
    
    # Load configuration
    try:
        cfg = load_config(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {device}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None, None, None
    
    # Load model
    try:
        if use_pretrained:
            print("🧠 Loading pre-trained ResNet50 model...")
            model = build_improved_cnn(num_classes=cfg.train.num_classes).to(device)
        else:
            print("🧠 Loading custom CNN model...")
            model = build_cnn(num_classes=cfg.train.num_classes).to(device)
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None
    
    # Preprocess image
    try:
        print("🔄 Preprocessing image...")
        input_tensor, original_size = preprocess_image_for_prediction(image_path, cfg.train.image_size)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        print(f"✅ Image preprocessed to: {input_tensor.shape}")
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None, None, None
    
    # Make prediction
    try:
        print("🤖 Making prediction...")
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        print("✅ Prediction completed")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, None, None
    
    # Infer/confirm class mapping from training data folders
    inferred_class_names = _infer_class_mapping(
        model=model,
        device=device,
        data_root=cfg.train.data_root,
        transform=get_medical_transforms(cfg.train.image_size, is_training=False),
        candidate_class_names=cfg.train.class_names,
    )
    class_names = inferred_class_names
    predicted_label = class_names[predicted_class]
    
    # Print detailed results
    print("\n" + "="*60)
    print("🧠 ENHANCED BRAIN STROKE DETECTION RESULTS")
    print("="*60)
    print(f"📸 Image: {os.path.basename(image_path)}")
    print(f"📏 Original Size: {original_size}")
    print(f"🎯 Prediction: {predicted_label}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"🤖 Model: {'ResNet50' if use_pretrained else 'Custom CNN'}")
    
    # Detailed probabilities
    print("\n📈 Detailed Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
        confidence_level = "🟢 HIGH" if prob > 0.8 else "🟡 MEDIUM" if prob > 0.6 else "🔴 LOW"
        print(f"  {class_name}: {prob.item():.2%} {confidence_level}")
    
    # Interpretation with more detail
    print("\n💡 Medical Interpretation:")
    if predicted_label == "Stroke":
        if confidence > 0.9:
            print("  ⚠️  HIGH CONFIDENCE: Strong indication of brain stroke")
            print("  🚨 IMMEDIATE ACTION: Seek medical attention immediately")
        elif confidence > 0.7:
            print("  ⚠️  MODERATE CONFIDENCE: Likely brain stroke")
            print("  🏥 MEDICAL ATTENTION: Professional evaluation recommended")
        else:
            print("  ⚠️  LOW CONFIDENCE: Possible stroke detected")
            print("  🔍 FURTHER EVALUATION: Medical review strongly advised")
    else:
        if confidence > 0.9:
            print("  ✅ HIGH CONFIDENCE: Normal brain scan")
            print("  👍 REASSURANCE: No immediate concerns detected")
        elif confidence > 0.7:
            print("  ✅ MODERATE CONFIDENCE: Likely normal brain scan")
            print("  📋 ROUTINE MONITORING: Continue regular health checks")
        else:
            print("  ⚠️  LOW CONFIDENCE: Uncertain classification")
            print("  🔍 PROFESSIONAL REVIEW: Medical consultation recommended")
    
    # Image quality assessment
    print("\n🔍 Image Quality Assessment:")
    if file_size < 0.1:
        print("  ⚠️  Small file size - may be low quality")
    elif file_size > 10:
        print("  ⚠️  Large file size - may need compression")
    else:
        print("  ✅ Appropriate file size")
    
    if original_size[0] < 100 or original_size[1] < 100:
        print("  ⚠️  Low resolution image")
    elif original_size[0] > 2000 or original_size[1] > 2000:
        print("  ✅ High resolution image")
    else:
        print("  ✅ Good resolution")
    
    print("="*60)
    
    return predicted_label, confidence, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Enhanced brain stroke prediction with debugging")
    parser.add_argument("image_path", help="Path to the brain image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_epoch_64.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/optimized.yaml",
                       help="Path to configuration file")
    parser.add_argument("--use_pretrained", action="store_true", default=True,
                       help="Use pre-trained ResNet model")
    args = parser.parse_args()
    
    try:
        predict_image_enhanced(args.image_path, args.checkpoint, args.config, args.use_pretrained)
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Ensure the image is a valid brain scan (CT, MRI, etc.)")
        print("  2. Check that the image format is supported (JPG, PNG, BMP)")
        print("  3. Verify the image is not corrupted")
        print("  4. Try with an image from your training dataset first")


if __name__ == "__main__":
    main()
