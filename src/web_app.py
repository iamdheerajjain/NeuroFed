import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import sys

# Ensure project root is on sys.path for Streamlit/Cloud
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config
from src.models.improved_cnn import build_model as build_improved_cnn
from src.data.transforms import get_medical_transforms


def load_model(checkpoint_path, config_path):
    """Load the trained model"""
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_improved_cnn(num_classes=cfg.train.num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, cfg, device


def predict_image(model, image, cfg, device):
    """Predict brain stroke from image"""
    # Preprocess image
    transform = get_medical_transforms(cfg.train.image_size, is_training=False)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities.max().item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()


def main():
    st.set_page_config(
        page_title="Brain Stroke Detection",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Brain Stroke Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.info("""
    **Model:** ResNet50 (Pre-trained)
    
    **Accuracy:** 91.2%
    
    **Classes:** Stroke, Normal
    
    **Training Data:** Medical brain scan images
    """)
    
    st.sidebar.header("How to Use")
    st.sidebar.markdown("""
    1. Upload a brain scan image
    2. Click 'Analyze Image'
    3. View prediction results
    4. Check confidence levels
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Brain Scan Image")
        uploaded_file = st.file_uploader(
            "Choose a brain scan image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a brain scan image (CT, MRI, etc.)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
    
    with col2:
        st.header("🔍 Analysis Results")
        
        if uploaded_file is not None:
            # Check if model checkpoint exists
            checkpoint_path = "checkpoints/best_epoch_64.pt"
            config_path = "configs/optimized.yaml"
            
            if not os.path.exists(checkpoint_path):
                st.error("❌ Model checkpoint not found! Please train the model first.")
                return
            
            try:
                # Load model
                with st.spinner("Loading model..."):
                    model, cfg, device = load_model(checkpoint_path, config_path)
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, probabilities = predict_image(model, image, cfg, device)
                
                # Display results
                class_names = cfg.train.class_names
                predicted_label = class_names[predicted_class]
                
                # Result card
                if predicted_label == "Stroke":
                    st.error("⚠️ **STROKE DETECTED**")
                    if confidence > 0.9:
                        st.warning("High confidence - Strong indication of brain stroke")
                    elif confidence > 0.7:
                        st.warning("Moderate confidence - Medical attention recommended")
                    else:
                        st.warning("Low confidence - Further medical evaluation needed")
                else:
                    st.success("✅ **NORMAL BRAIN SCAN**")
                    if confidence > 0.9:
                        st.info("High confidence - Normal brain scan")
                    elif confidence > 0.7:
                        st.info("Moderate confidence - Likely normal brain scan")
                    else:
                        st.warning("Low confidence - Medical review recommended")
                
                # Confidence meter
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Detailed probabilities
                st.subheader("📊 Detailed Probabilities")
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                        if i == predicted_class:
                            st.metric(f"**{class_name}**", f"{prob:.1%}", delta="Predicted")
                        else:
                            st.metric(class_name, f"{prob:.1%}")
                
                # Interpretation
                st.subheader("💡 Medical Interpretation")
                if predicted_label == "Stroke":
                    st.markdown("""
                    **⚠️ Medical Attention Required**
                    - This scan shows signs of a brain stroke
                    - Immediate medical evaluation is recommended
                    - Contact a healthcare professional
                    - Do not ignore these results
                    """)
                else:
                    st.markdown("""
                    **✅ Normal Brain Scan**
                    - No signs of stroke detected
                    - Brain appears normal
                    - Continue regular health monitoring
                    - Consult doctor for any concerns
                    """)
                
                # Disclaimer
                st.subheader("⚠️ Important Disclaimer")
                st.markdown("""
                **This is an AI-assisted tool and should NOT replace professional medical diagnosis.**
                
                - Always consult with qualified healthcare professionals
                - This tool is for educational and research purposes only
                - Medical decisions should be made by licensed physicians
                - The accuracy of this tool is approximately 91.2%
                """)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please ensure you uploaded a valid brain scan image.")
        
        else:
            st.info("👆 Please upload a brain scan image to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧠 Brain Stroke Detection System | AI-Powered Medical Imaging Analysis</p>
        <p><small>For educational and research purposes only. Not for clinical use.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
