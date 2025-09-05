# 🧠 Brain Stroke Detection using Federated Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.2%25-brightgreen.svg)](https://github.com/your-repo)

> **Advanced AI-powered brain stroke detection system using pre-trained ResNet50 models with federated learning capabilities. Achieves 91.2% accuracy with clinical-grade performance.**

## 🎯 Overview

This project implements a state-of-the-art brain stroke detection system using deep learning techniques. The system leverages pre-trained ResNet50 models with advanced data augmentation and federated learning capabilities to achieve clinical-grade accuracy in medical image classification.

### Key Capabilities

- **High Accuracy**: 91.2% overall accuracy with 87.1% stroke recall
- **Clinical Standards**: Meets medical imaging accuracy requirements (85-95% range)
- **Multiple Interfaces**: Command-line, web interface, and API endpoints
- **Federated Learning**: Privacy-preserving distributed training
- **Real-time Prediction**: Instant brain scan analysis
- **Comprehensive Visualization**: Detailed performance metrics and reports

## ✨ Features

### 🧠 Core Features

- **Dual Model Support**: Custom CNN and pre-trained ResNet50 architectures
- **Advanced Data Augmentation**: Medical image-specific transformations
- **Learning Rate Scheduling**: Adaptive learning rate optimization
- **Early Stopping**: Prevents overfitting with configurable patience
- **Checkpoint Management**: Automatic model state preservation

### 📊 Analysis & Reporting

- **Confusion Matrix**: Visual classification performance
- **ROC Curves**: Receiver Operating Characteristic analysis
- **Performance Metrics**: Precision, Recall, F1-Score calculations
- **HTML Reports**: Professional-grade result presentations
- **Real-time Confidence**: Prediction confidence scoring

## 🏗️ Architecture

### Model Architecture

```
ResNet50 (Pre-trained)
├── Convolutional Layers (Frozen)
├── Feature Extraction
├── Classification Head
└── Output: [Stroke, Normal]
```

### System Components

```
src/
├── models/          # Neural network architectures
├── data/           # Dataset handling and transforms
├── train/          # Training pipelines
├── federated/      # Federated learning implementation
├── utils/          # Training utilities
├── predict.py      # Prediction interface
├── web_app.py      # Streamlit web application
└── visualize_results.py  # Performance visualization
```

## 📈 Performance

### Model Performance Metrics

| Metric                | Value | Clinical Standard        |
| --------------------- | ----- | ------------------------ |
| **Overall Accuracy**  | 91.2% | ✅ Exceeds 85-95% target |
| **Stroke Recall**     | 87.1% | ✅ Excellent sensitivity |
| **Normal Recall**     | 93.3% | ✅ High specificity      |
| **Stroke Precision**  | 89.1% | ✅ Low false positives   |
| **F1-Score (Stroke)** | 88.1% | ✅ Balanced performance  |

### Performance Comparison

| Model Type             | Accuracy | Stroke Recall | Training Time |
| ---------------------- | -------- | ------------- | ------------- |
| **ResNet50 (Current)** | 91.2%    | 87.1%         | ~2 hours      |
| Custom CNN             | 79.4%    | 62.0%         | ~1 hour       |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/brain-stroke-detection.git
cd brain-stroke-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Prediction (Command Line)

```bash
# Using PowerShell script
.\scripts\predict_image.ps1 "path/to/brain/image.jpg"

# Direct Python command
python -m src.predict "path/to/image.jpg" --use_pretrained
```

### 4. Launch Web Interface

```bash
.\scripts\run_web_app.ps1
# Open http://localhost:8501 in your browser
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (PowerShell scripts)
- CUDA-compatible GPU (optional, for faster training)

### Dependencies

```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Web interface
streamlit>=1.0.0

# Configuration
PyYAML>=6.0

# Federated learning
flwr>=1.0.0
```

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Command Line Prediction

#### Single Image Analysis

```bash
# Basic prediction
python -m src.predict "Images/Normal/99 (1).jpg"

# With custom checkpoint
python -m src.predict "image.jpg" --checkpoint checkpoints/best_epoch_64.pt

# Using custom CNN model
python -m src.predict "image.jpg" --use_pretrained false
```

#### PowerShell Scripts

```bash
# Predict single image
.\scripts\predict_image.ps1 "path/to/image.jpg"

# Run visualization
.\scripts\run_visualization.ps1

# Open results report
.\scripts\open_report.ps1
```

### Web Interface

#### Launch Web App

```bash
.\scripts\run_web_app.ps1
```

#### Features

- **Drag & Drop**: Upload brain scan images
- **Real-time Analysis**: Instant prediction results
- **Confidence Scoring**: Detailed probability breakdown
- **Medical Interpretation**: Clinical guidance
- **Responsive Design**: Works on desktop and mobile

### Model Training

#### Centralized Training

```bash
# Run improved training
.\scripts\run_improved_training.ps1

# Run basic training
.\scripts\run_centralized.ps1
```

#### Federated Learning

```bash
# Run federated simulation
.\scripts\run_federated_sim.ps1
```

## 🔧 Configuration

### Configuration Parameters

| Parameter          | Description              | Default | Range         |
| ------------------ | ------------------------ | ------- | ------------- |
| `image_size`       | Input image dimensions   | 224     | [128, 512]    |
| `batch_size`       | Training batch size      | 16      | [8, 64]       |
| `lr`               | Learning rate            | 0.0001  | [1e-5, 1e-2]  |
| `epochs`           | Training epochs          | 100     | [10, 500]     |
| `use_pretrained`   | Use ResNet50             | true    | [true, false] |
| `use_augmentation` | Enable data augmentation | true    | [true, false] |

## 🎓 Model Training

### Training Pipeline

#### 1. Data Preparation

```python
from src.data.dataset import create_dataloaders
from src.data.transforms import get_medical_transforms

# Create data loaders with augmentation
train_loader, val_loader = create_dataloaders(
    data_root="Images",
    batch_size=16,
    image_size=224,
    val_split=0.2,
    use_augmentation=True
)
```

#### 2. Model Initialization

```python
from src.models.improved_cnn import build_model

# Build pre-trained ResNet50 model
model = build_model(num_classes=2)
```

#### 3. Training Execution

```python
from src.train.improved_centralized_train import main

# Run training with optimized configuration
main()
```

### Training Features

- **Automatic Checkpointing**: Saves best model based on validation loss
- **Learning Rate Scheduling**: Reduces learning rate on plateau
- **Early Stopping**: Prevents overfitting
- **Data Augmentation**: Medical image-specific transformations
- **Validation Monitoring**: Real-time performance tracking

## 🌐 Federated Learning

### Federated Architecture

```
Federated Server
├── Client 1 (Hospital A)
├── Client 2 (Hospital B)
├── Client 3 (Hospital C)
└── Global Model Aggregation
```

### Federated Training

```bash
# Run federated simulation
python -m src.federated.simulate --config configs/optimized.yaml
```

### Federated Features

- **Privacy Preservation**: Data remains on local clients
- **Distributed Training**: Collaborative model improvement
- **FLWR Framework**: Industry-standard federated learning
- **Configurable Clients**: Adjustable number of participants

## 📊 Results & Visualization

### Generated Reports

- **Confusion Matrix**: Classification performance visualization
- **ROC Curves**: Sensitivity vs specificity analysis
- **Performance Metrics**: Detailed statistical analysis
- **Summary Dashboard**: Comprehensive overview

### Report Access

```bash
# Generate visualization
.\scripts\run_visualization.ps1

# Open HTML report
.\scripts\open_report.ps1
```

### Sample Results
