# Melanoma Skin Cancer Classification

A deep learning model for binary classification of melanoma skin cancer images using transfer learning with EfficientNet-B3.

## Overview

This project implements a convolutional neural network to classify skin lesion images as either benign or malignant. The model uses a pre-trained EfficientNet-B3 backbone with ImageNet weights and employs a two-phase training strategy: warmup (frozen backbone) followed by fine-tuning (unfrozen backbone).

## Features

- **Transfer Learning**: Leverages pre-trained EfficientNet-B3 (also supports ResNet50 and EfficientNet-B0)
- **Two-Phase Training**: Warmup phase with frozen backbone followed by full fine-tuning
- **Advanced Data Augmentation**: AutoAugment, random transformations, color jitter, random erasing
- **Class Balancing**: Weighted sampling to handle class imbalance
- **Test-Time Augmentation (TTA)**: Multiple augmented predictions averaged for improved accuracy
- **Comprehensive Metrics**: ROC-AUC, PR-AUC, F1-score, precision, recall, accuracy
- **Early Stopping**: Prevents overfitting with patience-based monitoring
- **Mixed Precision Training**: Faster training with automatic mixed precision (AMP)

## Dataset

The project uses the [Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) from Kaggle, containing 10,000 images split into benign and malignant classes.

### Dataset Structure

```
melanoma_cancer_dataset/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory (for batch size 256)

See `requirements.txt` for full package dependencies.

## Installation

1. Clone or download this repository:

```bash
cd "4AL3 Final Report"
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA 13.0, PyTorch is already specified. For other CUDA versions, visit [PyTorch installation guide](https://pytorch.org/get-started/locally/).

3. Set up Kaggle API credentials:
   - Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
   - Generate an API token from your account settings
   - The notebook will prompt for credentials when run

## Usage

### Training

Open `model.ipynb` in Jupyter Notebook or VS Code and run all cells sequentially. The notebook will:

1. Download the dataset from Kaggle
2. Set up data loaders with augmentation
3. Train the model in two phases:
   - **Warmup** (3 epochs): Train only the classification head
   - **Fine-tuning** (up to 50 epochs): Train the entire model
4. Generate comprehensive evaluation metrics and visualizations

### Configuration

Key hyperparameters can be adjusted in the configuration cell:

```python
BACKBONE_NAME = "efficientnet_b3"  # Model architecture
BATCH_SIZE = 256                   # Batch size
IMG_SIZE = 300                     # Input image size
WARMUP_EPOCHS = 3                  # Warmup phase epochs
FINETUNE_EPOCHS = 50               # Fine-tuning phase epochs
LR_HEAD = 5e-4                     # Learning rate for head
LR_BACKBONE = 1e-4                 # Learning rate for backbone
PATIENCE = 10                      # Early stopping patience
DROPOUT = 0.3                      # Dropout rate
```

### Checkpoints

Model checkpoints are saved to the `checkpoints/` directory:

- `warmup.pth`: Best model from warmup phase
- `finetune.pth`: Best model from fine-tuning phase

## Model Architecture

- **Backbone**: EfficientNet-B3 (pre-trained on ImageNet)
- **Classifier Head**: Dropout → Linear layer
- **Input Size**: 300×300 RGB images
- **Output**: 2 classes (benign, malignant)

## Training Strategy

1. **Warmup Phase**: Train only the classification head with frozen backbone
2. **Fine-tuning Phase**: Unfreeze backbone and train end-to-end with lower learning rate
3. **Optimization**: AdamW with cosine annealing learning rate schedule
4. **Loss Function**: Cross-entropy loss with optional focal loss and label smoothing

## Evaluation

The model is evaluated using:

- ROC-AUC
- PR-AUC (Precision-Recall)
- F1-Score
- Precision
- Recall
- Accuracy
- Confusion Matrix

Test-time augmentation (TTA) is applied for final predictions, averaging results from 7 augmented versions of each image.

## Results

The notebook generates visualizations including:

- Training progression (loss and accuracy curves)
- Validation metrics over epochs
- Confusion matrix
- ROC curve
- Precision-Recall curve

## Project Structure

```
4AL3 Final Report/
├── model.ipynb          # Main training notebook
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── checkpoints/        # Saved model checkpoints (generated during training)
```

## GPU Support

The notebook automatically detects and uses available CUDA GPUs. Multi-GPU support is configured with:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
```

Adjust this setting based on your hardware configuration.

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: [Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
- Pre-trained models: PyTorch/TorchVision
- Framework: PyTorch

📜 License & Usage Modification: Not permitted.

Redistribution: Only allowed with proper attribution and without any changes to the original files.

Commercial Use: Only with prior written consent.

📌 Attribution All credits for the creation, design, and development of this project go to:

Andre Menezes 📧 Contact: andremenezes231@hotmail.com 🌐 Website: https://andremenezes.dev

If this project is used, cited, or referenced in any form (including partial code, design elements, or documentation), you must provide clear and visible attribution to the original author(s).

⚠️ Disclaimer This project is provided without any warranty of any kind, either expressed or implied. Use at your own risk.

📂 File Integrity Do not alter, rename, or remove any files, directories, or documentation included in this project. Checksum or signature verification may be used to ensure file authenticity.

© 2025 Andre Menezes. All Rights Reserved.
