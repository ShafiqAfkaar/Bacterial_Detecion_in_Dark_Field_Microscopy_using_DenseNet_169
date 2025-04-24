# 🧫 Bacteria Detection and enumeration using DenseNet169 

This project uses a customized **DenseNet-169** deep learning model for classifying microscope images of bacteria into **positive** and **negative** categories. The model is trained using high-resolution `.tif` image files and leverages transfer learning with PyTorch and torchvision.

---

## 📁 Dataset Structure

Dataset is structured as follows and contains **`.tif` images**:

dataset_split/ ├── train/ │ ├── positive/ │ └── negative/ ├── val/ │ ├── positive/ │ └── negative/ └── test/ ├── positive/ └── negative/




---

## ⚙️ Model Details

- **Architecture**: DenseNet-169 (pretrained on ImageNet)
- **Modified classifier**: Replaced to support binary classification (2 classes)
- **Loss function**: CrossEntropyLoss with class weights to address imbalance
- **Transforms**: Resize, rotation, color jitter, normalization
- **Format support**: `.tif` image loading is fully supported via `PIL.Image`

---
#Requirements
torch
torchvision
Pillow

