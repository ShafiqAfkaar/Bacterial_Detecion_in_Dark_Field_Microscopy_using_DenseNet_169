# Bacterial Detection in Dark Field Microscopy using DenseNet-169

A deep learning project for binary classification of bacteria images in dark field microscopy using DenseNet169 with transfer learning.

> **Note**: This repository has been restructured for better organization. The original notebook used in the published paper is available in [`archive/`](archive/) folder.

## 📁 Project Structure

Bacterial_Detecion_in_Dark_Field_Microscopy_using_DenseNet_169/
├── archive/ # Original notebook (as in paper)
│ └── Bacterial_Detection_in_Dark_Field_Microscopy_using_DenseNet_169.ipynb
├── models/ # Model architectures
│ ├── init.py
│ └── densenet_model.py
├── utils/ # Data loading and utilities
│ ├── init.py
│ ├── data_loader.py
│ ├── transforms.py
│ └── visualization.py
├── training/ # Training scripts
│ ├── init.py
│ ├── train.py
│ └── config.py
├── evaluation/ # Evaluation scripts
│ ├── init.py
│ └── evaluate.py
├── requirements.txt
├── README.md
└── main.py

text

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ShafiqAfkaar/Bacterial_Detecion_in_Dark_Field_Microscopy_using_DenseNet_169.git

cd Bacterial_Detecion_in_Dark_Field_Microscopy_using_DenseNet_169

pip install -r requirements.txt

```

## Training

```bash
python main.py --mode train

```

## Evaluation

```bash
python main.py --mode evaluate

```

## Visualization

```bash
python main.py --mode visualize

```

## 📊 Results

-Test Accuracy: 100%

-Precision/Recall: 1.00 for both classes

-Model: DenseNet169 with transfer learning

-Input Size: 96x96 RGB images

## 🔧 Features

-Modular code structure

-Data augmentation

-Comprehensive evaluation metrics

-Visualization tools

-Class imbalance handling

## 📦 Dependencies

-PyTorch >= 1.9.0

-Torchvision >= 0.10.0

-Pillow >= 8.0.0

-Matplotlib >= 3.3.0

-Scikit-learn >= 0.24.0

-NumPy >= 1.19.0

## 🏗️ Model Architecture

-Backbone: DenseNet169 (pretrained on ImageNet)

-Classifier: Linear layer with 2 output units

-Input size: 96x96 RGB images

-Loss function: CrossEntropyLoss with class weights

## 📈 Performance

-Metric Negative Class Positive Class
-Precision 1.00 1.00
-Recall 1.00 1.00
-F1-Score 1.00 1.00

## 🔍 Dataset

-Binary classification: Negative vs Positive bacteria

-Training/Validation/Test split provided

-Data augmentation applied during training

## 🎯 Usage

-For Reproduction (Original Paper Code):
-The exact code used in the published paper is available in the archive/ folder.

## For Development (New Modular Code):

```bash
# Training with custom data path
python main.py --mode train --data_path path/to/your/dataset

# Evaluation with custom model path
python main.py --mode evaluate --model_path models/densenet169_2class.pth

# Visualization
python main.py --mode visualize

```

### 📝 Citation

If you use this code in your research, please cite the original paper.

```bash
(Zhu H, Rahman SU, Pan T, Yuan J, Zhou X. Dark-field intelligent detection of V. parahaemolyticus using T4 bacteriophage displaying tail spike proteins and gold nanoparticles. Biosens Bioelectron. 2025 Nov 15;288:117830. doi: 10.1016/j.bios.2025.117830. Epub 2025 Jul 28. PMID: 40749397.)
```

## 👥 Contributing

-Fork the repository

-Create a feature branch

-Commit your changes

-Push to the branch

-Create a Pull Request

## 📄 License

This project is available for academic and research purposes.
