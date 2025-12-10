# Hybrid Deepfake Image Detection Using CNN + Vision Transformer Ensemble

This repository contains the complete implementation of a deepfake-image detection system built using PyTorch.  
The project evaluates multiple modern deep-learning models and introduces an ensemble method that combines their outputs for stronger and more stable performance.

The codebase is structured for easy reproducibility and is suitable for both research and practical deployment.

---

## 1. Overview

Deepfake images are becoming increasingly realistic, making manual verification unreliable.  
This project focuses on building a robust image-level detector by training several complementary neural architectures and then merging their predictions using a lightweight meta-learner.

The system includes:

- A custom CNN baseline  
- EfficientNet-B0  
- Xception-41  
- ViT-Tiny  
- A stacking-based ensemble that fuses all three pretrained models  

The ensemble uses the output probabilities from the individual networks to make a final, more stable prediction.

---

## 2. Features

- Full training and evaluation pipeline (PyTorch)
- Hybrid ensemble model with logistic-regression meta classifier
- Automated metric computation:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Confusion matrix and ROC curve generation
- Preprocessing pipeline for the LDIC deepfake dataset
- Modular code layout (dataset, models, training, evaluation, ensemble)
- Ready for PyCharm development

---

## 3. Project Structure

```
project-root/
│
├── src/
│ ├── dataset.py # Dataset loader + augmentations
│ ├── models.py # CNN, EfficientNet, ViT, Xception definitions
│ ├── train.py # Training script for all base models
│ ├── evaluate.py # Metrics + confusion matrix + ROC
│ ├── ensemble.py # Stacking ensemble implementation
│ └── utils.py # Helper functions
│
├── models/ # Trained weights (ignored in Git)
│ ├── efficientnet_b0_best.pth
│ ├── vit_tiny_patch16_224_best.pth
│ ├── xception41_best.pth
│ └── ensemble_meta.pkl
│
├── results/ # Generated evaluation figures
│ ├── *_confusion_matrix.png
│ ├── *_roc_curve.png
│ └── ensemble_architecture_diagram.png
│
├── paper/ # LaTeX source of the report
│ └── deepfake_paper.tex
│
├── requirements.txt # Dependencies
├── main.py # Entry point for training/evaluation
└── README.md # Project documentation
```

---

## 4. Installation

1. Clone the repository:

```bash
git clone https://github.com/SarJata/DeepFakeDetection.git
cd DeepFakeDetection
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
```

3. Activate the environment:

Windows:

```cmd
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Dataset Setup
This project uses the Labeled Deepfake Image Collection (LDIC) from Kaggle.

Download it manually and place the files like this:

```
data/
   real/
   fake/
   metadata.csv
```
The dataset directory is excluded from Git to avoid large files in the repo.

## 6. Training Models
Run any of the supported models (Baseline CNN, EfficientNet-B0, ViT-Tiny, Xception-41):

```bash
python main.py --model efficientnet_b0 --epochs 20
```

Example:

```bash
python main.py --model vit_tiny_patch16_224 --epochs 10
```

## 7. Evaluating a Trained Model
```bash
python main.py --eval --model efficientnet_b0
```
This generates:

- Confusion matrix image
- ROC curve
- Metrics report in terminal

Outputs are saved under `results/`.

## 8. Running the Ensemble
Once all three base models are trained:

```bash
python src/ensemble.py
```
This script:

- Loads predictions from EfficientNet, ViT, and Xception
- Trains a logistic-regression meta learner
- Evaluates ensemble performance
- Stores `ensemble_meta.pkl`

## 9. Requirements
All dependencies are listed in `requirements.txt`.
Built and tested with:

- Python 3.10+
- PyTorch 2.x
- PyCharm (recommended for development)

## 10. Ethical Use Notice
This project is intended strictly for research and protective applications such as:

- Verifying media authenticity
- Preventing misuse of synthetic imagery
- Supporting digital forensic workflows

It must not be used to create or distribute deceptive media.

## 11. Future Extensions
Possible additions:

- Cross-dataset benchmarking
- Defense mechanisms against adversarial attacks
- Temporal modeling for video deepfakes
- Mobile/embedded deployment via model quantization
- Training with multimodal features (audio + vision)

## 12. Contact
For academic or project-related inquiries:

sarathjata@outlook.com
