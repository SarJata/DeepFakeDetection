import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import DeepfakeDataset, get_transforms
from src.models import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_probabilities(model_name, test_loader):
    model = get_model(model_name)
    model.load_state_dict(torch.load(f"models/{model_name}_best.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    probs = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Predicting: {model_name}", leave=False):
            images = images.to(DEVICE)
            outputs = model(images)
            prob = torch.softmax(outputs, dim=1)[:, 1]  # probability for fake class
            probs.extend(prob.cpu().numpy())

    return np.array(probs)


def evaluate_ensemble(test_df, val_df, batch_size=32):
    # Create datasets and loaders
    test_dataset = DeepfakeDataset(test_df, train=False, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = DeepfakeDataset(val_df, train=False, transform=get_transforms(train=False))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Generate probabilities from base models
    effnet_val = predict_probabilities("efficientnet_b0", val_loader)
    vit_val = predict_probabilities("vit_tiny_patch16_224", val_loader)
    xcept_val = predict_probabilities("xception41", val_loader)

    effnet_test = predict_probabilities("efficientnet_b0", test_loader)
    vit_test = predict_probabilities("vit_tiny_patch16_224", test_loader)
    xcept_test = predict_probabilities("xception41", test_loader)

    # Train meta-classifier
    val_labels = np.concatenate([labels.numpy() for _, labels in val_loader])
    val_stack = np.vstack([effnet_val, vit_val, xcept_val]).T

    meta = LogisticRegression()
    meta.fit(val_stack, val_labels)

    # Test evaluation
    test_labels = np.concatenate([labels.numpy() for _, labels in test_loader])
    test_stack = np.vstack([effnet_test, vit_test, xcept_test]).T

    preds = meta.predict(test_stack)
    probs = meta.predict_proba(test_stack)[:, 1]

    acc = accuracy_score(test_labels, preds)
    prec = precision_score(test_labels, preds)
    rec = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    fpr, tpr, _ = roc_curve(test_labels, probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(test_labels, preds)

    print("\n--- Ensemble Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save ensemble meta model
    os.makedirs("models", exist_ok=True)
    with open("models/ensemble_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # --------- Save Confusion Matrix Plot ----------
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/ensemble_confusion_matrix.png")
    plt.close()

    # --------- Save ROC Curve Plot ----------
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Ensemble ROC Curve (AUC = {roc_auc:.4f})")
    plt.tight_layout()
    plt.savefig("results/ensemble_roc_curve.png")
    plt.close()

    return acc, prec, rec, f1, roc_auc, cm
