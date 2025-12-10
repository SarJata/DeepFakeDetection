import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.serialization as serialization
from pathlib import Path

from src.dataset import DeepfakeDataset, get_transforms
from src.models import get_model

serialization.add_safe_globals([pd.DataFrame])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model_name="efficientnet_b0", batch_size=32):
    print(f"Evaluating model: {model_name}")

    # Resolve paths safely
    ROOT_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = ROOT_DIR / "models"
    DATA_DIR = ROOT_DIR / "data"
    RESULTS_DIR = ROOT_DIR / "results"

    RESULTS_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / f"{model_name}_best.pth"
    test_split_path = DATA_DIR / "test_split.pt"

    # Load model
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load test data
    test_df = torch.load(test_split_path, weights_only=False)
    test_dataset = DeepfakeDataset(test_df, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(RESULTS_DIR / f"{model_name}_confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / f"{model_name}_roc_curve.png")
    plt.close()

    print(f"AUC: {roc_auc:.4f}")
    return acc, prec, rec, f1, roc_auc
