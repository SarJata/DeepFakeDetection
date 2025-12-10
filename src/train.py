import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.models import get_model
from src.dataset import create_dataloaders, build_dataframe, split_dataset
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def train_model(model_name="efficientnet_b0", epochs=5, batch_size=32, lr=1e-4):
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = build_dataframe()
    train_df, val_df, test_df = split_dataset(df)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size)

    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=correct / total)

        # Validation phase
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(MODEL_DIR, exist_ok=True)
            save_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")
            print("Model saved!")

    print("Training completed.")


