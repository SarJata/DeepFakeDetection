import torch
from src.train import train_model
from src.evaluate import evaluate
from src.ensemble import evaluate_ensemble
import pandas as pd

# De-Comment the Model you are Testing
# MODEL_NAME = "vit_tiny_patch16_224"
# MODEL_NAME = "efficientnet_b0"
# MODEL_NAME = "baseline"
# MODEL_NAME = "xception41"
MODEL_NAME = "ensemble"

def load_splits():
    train_df = torch.load("data/train_split.pt", weights_only=False)
    val_df   = torch.load("data/val_split.pt", weights_only=False)
    test_df  = torch.load("data/test_split.pt", weights_only=False)
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()

    # ---- Training Handling ----
    if MODEL_NAME != "ensemble":  # Ensemble does not train
        print(f"\nTraining model: {MODEL_NAME}")
        train_model(model_name=MODEL_NAME, epochs=5, batch_size=32, lr=1e-4)

    # ---- Evaluation Handling ----
    print(f"\nEvaluating model: {MODEL_NAME}")
    if MODEL_NAME == "ensemble":
        evaluate_ensemble(test_df=test_df, val_df=val_df, batch_size=32)
    else:
        evaluate(model_name=MODEL_NAME, batch_size=32)
