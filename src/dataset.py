import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATASET_PATH = "C:/Users/sarat/Documents/Python/DeepFakeDetection/data/images"


# ------------------------ DATASET CLASS ------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["path"]
        label = row["label"]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Always apply transforms
        img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)


# ------------------------ TRANSFORMS ------------------------
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(224, 224),  # REQUIRED for ViT
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ------------------------ DATALOADERS ------------------------
def create_dataloaders(train_df, val_df, test_df, batch_size=32):
    train_dataset = DeepfakeDataset(train_df, train=True, transform=get_transforms(train=True))
    val_dataset = DeepfakeDataset(val_df, train=False, transform=get_transforms(train=False))
    test_dataset = DeepfakeDataset(test_df, train=False, transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


# ------------------------ DATA HANDLING ------------------------
def build_dataframe():
    data = []
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        label = 0 if folder.lower() == "real" else 1
        method = folder
        for img in os.listdir(folder_path):
            data.append([os.path.join(folder_path, img), label, method])

    df = pd.DataFrame(data, columns=["path", "label", "method"])
    return df


def split_dataset(df, test_size=0.15, val_size=0.15):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df["label"], random_state=42)
    return train_df, val_df, test_df


# ------------------------ DEBUG TEST ------------------------
if __name__ == "__main__":
    df = build_dataframe()
    train_df, val_df, test_df = split_dataset(df)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df)

    for images, labels in train_loader:
        print("Batch images shape:", images.shape)
        print("Batch labels:", labels[:10])
        break
