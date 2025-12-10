import torch
from dataset import build_dataframe, split_dataset

def main():
    print("Building dataframe from image folders...")
    df = build_dataframe()
    print("Total images found:", len(df))

    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("Saving dataset splits to disk...")
    torch.save(train_df, "../data/train_split.pt")
    torch.save(val_df, "../data/val_split.pt")
    torch.save(test_df, "../data/test_split.pt")

    print("Done! Splits saved successfully.")

if __name__ == "__main__":
    main()
