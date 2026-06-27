import pandas as pd
import os

def load_dataset():
    print("Loading dataset...")

    # Load both files
    fake_df = pd.read_csv("data/raw/Fake.csv")
    true_df = pd.read_csv("data/raw/True.csv")

    # Add labels
    fake_df['binary_label'] = 0  # Fake
    true_df['binary_label'] = 1  # Real

    # Merge
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Columns: {df.columns.tolist()}")
    print(f"Total records: {len(df)}")
    print(f"Fake: {len(fake_df)} | Real: {len(true_df)}")

    # Drop nulls
    df = df.dropna(subset=['text'])

    # Split 80/10/10
    train_df = df.sample(frac=0.8, random_state=42)
    remaining = df.drop(train_df.index)
    test_df = remaining.sample(frac=0.5, random_state=42)
    val_df = remaining.drop(test_df.index)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)} | Val: {len(val_df)}")
    print("Saved to data/processed/")

    return train_df, test_df, val_df

if __name__ == "__main__":
    load_dataset()