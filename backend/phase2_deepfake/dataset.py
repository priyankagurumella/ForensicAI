import os
import shutil
import random
from pathlib import Path

def prepare_dataset():
    print("Preparing deepfake dataset...")

    real_dir = "data/raw/real_vs_fake/training_real"
    fake_dir = "data/raw/real_vs_fake/training_fake"

    # Output dirs
    splits = {
        "train": 0.8,
        "test": 0.1,
        "valid": 0.1
    }

    for split in splits:
        os.makedirs(f"data/raw/real_vs_fake/{split}/real", exist_ok=True)
        os.makedirs(f"data/raw/real_vs_fake/{split}/fake", exist_ok=True)

    # Get all images
    real_images = list(Path(real_dir).glob("*.jpg")) + \
                  list(Path(real_dir).glob("*.png"))
    fake_images = list(Path(fake_dir).glob("*.jpg")) + \
                  list(Path(fake_dir).glob("*.png"))

    print(f"Real images: {len(real_images)}")
    print(f"Fake images: {len(fake_images)}")

    # Limit to 5000 each for speed
    random.seed(42)
    real_images = random.sample(real_images, min(5000, len(real_images)))
    fake_images = random.sample(fake_images, min(5000, len(fake_images)))

    def split_and_copy(images, label):
        random.shuffle(images)
        n = len(images)
        train_end = int(n * 0.8)
        test_end = int(n * 0.9)

        sets = {
            "train": images[:train_end],
            "test": images[train_end:test_end],
            "valid": images[test_end:]
        }

        for split, imgs in sets.items():
            for img in imgs:
                dst = f"data/raw/real_vs_fake/{split}/{label}/{img.name}"
                shutil.copy(str(img), dst)
        
        print(f"{label} → train:{len(sets['train'])} test:{len(sets['test'])} valid:{len(sets['valid'])}")

    split_and_copy(real_images, "real")
    split_and_copy(fake_images, "fake")

    print("\nDataset prepared successfully!")

if __name__ == "__main__":
    prepare_dataset()