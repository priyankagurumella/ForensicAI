import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    print("Setting up data loaders...")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(
        "data/raw/real_vs_fake/train", 
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        "data/raw/real_vs_fake/test",
        transform=test_transform
    )
    valid_dataset = datasets.ImageFolder(
        "data/raw/real_vs_fake/valid",
        transform=test_transform
    )

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Valid: {len(valid_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, test_loader, valid_loader, train_dataset.classes

if __name__ == "__main__":
    get_dataloaders()