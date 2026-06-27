import torch
import torch.nn as nn
from torchvision import models
from preprocess import get_dataloaders
import os

def train_model(epochs=5):
    print("Training deepfake detection model...")

    train_loader, test_loader, valid_loader, classes = get_dataloaders()

    # Load EfficientNet
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Replace final layer for binary classification
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} → Loss: {running_loss:.3f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("backend/models", exist_ok=True)
            torch.save(model.state_dict(), "backend/models/deepfake_model.pt")
            print(f"  ✓ Best model saved ({val_acc:.2f}%)")

        scheduler.step()

    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    return model, device

if __name__ == "__main__":
    train_model(epochs=5)