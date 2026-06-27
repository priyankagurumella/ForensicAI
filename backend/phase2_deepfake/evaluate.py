import torch
import torch.nn as nn
from torchvision import models
from preprocess import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate():
    print("Evaluating deepfake model...")

    _, test_loader, _, classes = get_dataloaders()

    # Load model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(torch.load(
        "backend/models/deepfake_model.pt",
        map_location='cpu'
    ))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds,
          target_names=['FAKE', 'REAL']))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['FAKE', 'REAL'],
                yticklabels=['FAKE', 'REAL'])
    plt.title('Deepfake Detection - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    os.makedirs("report", exist_ok=True)
    plt.savefig("report/deepfake_confusion_matrix.png")
    print("Confusion matrix saved to report/")
    plt.show()

if __name__ == "__main__":
    evaluate()