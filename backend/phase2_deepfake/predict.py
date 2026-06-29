import io
from PIL import Image

# Global model variable - loads only when needed
model = None

def load_model():
    global model
    if model is None:
        import torch
        import torch.nn as nn
        from torchvision import models

        print("Loading EfficientNet model...")
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(
            m.classifier[1].in_features, 2
        )
        m.load_state_dict(torch.load(
       "models/deepfake_model.pt",
       map_location='cpu'
    ))
        m.eval()
        model = m
        print("Model loaded successfully!")
    return model

def predict_image(image_bytes):
    import torch
    from torchvision import transforms

    # Load model only when prediction is needed
    m = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = m(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    label = "REAL" if predicted.item() == 1 else "FAKE"
    conf_pct = round(confidence.item() * 100, 2)

    return {
        "prediction": label,
        "confidence": conf_pct,
        "fake_prob": round(probs[0][0].item() * 100, 2),
        "real_prob": round(probs[0][1].item() * 100, 2)
    }

if __name__ == "__main__":
    with open("data/raw/real_vs_fake/test/real/real_01031.jpg", "rb") as f:
        result = predict_image(f.read())
    print(f"Prediction: {result['prediction']} | Confidence: {result['confidence']}%")