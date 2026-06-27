import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io
import os

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into last conv layer
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def apply_gradcam(image_bytes):
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

    grad_cam = GradCAM(model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad_()

    cam = grad_cam.generate(input_tensor)

    # Overlay on image
    img_array = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (224, 224))
    overlay = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)

    # Save
    os.makedirs("report", exist_ok=True)
    output_path = "report/gradcam_output.png"
    Image.fromarray(overlay).save(output_path)
    print(f"Grad-CAM saved to {output_path}")

    return output_path

if __name__ == "__main__":
    with open("data/raw/real_vs_fake/test/real/real_01031.jpg", "rb") as f:
        apply_gradcam(f.read())