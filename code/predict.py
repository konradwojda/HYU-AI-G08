import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def predict_image(image_path, model_path):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return "Real" if predicted.item() == 0 else "Fake"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model_path", default="deepfake_detector.pth", required=False)
    args = parser.parse_args()
    result = predict_image(args.image_path, args.model_path)
    print(f"The image is: {result}")
