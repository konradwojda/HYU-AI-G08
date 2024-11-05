import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dataset import DeepFakeDataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from train import get_efficientnet, get_resnet50


def load_model(model_path, model_name="RESNET50"):
    if model_name == "EFFICIENTNET":
        model = get_efficientnet()
    else:
        model = get_resnet50()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model, device


def evaluate_and_visualize(model, test_loader, device, model_name, epochs):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    os.makedirs("evaluation_images", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name} ({epochs} epochs)")
    plt.savefig(f"evaluation_images/confusion_matrix_{model_name}_{epochs}_epochs.png")
    plt.close()


def main_eval(test_path, model_folder):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = DeepFakeDataset(root_dir=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_paths = glob.glob(f"{model_folder}/*.pth")
    print(model_paths)

    for model_path in model_paths:
        model_name = model_path.split("_")[-3]
        epochs = model_path.split("_")[-2]
        model, device = load_model(model_path, model_name=model_name)
        evaluate_and_visualize(model, test_loader, device, model_name, epochs)


if __name__ == "__main__":
    test_path = "data_test/Dataset/Test"
    model_folder = "."
    main_eval(test_path, model_folder)
