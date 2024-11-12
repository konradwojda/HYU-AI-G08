# AI & Application Course Project (G08, Fall 2024)

This document contains explanation of AI-related code for this project.  
Whole code base was split into couple of files.

## dataset.py

In this file the `DeepFakeDataset` class is defined. It inherits from `torch.utils.data.Dataset` and is designed for loading images from a specified `root_dir`, organizing them into two categories: "Real" and "Fake". It assigns labels (0 for real images and 1 for fake images) based on the directory structure.

```py
class DeepFakeDataset(Dataset):
    """
    Class for loading and storing images from root_dir as a dataset.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, class_name in enumerate(["Real", "Fake"]):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(label)  # 0 for real, 1 for fake

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

## train.py

This is a script used for training models based on dataset. Let's look into it briefly:  

First, all of necessary imports:  
```py
import argparse
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import DeepFakeDataset
```

Next, the `train` function, which is used to train model for one epoch:

```py
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode to enable dropout and batch normalization updates
    running_loss = 0.0  # Track cumulative loss for averaging

    # Loop through each batch of images and labels in the training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the specified device (GPU or CPU)

        optimizer.zero_grad()  # Clear gradients from the previous iteration
        outputs = model(images)  # Forward pass: make predictions with the model
        loss = criterion(outputs, labels)  # Calculate loss between predicted and true labels
        loss.backward()  # Backward pass: calculate gradients of the loss
        optimizer.step()  # Update model weights based on gradients

        running_loss += loss.item()  # Accumulate loss to calculate the average

    return running_loss / len(train_loader)  # Return average loss for the epoch
```

Function to evaluate the model on the test set:
```py
def evaluate(model, test_loader, device):
    model.eval()  # Set model to evaluation mode to disable dropout and batch normalization updates
    y_true, y_pred = [], []  # Lists to hold true and predicted labels for metric calculation

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Get model predictions for the batch
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            y_true.extend(labels.cpu().numpy())  # Collect true labels
            y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

    # Calculate and return accuracy, precision, recall, and F1 score for performance evaluation
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1
```

Function to set up model, data loaders, loss function, and optimizer:
```py
def prepare_model(train_path: str, test_path: str, model_name: str = "RESNET50"):
    # Define transformations to preprocess images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet means and std devs
    ])

    # Load training and test datasets with transformations
    train_dataset = DeepFakeDataset(root_dir=train_path, transform=transform)
    test_dataset = DeepFakeDataset(root_dir=test_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle training data for randomness
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Don't shuffle test data

    # Select model based on model_name input
    match model_name:
        case "RESNET50":
            model = get_resnet50()
        case "EFFICIENTNET":
            model = get_efficientnet()
        case _:
            model = get_resnet50()

    # Choose device for model training: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)  # Move model to the selected device

    # Set the loss function to cross-entropy for binary classification
    criterion = nn.CrossEntropyLoss()
    # Initialize Adam optimizer for model parameter updates with a learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, device, train_loader, test_loader, criterion, optimizer


# Define and modify a ResNet50 model for binary classification
def get_resnet50():
    model = models.resnet50(pretrained=True)  # Load a pretrained ResNet50 model
    model.fc = nn.Linear(model.fc.in_features, 2)  # Modify output layer for binary classification
    return model


# Define and modify an EfficientNet model for binary classification
def get_efficientnet():
    model = models.efficientnet_b0(pretrained=True)  # Load a pretrained EfficientNet-B0 model
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Modify output layer for binary classification
    return model
```

Main function to execute the training and evaluation pipeline:
```py
def main(
    train_path: str,
    test_path: str,
    model_out: str,
    model_name: str,
    epochs: int,
    dump_csv: bool,
):
    # Prepare model, data loaders, loss function, and optimizer for training and evaluation
    model, device, train_loader, test_loader, criterion, optimizer = prepare_model(
        train_path, test_path, model_name
    )

    # Loop through the specified number of epochs for training
    for epoch in range(epochs):
        # Train model on training data and get the training loss for this epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # Evaluate model on test data to get performance metrics
        accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

        # Print results for this epoch
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )

        # Optionally save metrics to CSV
        if dump_csv:
            with open(f"metrics_{model_name}_{epochs}_epochs.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["epoch", "training_loss", "accuracy", "precision", "recall", "f1score"])
                writer.writerow([epoch + 1, train_loss, accuracy, precision, recall, f1])

    # Save the trained model state to a file
    torch.save(model.state_dict(), model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=False, default="data/train")  # Path to training data
    parser.add_argument("--test_path", required=False, default="data/test")  # Path to test data
    parser.add_argument("--model_out", required=False, default="deepfake_detector_model.pth")  # Output file for model
    parser.add_argument("--model_name", default="RESTNET50")  # Model choice (ResNet50 or EfficientNet)
    parser.add_argument("--epochs", type=int, default=5)  # Number of training epochs
    parser.add_argument("--csv_dump", action="store_true", default=False)  # Option to save metrics to CSV

    args = parser.parse_args()

    main(
        args.train_path,
        args.test_path,
        args.model_out,
        args.model_name,
        args.epochs,
        args.csv_dump,
    )
```

## predict.py

This script is used to predict image based on already trained model.

```py
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Make a prediction on a single image
def predict_image(image_path, model_path):
    # Define transformations to preprocess the input image to the format required by the model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize image to 224x224 pixels
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet means and std devs
        ]
    )

    # Determine the device to use: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a ResNet50 model architecture without pretrained weights
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Modify the output layer to have 2 classes (Real and Fake)

    # Load the model parameters from the specified file
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)  # Move the model to the chosen device
    model.eval()  # Set model to evaluation mode to disable dropout and batch normalization updates

    # Open and preprocess the input image
    image = Image.open(image_path).convert("RGB")  # Open the image and ensure it's in RGB format
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations, add batch dimension, and move to device

    # Make prediction with the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image)  # Forward pass through the model to get predictions
        _, predicted = torch.max(output, 1)  # Get the predicted class with the highest score
        return "Real" if predicted.item() == 0 else "Fake"  # Return label based on prediction


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model_path", default="deepfake_detector.pth", required=False)
    args = parser.parse_args()

    result = predict_image(args.image_path, args.model_path)
    print(f"The image is: {result}")
```

## eval.py

This file contains script used to evaluate trained models and create confusion matrices.

```py
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


# Function to load a specified model and prepare it for evaluation
def load_model(model_path, model_name="RESNET50"):
    if model_name == "EFFICIENTNET":
        model = get_efficientnet()
    else:
        model = get_resnet50()

    # Choose device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model weights from the specified file
    model.load_state_dict(torch.load(model_path))
    # Move model to the chosen device
    model = model.to(device)
    # Set model to evaluation mode (disables dropout and batch norm updates)
    model.eval()
    return model, device


# Function to evaluate the model on test data and visualize the confusion matrix
def evaluate_and_visualize(model, test_loader, device, model_name, epochs):
    y_true = []  
    y_pred = [] 

    # Inference without computing gradients
    with torch.no_grad():
        for images, labels in test_loader:
            # Move images and labels to the device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            # Get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Create directory for saving evaluation images if it doesn't exist
    os.makedirs("evaluation_images", exist_ok=True)

    # Compute the confusion matrix using true and predicted labels
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


# Main evaluation function to load each model and evaluate it on the test dataset
def main_eval(test_path, model_folder):
    # Define transformations to preprocess the input images for the model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize image to 224x224 pixels
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet means and std devs
        ]
    )

    # Load test dataset and create data loader for batching images during evaluation
    test_dataset = DeepFakeDataset(root_dir=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Find all model files in the specified folder with .pth extension
    model_paths = glob.glob(f"{model_folder}/*.pth")

    # Loop through each model file and evaluate it
    for model_path in model_paths:
        model_name = model_path.split("_")[-3]
        epochs = model_path.split("_")[-2]
        # Load the model and evaluate it, saving the confusion matrix as an image
        model, device = load_model(model_path, model_name=model_name)
        evaluate_and_visualize(model, test_loader, device, model_name, epochs)

if __name__ == "__main__":
    test_path = "data_test/Dataset/Test"
    model_folder = "."
    main_eval(test_path, model_folder)
```

## train_models.sh

This is simple bash script used to start training on virtual machine.

```bash
#!/bin/bash

# Check if KAGGLE_USERNAME and KAGGLE_KEY environment variables are set, needed for Kaggle API authentication
if [[ -z "${KAGGLE_USERNAME}" || -z "${KAGGLE_KEY}" ]]; then
  echo "Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set."
  exit 1
fi

# Create the Kaggle configuration directory if it doesn't already exist
mkdir -p ~/.kaggle

# Write Kaggle credentials to a JSON file needed by the Kaggle CLI for authentication
cat <<EOF > ~/.kaggle/kaggle.json
{
  "username": "${KAGGLE_USERNAME}",
  "key": "${KAGGLE_KEY}"
}
EOF

# Set file permissions to read/write only by the user
chmod 600 ~/.kaggle/kaggle.json

# Check if the Kaggle CLI tool is installed; if not, install it using pip
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Define dataset details and target download directory
DATASET_NAME="manjilkarki/deepfake-and-real-images"
DATASET_FILE="deepfake-and-real-images.zip"
DATA_DIR="data_test"

# Create the data directory if it doesn't already exist
mkdir -p $DATA_DIR

# Download the dataset from Kaggle into the specified directory
echo "Downloading dataset..."
kaggle datasets download -d ${DATASET_NAME} -p $DATA_DIR

# Unzip the downloaded dataset file into the data directory
echo "Unzipping dataset..."
unzip -q $DATA_DIR/${DATASET_FILE} -d $DATA_DIR

# Train models
for model in "RESNET50" "EFFICIENTNET"; do
    for i in 1 3 5 8; do
        python3 train.py --train_path ${DATA_DIR}/Dataset/Train \
                         --test_path ${DATA_DIR}/Dataset/Test \
                         --model_out model_${model}_${i}_epochs.pth \
                         --model_name ${model} \
                         --epochs ${i} \
                         --csv_dump
    done
done
```