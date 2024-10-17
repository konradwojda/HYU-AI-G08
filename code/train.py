import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import DeepFakeDataset
import argparse


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


def prepare_model(train_path: str, test_path: str, model_name: str = "RESNET50"):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = DeepFakeDataset(root_dir=train_path, transform=transform)
    test_dataset = DeepFakeDataset(root_dir=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    match(model_name):
        case "RESNET50":
            model = get_resnet50()
        case "EFFICIENTNET":
            model = get_efficientnet()
        case _:
            model = get_resnet50()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, device, train_loader, test_loader, criterion, optimizer


def get_resnet50():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def get_efficientnet():
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def main(train_path: str, test_path: str, model_out: str, model_name: str):

    model, device, train_loader, test_loader, criterion, optimizer = (
        prepare_model(train_path, test_path, model_name)
    )

    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )

    torch.save(model.state_dict(), model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=False, default="data/train")
    parser.add_argument("--test_path", required=False, default="data/test")
    parser.add_argument(
        "--model_out", required=False, default="deepfake_detector_model.pth"
    )
    parser.add_argument("--model_name", default="RESTNET50")

    args = parser.parse_args()

    main(args.train_path, args.test_path, args.model_out, args.model_name)
