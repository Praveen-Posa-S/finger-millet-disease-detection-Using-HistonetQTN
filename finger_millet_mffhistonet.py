import argparse
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from skimage import color
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TEXTURE_FEATURE_DIM = 6 + 26 + 6  # GLCM + LBP + Gabor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_glcm_features(image: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(image)
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = [
        graycoprops(glcm, prop).ravel()
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    ]
    return np.concatenate(features)


def extract_lbp_features(image: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
    gray = color.rgb2gray(image)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


def extract_gabor_features(image: np.ndarray, frequencies: Sequence[float] = (0.1, 0.5, 0.9), theta: float = 0) -> np.ndarray:
    gray = color.rgb2gray(image)
    features = []
    for frequency in frequencies:
        kernel = cv2.getGaborKernel((21, 21), 5, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features.extend([filtered.mean(), filtered.var()])
    return np.array(features)


class FingerMilletDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class DenseNet121Base(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.base_model = models.densenet121(weights=weights)
        except AttributeError:
            self.base_model = models.densenet121(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class QTN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.tensor_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.tensor_network(x)
        x = self.fc2(x)
        return x


class MFFHistoNet(nn.Module):
    def __init__(self, cnn_model: DenseNet121Base, qtn_model: QTN, num_classes: int):
        super().__init__()
        self.cnn_model = cnn_model
        self.qtn_model = qtn_model
        cnn_dim = cnn_model.base_model.classifier.out_features
        qtn_dim = qtn_model.fc2.out_features
        self.fc_texture = nn.Linear(TEXTURE_FEATURE_DIM, 128)
        self.fc_fusion = nn.Linear(cnn_dim + qtn_dim + 128, 256)
        self.fc_final = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cnn_features = self.cnn_model(x)
        qtn_features = self.qtn_model(x.view(batch_size, -1))
        texture_features = self._compute_texture_batch(x)
        combined = torch.cat((cnn_features, qtn_features, texture_features), dim=1)
        fused = torch.relu(self.fc_fusion(combined))
        return self.fc_final(fused)

    def _compute_texture_batch(self, x: torch.Tensor) -> torch.Tensor:
        imgs = x.detach().cpu().numpy()
        texture_vectors: List[np.ndarray] = []
        for img in imgs:
            img = self._denormalize(img)
            glcm = extract_glcm_features(img)
            lbp = extract_lbp_features(img)
            gabor = extract_gabor_features(img)
            texture_vectors.append(np.concatenate([glcm, lbp, gabor]))
        texture = torch.tensor(np.stack(texture_vectors), dtype=torch.float32, device=x.device)
        return torch.relu(self.fc_texture(texture))

    @staticmethod
    def _denormalize(img: np.ndarray) -> np.ndarray:
        for c in range(3):
            img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
        img = np.clip(img, 0.0, 1.0)
        return np.transpose(img, (1, 2, 0))


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".avif"}


def is_image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except UnidentifiedImageError:
        print(f"Skipping unreadable image: {path}")
        return False
    except (OSError, ValueError):
        print(f"Skipping corrupted image: {path}")
        return False


def collect_samples(data_dir: str, class_names: Sequence[str]) -> List[Tuple[str, int]]:
    samples = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(data_dir) / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in class_dir.glob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and is_image_readable(path):
                samples.append((str(path), class_idx))
    if not samples:
        raise RuntimeError(f"No images found in {data_dir}")
    return samples


def train_val_split(
    samples: Sequence[Tuple[str, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    return train_samples, val_samples


def build_dataloaders(
    train_samples: Sequence[Tuple[str, int]],
    val_samples: Sequence[Tuple[str, int]],
    image_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_dataset = FingerMilletDataset(train_samples, transform=transform)
    val_dataset = FingerMilletDataset(val_samples, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    num_epochs: int,
):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_acc = 100.0 * correct / max(1, total)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"Train Loss {epoch_loss:.4f} Acc {epoch_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}%"
        )
    return history


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    avg_loss = total_loss / max(1, len(loader))
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


def evaluate_model(model: nn.Module, loader: DataLoader, class_names: Sequence[str], device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_history(history: dict):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Finger Millet Disease Detection with MFFHistoNet")
    parser.add_argument("--data_dir", type=str, default="ragi dataset", help="Root directory containing class subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="mff_histonet_millet.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["downy", "healthy", "mottle", "seedling", "smut", "wilt"]

    print("Loading dataset...")
    samples = collect_samples(args.data_dir, class_names)
    train_samples, val_samples = train_val_split(samples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    train_loader, val_loader = build_dataloaders(
        train_samples,
        val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    cnn_model = DenseNet121Base(num_classes=len(class_names))
    qtn_model = QTN(input_dim=3 * args.image_size * args.image_size, hidden_dim=512, output_dim=128)
    model = MFFHistoNet(cnn_model, qtn_model, num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
    )

    plot_history(history)
    evaluate_model(model, val_loader, class_names, device)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")
    


if __name__ == "__main__":
    main()

