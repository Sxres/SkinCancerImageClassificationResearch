"""
Shared utilities for skin cancer image classification experiments.

This module provides common constants, datasets, transforms, loss functions,
and evaluation utilities used across different model training notebooks.
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================================
# Constants
# ============================================================================

# Binary label mapping: cancer_type -> 0 (Benign) or 1 (Malignant)
BINARY_MAP: Dict[str, int] = {
    # Malignant (label = 1)
    "Melanoma": 1,
    "BCC": 1,
    "SCC": 1,
    "Actinic_Keratosis": 1,
    "Malignant_Other": 1,
    # Benign (label = 0)
    "Melanocytic_Nevus": 0,
    "Seborrheic_Keratosis": 0,
    "Dermatofibroma": 0,
    "Hemangioma": 0,
    "Fibrous_Papule": 0,
    "Other_Benign": 0,
}

# Human-readable class names indexed by label
CLASS_NAMES = ["Benign", "Malignant"]

# ImageNet normalization statistics (used by pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default target for clinical sensitivity threshold optimization
MIN_SENSITIVITY_TARGET = 0.95


# ============================================================================
# Dataset
# ============================================================================

class BinaryDermoscopeSkinDataset(Dataset):
    """
    PyTorch Dataset for dermoscope skin lesion images with binary labels.

    Reads image filenames from a manifest CSV and loads images directly from
    the specified images directory. Supports multiple dermoscope images per
    instance (semicolon-separated in 'dscope_files' column).

    Args:
        manifest_df: DataFrame with 'binary_label' and 'dscope_files' columns.
        images_dir: Path to directory containing the image files.
        transform: Optional torchvision transform to apply to images.

    Example:
        >>> manifest = pd.read_csv("instances.csv")
        >>> manifest["binary_label"] = manifest["cancer_type"].map(BINARY_MAP)
        >>> dataset = BinaryDermoscopeSkinDataset(manifest, "images/", transform)
    """

    def __init__(self, manifest_df, images_dir: str, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        for _, row in manifest_df.iterrows():
            label = row["binary_label"]
            for filename in str(row["dscope_files"]).split(";"):
                filepath = os.path.join(images_dir, filename)
                self.samples.append((filepath, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
# Transforms
# ============================================================================

def get_transforms(
    image_size: int = 224,
    resize_size: int = 256
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and evaluation transforms for dermoscope images.

    Training transforms include data augmentation (random crop, flips,
    rotation, color jitter, affine transforms). Evaluation transforms
    use deterministic center crop.

    Both transforms apply ImageNet normalization for compatibility with
    pretrained models (DINOv2, ResNet, etc.).

    Args:
        image_size: Final image size after cropping (default: 224).
        resize_size: Size to resize images before cropping (default: 256).

    Returns:
        Tuple of (train_transform, eval_transform).

    Example:
        >>> train_tf, eval_tf = get_transforms(image_size=224, resize_size=256)
        >>> train_dataset = BinaryDermoscopeSkinDataset(df, "images/", train_tf)
        >>> val_dataset = BinaryDermoscopeSkinDataset(df, "images/", eval_tf)
    """
    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


# ============================================================================
# Loss Functions
# ============================================================================

class BinaryFocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with optional label smoothing for binary classification.

    Focal loss down-weights easy examples and focuses training on hard
    negatives, which is useful for imbalanced datasets. Label smoothing
    regularizes the model by softening the target distribution.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values increase focus on hard
            examples. Default: 2.0.
        pos_weight: Weight for positive class to handle class imbalance.
        label_smoothing: Amount of smoothing (0 = no smoothing). Default: 0.0.
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'.

    Example:
        >>> criterion = BinaryFocalLossWithSmoothing(gamma=2.0, label_smoothing=0.05)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================================
# Mixup Augmentation
# ============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup augmentation to a batch of images and labels.

    Mixup creates virtual training examples by linearly interpolating
    random pairs of samples and their labels, which regularizes the
    model and improves generalization.

    Args:
        x: Batch of images, shape (N, C, H, W).
        y: Batch of labels, shape (N,).
        alpha: Mixup interpolation strength. If alpha > 0, lambda is
            drawn from Beta(alpha, alpha). If alpha <= 0, no mixing
            is applied (lambda = 1). Default: 0.2.

    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda) where:
            - mixed_x: Mixed images
            - y_a: Original labels
            - y_b: Shuffled labels
            - lambda: Mixing coefficient

    Example:
        >>> mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
        >>> outputs = model(mixed_x)
        >>> loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute loss for mixup-augmented predictions.

    The loss is a weighted combination of losses computed against
    both original (y_a) and shuffled (y_b) labels.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss, FocalLoss).
        pred: Model predictions (logits).
        y_a: Original labels.
        y_b: Shuffled labels from mixup.
        lam: Mixing coefficient from mixup_data().

    Returns:
        Combined loss value.

    Example:
        >>> loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Evaluation Utilities
# ============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_sensitivity: float = 0.95
) -> Tuple[float, float, float]:
    """
    Find the optimal classification threshold that maximizes specificity
    while maintaining a minimum sensitivity level.

    This is clinically important for cancer screening where high
    sensitivity (low false negative rate) is critical to avoid
    missing malignant cases.

    Args:
        y_true: Ground truth binary labels, shape (N,).
        y_probs: Predicted probabilities for positive class, shape (N,).
        min_sensitivity: Minimum required sensitivity (recall for positive
            class). Default: 0.95.

    Returns:
        Tuple of (optimal_threshold, achieved_sensitivity, achieved_specificity).
        If min_sensitivity cannot be achieved, returns the threshold that
        maximizes sensitivity.

    Example:
        >>> thresh, sens, spec = find_optimal_threshold(y_true, y_probs, 0.95)
        >>> print(f"Threshold: {thresh:.3f}, Sens: {sens:.3f}, Spec: {spec:.3f}")
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_sens = 0.0
    best_spec = 0.0

    # First pass: find threshold meeting sensitivity constraint with best specificity
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        if sensitivity >= min_sensitivity:
            if specificity > best_spec:
                best_threshold = thresh
                best_sens = sensitivity
                best_spec = specificity

    # Fallback: if min_sensitivity not achievable, maximize sensitivity
    if best_sens < min_sensitivity:
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            if sensitivity > best_sens:
                best_threshold = thresh
                best_sens = sensitivity
                best_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return best_threshold, best_sens, best_spec


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for binary predictions.

    Args:
        y_true: Ground truth binary labels, shape (N,).
        y_probs: Predicted probabilities for positive class, shape (N,).
        threshold: Classification threshold. Default: 0.5.

    Returns:
        Dictionary containing:
            - accuracy: Overall classification accuracy
            - sensitivity: True positive rate (recall for positive class)
            - specificity: True negative rate
            - f1: F1 score (harmonic mean of precision and recall)
            - auc_roc: Area under the ROC curve

    Example:
        >>> metrics = compute_metrics(y_true, y_probs, threshold=0.5)
        >>> print(f"AUC: {metrics['auc_roc']:.4f}, Sens: {metrics['sensitivity']:.4f}")
    """
    y_pred = (y_probs >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC requires probability scores
    try:
        auc_roc = roc_auc_score(y_true, y_probs)
    except ValueError:
        # Handle edge case where only one class is present
        auc_roc = 0.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc_roc": auc_roc,
    }
