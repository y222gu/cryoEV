"""
Inference and Visualization Script for Cryo-EM Instance Segmentation

Features:
- Load trained models (UNet/FPN/DeepLabV3+ and YOLO)
- Run inference on test images
- Extract per-object confidence scores
- Interactive review of low-confidence detections
- Save prediction masks and review decisions
- Batch processing
"""

import os
import csv
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from scipy.ndimage import label as label_connected_components

from datasets.cryo_instance_dataset import CryoInstanceDataset, custom_collate_fn
from transforms.cryo_transforms import get_test_transforms
from training.train_unet import create_model


# ============================================================================
# Model Loading
# ============================================================================

def load_trained_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a trained UNet/FPN/DeepLabV3+ model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model_name = config['model_name']
    encoder_name = config['encoder_name']

    model = create_model(model_name, encoder_name, encoder_weights=None)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model: {model_name} with {encoder_name}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"Val IoU: {checkpoint['val_iou']:.4f}")

    return model


# ============================================================================
# Prediction Functions
# ============================================================================

@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on a batch of images.

    Returns:
        (probabilities, binary_predictions) both of shape (B, H, W)
    """
    images = images.to(device)
    logits = model(images)
    probs = torch.sigmoid(logits).squeeze(1)
    preds = (probs > threshold).float()
    return probs, preds


@torch.no_grad()
def predict_single_image(
    model: torch.nn.Module,
    image: np.ndarray,
    transform,
    device: torch.device,
    threshold: float = 0.5,
    original_size: Optional[Tuple[int, int]] = None,
    confidence_threshold: float = 0.0,
    accept_all: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single image.

    Args:
        model: Trained model
        image: Grayscale image (H, W)
        transform: Transform to apply
        device: Device
        threshold: Threshold for binary prediction
        original_size: If provided, resize output to this (H, W)
        confidence_threshold: Minimum confidence for automatic acceptance
        accept_all: If True, skip interactive review (backward compatible)

    Returns:
        (probability_map, binary_mask) both as numpy arrays
    """
    if transform is not None:
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
    else:
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0

    probs, preds = predict_batch(model, image_tensor, device, threshold)

    prob_map = probs[0].cpu().numpy()
    binary_mask = preds[0].cpu().numpy()

    if original_size is not None:
        h, w = original_size
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if not accept_all:
        instance_masks, confidences, bbox_info = extract_instances_unet(prob_map, binary_mask)
        if instance_masks:
            filtered_masks, filtered_confs, decisions = interactive_review_objects(
                image, instance_masks, confidences,
                threshold=confidence_threshold, accept_all=False
            )
            # Rebuild binary mask from accepted instances
            binary_mask = np.zeros_like(binary_mask)
            for m in filtered_masks:
                binary_mask[m] = 1.0

    return prob_map, binary_mask


# ============================================================================
# Instance Extraction
# ============================================================================

def extract_instances_unet(
    prob_map: np.ndarray,
    binary_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Extract individual object instances from a UNet probability map via
    connected-component analysis on the binary mask.

    Args:
        prob_map: Probability map (H, W), values in [0, 1]
        binary_mask: Binary prediction mask (H, W)

    Returns:
        instance_masks: List of boolean masks, one per object
        confidences: List of mean probability (confidence) per object
        bbox_info: List of dicts with bbox coordinates per object
    """
    labeled, n_objects = label_connected_components(binary_mask > 0.5)

    instance_masks: List[np.ndarray] = []
    confidences: List[float] = []
    bbox_info: List[Dict] = []

    for obj_id in range(1, n_objects + 1):
        obj_mask = labeled == obj_id
        coords = np.argwhere(obj_mask)
        if len(coords) == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        mean_prob = float(prob_map[obj_mask].mean())

        instance_masks.append(obj_mask)
        confidences.append(mean_prob)
        bbox_info.append({
            'x_min': int(x_min), 'y_min': int(y_min),
            'x_max': int(x_max), 'y_max': int(y_max),
            'area': int(obj_mask.sum()),
        })

    return instance_masks, confidences, bbox_info


def extract_instances_yolo(
    model_path: str,
    img_path: str,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = 'cpu'
) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Extract individual object instances from a YOLO segmentation model.

    Wraps the YOLO prediction pipeline and returns the same format as
    extract_instances_unet for a unified downstream workflow.

    Args:
        model_path: Path to YOLO .pt weights
        img_path: Path to input image
        imgsz: Inference image size
        conf: Confidence threshold for YOLO
        iou: IoU threshold for NMS
        device: Device string ('cpu' or 'cuda')

    Returns:
        instance_masks: List of boolean masks
        confidences: List of confidence scores
        bbox_info: List of dicts with bbox coordinates
    """
    from training.train_yolo import load_predictions_from_model

    pred_masks, pred_confidences = load_predictions_from_model(
        model_path, img_path, imgsz, conf, iou, device
    )

    instance_masks: List[np.ndarray] = []
    confidences: List[float] = []
    bbox_info: List[Dict] = []

    for mask, confidence in zip(pred_masks, pred_confidences):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        instance_masks.append(mask)
        confidences.append(confidence)
        bbox_info.append({
            'x_min': int(x_min), 'y_min': int(y_min),
            'x_max': int(x_max), 'y_max': int(y_max),
            'area': int(mask.sum()),
        })

    return instance_masks, confidences, bbox_info


# ============================================================================
# Interactive Review
# ============================================================================

def interactive_review_objects(
    image: np.ndarray,
    instance_masks: List[np.ndarray],
    confidences: List[float],
    threshold: float = 0.5,
    accept_all: bool = False
) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Interactively review detected objects.

    Objects with confidence >= threshold are automatically accepted.
    Objects below threshold are shown in a matplotlib popup for manual
    accept/reject (unless accept_all is True).

    Args:
        image: Original grayscale image (H, W)
        instance_masks: List of boolean masks per object
        confidences: List of confidence scores per object
        threshold: Confidence threshold for automatic acceptance
        accept_all: If True, accept everything without popups

    Returns:
        filtered_masks: Masks of accepted objects
        filtered_confs: Confidences of accepted objects
        decisions: List of dicts recording each decision
    """
    filtered_masks: List[np.ndarray] = []
    filtered_confs: List[float] = []
    decisions: List[Dict] = []

    for idx, (mask, conf) in enumerate(zip(instance_masks, confidences)):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        if accept_all or conf >= threshold:
            decision = 'accepted'
            filtered_masks.append(mask)
            filtered_confs.append(conf)
        else:
            decision = _show_review_popup(
                image, instance_masks, confidences, idx, threshold
            )
            if decision == 'accepted':
                filtered_masks.append(mask)
                filtered_confs.append(conf)

        decisions.append({
            'object_index': idx,
            'confidence': conf,
            'decision': decision,
            'bbox_x_min': int(x_min),
            'bbox_y_min': int(y_min),
            'bbox_x_max': int(x_max),
            'bbox_y_max': int(y_max),
        })

    return filtered_masks, filtered_confs, decisions


def _show_review_popup(
    image: np.ndarray,
    instance_masks: List[np.ndarray],
    confidences: List[float],
    current_idx: int,
    threshold: float
) -> str:
    """Show a matplotlib popup for reviewing a single object. Returns 'accepted' or 'rejected'."""
    result = {'decision': 'rejected'}

    fig, (ax_full, ax_crop) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Object {current_idx} | Confidence: {confidences[current_idx]:.3f} "
        f"(threshold: {threshold:.3f})",
        fontsize=13, fontweight='bold'
    )

    # --- Left panel: full image with all objects overlaid ---
    if len(image.shape) == 2:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        display_img = image.copy()
    overlay = display_img.copy()

    for i, (mask, conf) in enumerate(zip(instance_masks, confidences)):
        if i == current_idx:
            color = (255, 255, 0)  # yellow = under review
        elif conf >= threshold:
            color = (0, 200, 0)    # green = accepted
        else:
            color = (200, 200, 0)  # dim yellow = pending
        overlay[mask] = color

    blended = cv2.addWeighted(display_img, 0.6, overlay, 0.4, 0)
    ax_full.imshow(blended)
    ax_full.set_title("Full image (green=accepted, yellow=under review)")
    ax_full.axis('off')

    # --- Right panel: zoomed crop of current object ---
    current_mask = instance_masks[current_idx]
    coords = np.argwhere(current_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad = max(20, int(0.2 * max(y_max - y_min, x_max - x_min)))
    h, w = image.shape[:2]
    cy0 = max(0, y_min - pad)
    cy1 = min(h, y_max + pad)
    cx0 = max(0, x_min - pad)
    cx1 = min(w, x_max + pad)

    crop = blended[cy0:cy1, cx0:cx1]
    ax_crop.imshow(crop)
    ax_crop.set_title("Zoomed crop")
    ax_crop.axis('off')

    # --- Buttons ---
    ax_accept = fig.add_axes([0.3, 0.02, 0.15, 0.06])
    ax_reject = fig.add_axes([0.55, 0.02, 0.15, 0.06])
    btn_accept = Button(ax_accept, 'Accept', color='lightgreen', hovercolor='green')
    btn_reject = Button(ax_reject, 'Reject', color='lightsalmon', hovercolor='red')

    def on_accept(event):
        result['decision'] = 'accepted'
        plt.close(fig)

    def on_reject(event):
        result['decision'] = 'rejected'
        plt.close(fig)

    btn_accept.on_clicked(on_accept)
    btn_reject.on_clicked(on_reject)

    plt.show()
    return result['decision']


# ============================================================================
# Saving Decisions
# ============================================================================

def save_review_decisions(
    decisions: List[Dict],
    output_path: str
):
    """
    Save review decisions to a CSV file.

    Columns: object_index, confidence, decision,
             bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
    """
    fieldnames = [
        'object_index', 'confidence', 'decision',
        'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max',
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)
    print(f"Saved review decisions to {output_path}")


# ============================================================================
# High-Level Orchestrator
# ============================================================================

def predict_with_review(
    model_type: str,
    image_path: str,
    output_dir: str,
    # UNet params
    checkpoint_path: Optional[str] = None,
    image_size: int = 1024,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    # YOLO params
    yolo_model_path: Optional[str] = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    # Review params
    confidence_threshold: float = 0.5,
    accept_all: bool = True
) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    End-to-end: predict -> extract instances -> review -> save.

    Args:
        model_type: 'unet' or 'yolo'
        image_path: Path to input image
        output_dir: Directory to save outputs (masks, decisions CSV)
        checkpoint_path: Path to UNet .pth checkpoint (required if model_type='unet')
        image_size: Resize for UNet inference
        threshold: Binary threshold for UNet
        device: Torch device
        yolo_model_path: Path to YOLO .pt weights (required if model_type='yolo')
        yolo_conf: YOLO confidence threshold
        yolo_iou: YOLO NMS IoU threshold
        confidence_threshold: Objects below this go to interactive review
        accept_all: If True, accept all objects without popups

    Returns:
        (accepted_masks, accepted_confidences, decisions)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # --- Extract instances ---
    if model_type == 'unet':
        if checkpoint_path is None:
            raise ValueError("checkpoint_path required for model_type='unet'")
        model = load_trained_model(checkpoint_path, device)
        transform = get_test_transforms(image_size)
        h, w = image.shape
        prob_map, binary_mask = predict_single_image(
            model, image, transform, device, threshold,
            original_size=(h, w), accept_all=True  # extraction handled below
        )
        instance_masks, confidences, bbox_info = extract_instances_unet(prob_map, binary_mask)

    elif model_type == 'yolo':
        if yolo_model_path is None:
            raise ValueError("yolo_model_path required for model_type='yolo'")
        dev_str = 'cpu' if device.type == 'cpu' else str(device)
        instance_masks, confidences, bbox_info = extract_instances_yolo(
            yolo_model_path, image_path, image_size, yolo_conf, yolo_iou, dev_str
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'unet' or 'yolo'.")

    print(f"Detected {len(instance_masks)} objects")

    # --- Review ---
    filtered_masks, filtered_confs, decisions = interactive_review_objects(
        image, instance_masks, confidences,
        threshold=confidence_threshold, accept_all=accept_all
    )

    n_accepted = sum(1 for d in decisions if d['decision'] == 'accepted')
    n_rejected = sum(1 for d in decisions if d['decision'] == 'rejected')
    print(f"Accepted: {n_accepted}, Rejected: {n_rejected}")

    # --- Save outputs ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem

    # Save accepted binary mask
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for m in filtered_masks:
        combined_mask[m] = 255
    cv2.imwrite(str(output_path / f"{base_name}_mask.png"), combined_mask)

    # Save decisions CSV
    save_review_decisions(decisions, str(output_path / f"{base_name}_decisions.csv"))

    return filtered_masks, filtered_confs, decisions


# ============================================================================
# Visualization Helpers
# ============================================================================

def visualize_prediction(
    image: np.ndarray,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Prediction",
    save_path: Optional[str] = None
):
    """Visualize prediction results."""
    n_cols = 3 if ground_truth is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    im1 = axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(binary_mask, cmap='Reds', alpha=0.5)
    axes[2].set_title('Prediction Overlay')
    axes[2].axis('off')

    if ground_truth is not None:
        axes[3].imshow(image, cmap='gray')
        axes[3].imshow(ground_truth, cmap='Greens', alpha=0.5)
        axes[3].set_title('Ground Truth Overlay')
        axes[3].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def create_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """Create overlay of mask on image. Returns RGB overlay image (H, W, 3)."""
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0] = color

    overlay = cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)
    return overlay


# ============================================================================
# Batch Inference
# ============================================================================

def batch_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    threshold: float = 0.5,
    save_overlays: bool = True,
    save_masks: bool = True,
    visualize_every: int = 10,
    confidence_threshold: float = 0.0,
    accept_all: bool = True
):
    """
    Run inference on entire dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
        output_dir: Directory to save outputs
        threshold: Prediction threshold
        save_overlays: Whether to save overlay images
        save_masks: Whether to save binary masks
        visualize_every: Create visualization every N samples
        confidence_threshold: Minimum confidence for auto-accept
        accept_all: If True, skip interactive review (backward compatible)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_overlays:
        overlay_dir = output_path / 'overlays'
        overlay_dir.mkdir(exist_ok=True)

    if save_masks:
        mask_dir = output_path / 'masks'
        mask_dir.mkdir(exist_ok=True)

    viz_dir = output_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    model.eval()

    print(f"\nRunning inference on {len(data_loader.dataset)} samples...")
    print(f"Output directory: {output_dir}")

    for batch_idx, batch in enumerate(data_loader):
        images = batch['image'].to(device)
        sample_ids = batch['sample_id']

        has_gt = 'mask' in batch

        probs, preds = predict_batch(model, images, device, threshold)

        for i in range(len(sample_ids)):
            sample_id = sample_ids[i]
            base_name = Path(sample_id).stem

            prob_map = probs[i].cpu().numpy()
            binary_mask = preds[i].cpu().numpy()

            image = images[i].squeeze().cpu().numpy()
            image = (image * 255).astype(np.uint8)

            # Optional confidence-based review
            if not accept_all:
                inst_masks, confs, bbox_info = extract_instances_unet(prob_map, binary_mask)
                if inst_masks:
                    filtered, filt_confs, decisions = interactive_review_objects(
                        image, inst_masks, confs,
                        threshold=confidence_threshold, accept_all=False
                    )
                    binary_mask = np.zeros_like(binary_mask)
                    for m in filtered:
                        binary_mask[m] = 1.0
                    save_review_decisions(
                        decisions,
                        str(output_path / 'decisions' / f"{base_name}_decisions.csv")
                    )

            if save_masks:
                mask_path = mask_dir / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), (binary_mask * 255).astype(np.uint8))

            if save_overlays:
                overlay = create_overlay_image(image, binary_mask, color=(255, 0, 0), alpha=0.5)
                overlay_path = overlay_dir / f"{base_name}_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            sample_num = batch_idx * data_loader.batch_size + i
            if sample_num % visualize_every == 0:
                gt_mask = batch['mask'][i].cpu().numpy() if has_gt else None
                viz_path = viz_dir / f"{base_name}_viz.png"
                visualize_prediction(
                    image, prob_map, binary_mask,
                    ground_truth=gt_mask,
                    title=f"Sample: {sample_id}",
                    save_path=str(viz_path)
                )

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * data_loader.batch_size}/{len(data_loader.dataset)} samples")

    print(f"\nInference complete! Results saved to: {output_dir}")


def compare_models(
    model_paths: List[str],
    test_image_path: str,
    output_path: str,
    threshold: float = 0.5
):
    """Compare predictions from multiple models on the same image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    transform = get_test_transforms(image_size=1024)

    predictions = []
    model_names = []

    for model_path in model_paths:
        model = load_trained_model(model_path, device)
        prob_map, binary_mask = predict_single_image(
            model, image, transform, device, threshold, original_size=(h, w)
        )
        predictions.append((prob_map, binary_mask))
        model_names.append(Path(model_path).parent.name)
        print()

    n_models = len(model_paths)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for i, ((prob_map, binary_mask), model_name) in enumerate(zip(predictions, model_names)):
        col = i + 1
        im = axes[0, col].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        axes[0, col].set_title(f'{model_name}\nProbability', fontsize=10)
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)

        axes[1, col].imshow(image, cmap='gray')
        axes[1, col].imshow(binary_mask, cmap='Reds', alpha=0.5)
        axes[1, col].set_title(f'{model_name}\nPrediction', fontsize=10)
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nModel comparison saved to: {output_path}")
    plt.close()


# ============================================================================
# Main Script
# ============================================================================

if __name__ == '__main__':
    """
    Example usage for inference with confidence review.
    """

    CONFIG = {
        'model_path': r'C:\Users\Yifei\Documents\cryo\revised_annotation\results_instance_seg\unet_resnet34_combined\best_model_iou.pth',
        'test_data_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split\val',
        'output_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\inference_results',
        'image_size': 1024,
        'batch_size': 4,
        'threshold': 0.5,
        'visualize_every': 5,
        'confidence_threshold': 0.5,
        'accept_all': True,  # Set to False to enable interactive review
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("Loading model...")
    model = load_trained_model(CONFIG['model_path'], device)
    print()

    print("Loading test dataset...")
    test_transform = get_test_transforms(CONFIG['image_size'])
    test_dataset = CryoInstanceDataset(
        CONFIG['test_data_dir'],
        transform=test_transform,
        mode='train'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    batch_inference(
        model=model,
        data_loader=test_loader,
        device=device,
        output_dir=CONFIG['output_dir'],
        threshold=CONFIG['threshold'],
        save_overlays=True,
        save_masks=True,
        visualize_every=CONFIG['visualize_every'],
        confidence_threshold=CONFIG['confidence_threshold'],
        accept_all=CONFIG['accept_all'],
    )

    print("\nInference complete!")

    # Example: Single image with interactive review
    # predict_with_review(
    #     model_type='unet',
    #     image_path='test_image.png',
    #     output_dir='review_output',
    #     checkpoint_path='best_model_iou.pth',
    #     confidence_threshold=0.5,
    #     accept_all=False,  # enables popup review
    # )
