"""
Instance Segmentation Training Script for Cryo-EM Vesicles

Supports multiple architectures:
- UNet (ResNet34, ResNet50, EfficientNet-B0)
- FPN (ResNet34, ResNet50)
- DeepLabV3+ (ResNet34, ResNet50)

Features:
- Proper IoU calculation (accumulated across batches)
- Combined loss functions (Dice + BCE + Focal)
- Mixed precision training (AMP)
- Early stopping on validation loss
- Saves best model by both loss and IoU
- Comprehensive logging
- Learning rate scheduling
"""

import os
import json
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
import time
from pathlib import Path
from datasets.cryo_instance_dataset import CryoInstanceDataset, custom_collate_fn
from transforms.cryo_transforms import (
    get_train_transforms,
    get_val_transforms,
    get_heavy_train_transforms,
    get_light_train_transforms
)


def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Model Factory
# ============================================================================

def create_model(
    model_name: str,
    encoder_name: str,
    encoder_weights: str = 'imagenet',
    in_channels: int = 1,
    classes: int = 1
) -> nn.Module:
    """
    Create segmentation model.
    
    Args:
        model_name: 'unet', 'fpn', or 'deeplabv3plus'
        encoder_name: Encoder backbone (e.g., 'resnet34', 'resnet50', 'efficientnet-b0')
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels
        classes: Number of output classes
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    elif model_name == 'deeplabv3plus' or model_name == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'unet', 'fpn', or 'deeplabv3plus'")
    
    return model


# ============================================================================
# Loss Functions
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss for binary segmentation:
    - Dice Loss: Handles class imbalance
    - BCE Loss: Pixel-wise classification
    - Focal Loss (optional): Focuses on hard examples
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        focal_weight: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        if focal_weight > 0:
            self.focal_loss = smp.losses.FocalLoss(
                mode='binary',
                alpha=focal_alpha,
                gamma=focal_gamma
            )
        else:
            self.focal_loss = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        
        loss = self.dice_weight * dice + self.bce_weight * bce
        
        if self.focal_loss is not None and self.focal_weight > 0:
            focal = self.focal_loss(logits, targets)
            loss = loss + self.focal_weight * focal
        
        return loss


def get_loss_function(loss_type: str = 'combined') -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: 'dice', 'bce', 'focal', 'combined', 'combined_focal'
        
    Returns:
        Loss function
    """
    if loss_type == 'dice':
        return smp.losses.DiceLoss(mode='binary', from_logits=True)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        return smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
    elif loss_type == 'combined':
        return CombinedLoss(dice_weight=0.5, bce_weight=0.5, focal_weight=0.0)
    elif loss_type == 'combined_focal':
        return CombinedLoss(dice_weight=0.4, bce_weight=0.3, focal_weight=0.3)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# Metrics
# ============================================================================

def compute_iou_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Compute IoU and related metrics.
    
    Args:
        tp, fp, fn, tn: True/false positives/negatives (scalars)
        eps: Small epsilon for numerical stability
        
    Returns:
        Dictionary with metrics
    """
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Returns:
        (avg_loss, metrics_dict)
    """
    model.train()
    running_loss = 0.0
    
    # Accumulators for metrics
    tp_total = torch.tensor(0.0, device=device)
    fp_total = torch.tensor(0.0, device=device)
    fn_total = torch.tensor(0.0, device=device)
    tn_total = torch.tensor(0.0, device=device)
    
    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).unsqueeze(1)  # Add channel dim
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(device_type=device.type):
            logits = model(images)
            loss = loss_fn(logits, masks)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Compute metrics (on thresholded predictions)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            targets = masks.long()
            
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds,
                targets,
                mode='binary',
                threshold=None  # Already thresholded
            )
            
            tp_total += tp.sum()
            fp_total += fp.sum()
            fn_total += fn.sum()
            tn_total += tn.sum()
    
    avg_loss = running_loss / len(loader)
    metrics = compute_iou_metrics(tp_total, fp_total, fn_total, tn_total)
    
    return avg_loss, metrics


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Validate model.
    
    Returns:
        (avg_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0
    
    # Accumulators for metrics
    tp_total = torch.tensor(0.0, device=device)
    fp_total = torch.tensor(0.0, device=device)
    fn_total = torch.tensor(0.0, device=device)
    tn_total = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).unsqueeze(1)
            
            with autocast(device_type=device.type):
                logits = model(images)
                loss = loss_fn(logits, masks)
            
            running_loss += loss.item()
            
            # Compute metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            targets = masks.long()
            
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds,
                targets,
                mode='binary',
                threshold=None
            )
            
            tp_total += tp.sum()
            fp_total += fp.sum()
            fn_total += fn.sum()
            tn_total += tn.sum()
    
    avg_loss = running_loss / len(loader)
    metrics = compute_iou_metrics(tp_total, fp_total, fn_total, tn_total)
    
    return avg_loss, metrics


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    encoder_name: str,
    output_dir: str,
    experiment_name: str,
    loss_type: str = 'combined',
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    scheduler_type: str = 'plateau',
    device: Optional[torch.device] = None
) -> Dict:
    """
    Main training function with all the bells and whistles.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Model architecture name
        encoder_name: Encoder backbone name
        output_dir: Directory to save outputs
        experiment_name: Name for this experiment
        loss_type: Loss function type
        epochs: Maximum number of epochs
        lr: Initial learning rate
        weight_decay: Weight decay for optimizer
        patience: Early stopping patience
        scheduler_type: 'plateau' or 'cosine' or 'onecycle'
        device: Device to use (auto-detected if None)
        
    Returns:
        Dictionary with training history
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*70)
    print(f"Model: {model_name} with {encoder_name}")
    print(f"Loss: {loss_type}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Max epochs: {epochs}")
    print(f"Patience: {patience}")
    print("="*70 + "\n")
    
    # Create model
    model = create_model(model_name, encoder_name, encoder_weights='imagenet')
    model = model.to(device)
    
    # Loss function
    loss_fn = get_loss_function(loss_type)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-7
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    best_val_iou = -1.0
    early_stop_counter = 0
    
    # Paths for best models
    best_loss_path = output_path / 'best_model_loss.pth'
    best_iou_path = output_path / 'best_model_iou.pth'
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(current_lr)
        
        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            elif scheduler_type == 'onecycle':
                # OneCycleLR steps per batch, not per epoch
                pass
            else:
                scheduler.step()
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"Time: {epoch_time:.1f}s  "
            f"LR: {current_lr:.2e}  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Train IoU: {train_metrics['iou']:.4f}  "
            f"Val IoU: {val_metrics['iou']:.4f}  "
            f"Val F1: {val_metrics['f1']:.4f}"
        )
        
        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'config': {
                    'model_name': model_name,
                    'encoder_name': encoder_name,
                    'loss_type': loss_type
                }
            }, best_loss_path)
            print(f"  → Best model (loss) saved: {val_loss:.4f}")
        else:
            early_stop_counter += 1
        
        # Save best model by validation IoU
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'config': {
                    'model_name': model_name,
                    'encoder_name': encoder_name,
                    'loss_type': loss_type
                }
            }, best_iou_path)
            print(f"  → Best model (IoU) saved: {val_metrics['iou']:.4f}")
        
        # Early stopping
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Best validation IoU: {best_val_iou:.4f}")
            break
        
        # Periodic checkpoint
        if epoch % 25 == 0:
            checkpoint_path = output_path / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_metrics['iou']
            }, checkpoint_path)
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Models saved to: {output_path}")
    print("="*70 + "\n")
    
    # Save training history
    history_path = output_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


# ============================================================================
# Main Script
# ============================================================================

if __name__ == '__main__':
    # Configuration
    CONFIG = {
        'seed': 42,
        'data_dirs': {
            'train': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split\train',
            'val': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split\val'
        },
        'output_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\results_instance_seg',
        'image_size': 1024,
        'batch_size': 8,
        'num_workers': 0,
        'epochs': 300,
        'patience': 15,
        'base_lr': 5e-4,
        'weight_decay': 1e-4,
        'augmentation': 'light',  # 'light', 'standard', or 'heavy'
        'scheduler': 'plateau',  # 'plateau', 'cosine', or 'onecycle'
    }
    
    # Experiments to run
    EXPERIMENTS = [
        # Format: (model_name, encoder_name, loss_type)
        ('unet', 'resnet34', 'combined'),
        ('unet', 'resnet50', 'combined'),
        ('fpn', 'resnet34', 'combined'),
        ('fpn', 'resnet50', 'combined'),
        ('unet', 'efficientnet-b0', 'combined'),
    ]
    
    # Set seeds
    set_all_seeds(CONFIG['seed'])
    
    # Select augmentation
    if CONFIG['augmentation'] == 'light':
        train_transform = get_light_train_transforms(CONFIG['image_size'])
    elif CONFIG['augmentation'] == 'heavy':
        train_transform = get_heavy_train_transforms(CONFIG['image_size'])
    else:
        train_transform = get_train_transforms(CONFIG['image_size'])
    
    val_transform = get_val_transforms(CONFIG['image_size'])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CryoInstanceDataset(
        CONFIG['data_dirs']['train'],
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = CryoInstanceDataset(
        CONFIG['data_dirs']['val'],
        transform=val_transform,
        mode='train'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Run experiments
    results_summary = []
    
    for model_name, encoder_name, loss_type in EXPERIMENTS:
        experiment_name = f"{model_name}_{encoder_name}_{loss_type}"
        
        try:
            history = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model_name=model_name,
                encoder_name=encoder_name,
                output_dir=CONFIG['output_dir'],
                experiment_name=experiment_name,
                loss_type=loss_type,
                epochs=CONFIG['epochs'],
                lr=CONFIG['base_lr'],
                weight_decay=CONFIG['weight_decay'],
                patience=CONFIG['patience'],
                scheduler_type=CONFIG['scheduler']
            )
            
            # Record results
            best_val_loss = min(history['val_loss'])
            best_val_iou = max(history['val_iou'])
            
            results_summary.append({
                'experiment': experiment_name,
                'model': model_name,
                'encoder': encoder_name,
                'loss': loss_type,
                'best_val_loss': best_val_loss,
                'best_val_iou': best_val_iou
            })
            
        except Exception as e:
            print(f"\nERROR in experiment {experiment_name}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    for result in results_summary:
        print(f"{result['experiment']:50s} | Loss: {result['best_val_loss']:.4f} | IoU: {result['best_val_iou']:.4f}")
    print("="*70)
    
    # Save summary
    summary_path = Path(CONFIG['output_dir']) / 'experiments_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")