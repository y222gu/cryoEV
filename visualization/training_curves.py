"""
Visualization Script for Training Metrics

Features:
- Plot training/validation curves
- Compare multiple experiments
- Generate summary reports
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def plot_training_curves(
    history_path: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Plot training and validation curves from history JSON.
    
    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure (if None, displays plot)
        title: Title for the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU curves
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train IoU', linewidth=2)
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Val IoU', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('IoU Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # F1 curves
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        experiment_name = Path(history_path).parent.name
        fig.suptitle(f'Training Curves: {experiment_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(epochs)}")
    print(f"Best train loss: {min(history['train_loss']):.4f} (epoch {np.argmin(history['train_loss'])+1})")
    print(f"Best val loss: {min(history['val_loss']):.4f} (epoch {np.argmin(history['val_loss'])+1})")
    print(f"Best train IoU: {max(history['train_iou']):.4f} (epoch {np.argmax(history['train_iou'])+1})")
    print(f"Best val IoU: {max(history['val_iou']):.4f} (epoch {np.argmax(history['val_iou'])+1})")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final train IoU: {history['train_iou'][-1]:.4f}")
    print(f"Final val IoU: {history['val_iou'][-1]:.4f}")
    print("="*60 + "\n")


def compare_experiments(
    experiment_dirs: List[str],
    experiment_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Compare multiple experiments side by side.
    
    Args:
        experiment_dirs: List of paths to experiment directories
        experiment_names: Optional custom names for experiments
        save_path: Path to save comparison figure
    """
    if experiment_names is None:
        experiment_names = [Path(d).name for d in experiment_dirs]
    
    # Load all histories
    histories = []
    for exp_dir in experiment_dirs:
        history_path = Path(exp_dir) / 'training_history.json'
        if not history_path.exists():
            print(f"Warning: History not found for {exp_dir}")
            continue
        
        with open(history_path, 'r') as f:
            histories.append(json.load(f))
    
    if len(histories) == 0:
        print("Error: No valid experiment histories found")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, name) in enumerate(zip(histories, experiment_names[:len(histories)])):
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors[i]
        
        # Validation loss
        axes[0, 0].plot(epochs, history['val_loss'], color=color, label=name, linewidth=2)
        
        # Validation IoU
        axes[0, 1].plot(epochs, history['val_iou'], color=color, label=name, linewidth=2)
        
        # Training loss
        axes[1, 0].plot(epochs, history['train_loss'], color=color, label=name, linewidth=2, alpha=0.7)
        
        # Training IoU
        axes[1, 1].plot(epochs, history['train_iou'], color=color, label=name, linewidth=2, alpha=0.7)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('Training IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print comparison table
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON")
    print("="*100)
    print(f"{'Experiment':<40} {'Best Val Loss':<15} {'Best Val IoU':<15} {'Final Val IoU':<15}")
    print("-"*100)
    
    for history, name in zip(histories, experiment_names[:len(histories)]):
        best_loss = min(history['val_loss'])
        best_iou = max(history['val_iou'])
        final_iou = history['val_iou'][-1]
        print(f"{name:<40} {best_loss:<15.4f} {best_iou:<15.4f} {final_iou:<15.4f}")
    
    print("="*100 + "\n")


def plot_metric_summary(
    experiment_dirs: List[str],
    experiment_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Create bar chart comparing final metrics across experiments.
    
    Args:
        experiment_dirs: List of paths to experiment directories
        experiment_names: Optional custom names for experiments
        save_path: Path to save figure
    """
    if experiment_names is None:
        experiment_names = [Path(d).name for d in experiment_dirs]
    
    # Load final metrics
    val_losses = []
    val_ious = []
    val_f1s = []
    
    for exp_dir in experiment_dirs:
        history_path = Path(exp_dir) / 'training_history.json'
        if not history_path.exists():
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        val_losses.append(min(history['val_loss']))
        val_ious.append(max(history['val_iou']))
        val_f1s.append(max(history['val_f1']))
    
    if len(val_losses) == 0:
        print("Error: No valid experiments found")
        return
    
    # Truncate names if too long
    display_names = [name[:30] + '...' if len(name) > 30 else name 
                     for name in experiment_names[:len(val_losses)]]
    
    # Create bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(display_names))
    width = 0.6
    
    # Validation loss (lower is better)
    axes[0].bar(x, val_losses, width, color='steelblue')
    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Best Validation Loss (Lower is Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(val_losses):
        axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Validation IoU (higher is better)
    axes[1].bar(x, val_ious, width, color='forestgreen')
    axes[1].set_xlabel('Experiment')
    axes[1].set_ylabel('Validation IoU')
    axes[1].set_title('Best Validation IoU (Higher is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(val_ious):
        axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Validation F1 (higher is better)
    axes[2].bar(x, val_f1s, width, color='coral')
    axes[2].set_xlabel('Experiment')
    axes[2].set_ylabel('Validation F1')
    axes[2].set_title('Best Validation F1 (Higher is Better)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(display_names, rotation=45, ha='right')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(val_f1s):
        axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Experiment Metrics Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics summary to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    """
    Example usage
    """
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python plot_training.py <experiment_dir>                    # Plot single experiment")
        print("  python plot_training.py <exp1> <exp2> ...                   # Compare experiments")
        print("\nExample:")
        print("  python plot_training.py results/unet_resnet34_combined")
        print("  python plot_training.py results/unet_* results/fpn_*")
        sys.exit(1)
    
    experiment_dirs = sys.argv[1:]
    
    if len(experiment_dirs) == 1:
        # Single experiment - plot detailed curves
        exp_dir = experiment_dirs[0]
        history_path = Path(exp_dir) / 'training_history.json'
        
        if not history_path.exists():
            print(f"Error: History file not found: {history_path}")
            sys.exit(1)
        
        output_path = Path(exp_dir) / 'training_curves.png'
        plot_training_curves(str(history_path), save_path=str(output_path))
    
    else:
        # Multiple experiments - compare
        output_dir = Path(experiment_dirs[0]).parent / 'comparisons'
        output_dir.mkdir(exist_ok=True)
        
        compare_experiments(
            experiment_dirs,
            save_path=str(output_dir / 'comparison_curves.png')
        )
        
        plot_metric_summary(
            experiment_dirs,
            save_path=str(output_dir / 'metrics_summary.png')
        )