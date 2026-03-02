import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import albumentations as A
from transforms.cryo_transforms import get_light_train_transforms


class CryoInstanceDataset(Dataset):
    """
    Dataset for instance segmentation of cryo-EM vesicles.
    Loads YOLO polygon annotations and creates proper instance masks.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        transform: Optional[A.Compose] = None,
        mode: str = 'train'  # 'train' or 'inference'
    ):
        """
        Args:
            data_dir: Root directory containing 'images' and 'labels' folders
            transform: Albumentations transform pipeline
            mode: 'train' for training with annotations, 'inference' for prediction only
        """
        self.image_dir = os.path.join(data_dir, 'images')
        self.annotation_dir = os.path.join(data_dir, 'labels') if mode == 'train' else None
        self.transform = transform
        self.mode = mode
        
        # Get all image filenames
        self.sample_ids = sorted([
            f for f in os.listdir(self.image_dir) 
            if os.path.isfile(os.path.join(self.image_dir, f))
        ])
        
        print(f"Loaded {len(self.sample_ids)} samples from {data_dir}")
        
        if mode == 'train':
            self._verify_annotations()
    
    def _verify_annotations(self):
        """Check annotation files and print statistics"""
        n_with_annotations = 0
        n_instances_total = 0
        
        for sample_id in self.sample_ids:
            base_id = os.path.splitext(sample_id)[0]
            label_file = os.path.join(self.annotation_dir, f"{base_id}.txt")
            
            if os.path.isfile(label_file):
                with open(label_file, 'r') as f:
                    n_instances = len([line for line in f if line.strip()])
                if n_instances > 0:
                    n_with_annotations += 1
                    n_instances_total += n_instances
        
        print(f"  - Samples with annotations: {n_with_annotations}/{len(self.sample_ids)}")
        print(f"  - Total instances: {n_instances_total}")
        if n_with_annotations > 0:
            print(f"  - Avg instances per annotated image: {n_instances_total/n_with_annotations:.1f}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def load_image(self, sample_id: str) -> np.ndarray:
        """Load grayscale image"""
        image_path = os.path.join(self.image_dir, sample_id)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image
    
    def load_polygons(self, sample_id: str, image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Load YOLO polygon annotations and convert to pixel coordinates.
        
        Args:
            sample_id: Image filename
            image_shape: (height, width) of the image
            
        Returns:
            List of polygons, each as numpy array of shape (N, 2) with pixel coordinates
        """
        if self.annotation_dir is None:
            return []
        
        base_id = os.path.splitext(sample_id)[0]
        label_file = os.path.join(self.annotation_dir, f"{base_id}.txt")
        
        if not os.path.isfile(label_file):
            return []
        
        h, w = image_shape
        polygons = []
        
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                tokens = line.split()
                # Format: class_id x1 y1 x2 y2 ... (normalized coords)
                if len(tokens) < 7:  # Need at least 3 points (class + 6 coords)
                    continue
                
                try:
                    # Skip class_id, get normalized coordinates
                    coords = list(map(float, tokens[1:]))
                    
                    # Convert to pixel coordinates
                    points = []
                    for i in range(0, len(coords), 2):
                        x = int(np.clip(coords[i] * w, 0, w - 1))
                        y = int(np.clip(coords[i + 1] * h, 0, h - 1))
                        points.append([x, y])
                    
                    if len(points) >= 3:  # Valid polygon needs at least 3 points
                        polygons.append(np.array(points, dtype=np.int32))
                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line in {label_file}: {line}")
                    continue
        
        return polygons
    
    def create_instance_mask(
        self, 
        polygons: List[np.ndarray], 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create instance segmentation mask from polygons.
        Each instance gets a unique integer ID.
        
        Args:
            polygons: List of polygons (pixel coordinates)
            image_shape: (height, width)
            
        Returns:
            Instance mask of shape (H, W) where each pixel contains instance ID (0 = background)
        """
        h, w = image_shape
        instance_mask = np.zeros((h, w), dtype=np.int32)
        
        for idx, polygon in enumerate(polygons, start=1):
            # Each instance gets ID = idx (1, 2, 3, ...)
            cv2.fillPoly(instance_mask, [polygon.reshape(-1, 1, 2)], color=idx)
        
        return instance_mask
    
    def instance_mask_to_binary(self, instance_mask: np.ndarray) -> np.ndarray:
        """
        Convert instance mask to binary semantic mask.
        
        Args:
            instance_mask: Shape (H, W) with instance IDs
            
        Returns:
            Binary mask of shape (H, W) where 0=background, 1=foreground
        """
        return (instance_mask > 0).astype(np.float32)
    
    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        
        # Load image
        image = self.load_image(sample_id)
        h, w = image.shape
        
        if self.mode == 'train':
            # Load polygons and create instance mask
            polygons = self.load_polygons(sample_id, (h, w))
            instance_mask = self.create_instance_mask(polygons, (h, w))
            
            # Convert instance mask to binary for semantic segmentation
            binary_mask = self.instance_mask_to_binary(instance_mask)
            
            # Store original for visualization
            image_original = image.copy()
            instance_mask_original = instance_mask.copy()
            binary_mask_original = binary_mask.copy()
            
            # Apply transforms
            if self.transform:
                # Transform both binary mask AND instance mask together
                # Use additional_targets to transform instance_mask
                transformed = self.transform(
                    image=image,
                    mask=binary_mask,
                    instance_mask=instance_mask.astype(np.float32)  # Add instance mask
                )
                image = transformed['image']
                binary_mask = transformed['mask']
                instance_mask = transformed['instance_mask'].long() if isinstance(transformed['instance_mask'], torch.Tensor) else transformed['instance_mask'].astype(np.int32)
            else:
                # Convert to tensor if no transform
                image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
                binary_mask = torch.from_numpy(binary_mask).float()
                instance_mask = torch.from_numpy(instance_mask).long()
            
            return {
                'image': image,
                'mask': binary_mask,
                'instance_mask': instance_mask,  # Transformed instance mask
                'sample_id': sample_id,
                'n_instances': len(polygons),
                # Original versions for visualization
                'image_original': image_original,
                'instance_mask_original': instance_mask_original,
                'binary_mask_original': binary_mask_original
            }
        
        else:  # inference mode
            # Apply transforms (image only)
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            
            return {
                'image': image,
                'sample_id': sample_id,
                'original_shape': (h, w)
            }


def custom_collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function to handle variable-sized data.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with stacked tensors where possible
    """
    if 'mask' in batch[0]:  # Training mode
        result = {
            'image': torch.stack([item['image'] for item in batch]),
            'mask': torch.stack([item['mask'] for item in batch]),
            'instance_mask': [item['instance_mask'] for item in batch],  # Keep as list (variable instances)
            'sample_id': [item['sample_id'] for item in batch],
            'n_instances': [item['n_instances'] for item in batch]
        }
        
        # Add original versions if present
        if 'image_original' in batch[0]:
            result['image_original'] = [item['image_original'] for item in batch]
            result['instance_mask_original'] = [item['instance_mask_original'] for item in batch]
            result['binary_mask_original'] = [item['binary_mask_original'] for item in batch]
        
        return result
    else:  # Inference mode
        return {
            'image': torch.stack([item['image'] for item in batch]),
            'sample_id': [item['sample_id'] for item in batch],
            'original_shape': [item['original_shape'] for item in batch]
        }


# ============================================================================
# Utility functions for analysis
# ============================================================================

def visualize_sample(dataset: CryoInstanceDataset, idx: int, save_path: Optional[str] = None):
    """Visualize a sample with instance masks overlaid, showing both before and after transforms"""
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    
    # Convert tensor back to numpy if needed
    if isinstance(sample['image'], torch.Tensor):
        image = sample['image'].squeeze().numpy()
    else:
        image = sample['image']
    
    if dataset.mode == 'train':
        instance_mask = sample['instance_mask']
        binary_mask = sample['mask']
        
        # Convert tensors to numpy
        if isinstance(instance_mask, torch.Tensor):
            instance_mask = instance_mask.numpy()
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = binary_mask.numpy()
        
        # Get original versions
        image_original = sample['image_original']
        instance_mask_original = sample['instance_mask_original']
        binary_mask_original = sample['binary_mask_original']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 0: Original (before transforms)
        axes[0, 0].imshow(image_original, cmap='gray')
        axes[0, 0].set_title(f"Original Image\n{sample['sample_id']}")
        axes[0, 0].axis('off')
        
        n_instances = sample['n_instances']
        if n_instances > 0:
            cmap = plt.cm.get_cmap('turbo', n_instances + 1)
            im = axes[0, 1].imshow(instance_mask_original, cmap=cmap, interpolation='nearest')
            axes[0, 1].set_title(f"Original Instance Mask\n{n_instances} instances")
            plt.colorbar(im, ax=axes[0, 1])
        else:
            axes[0, 1].imshow(instance_mask_original, cmap='gray')
            axes[0, 1].set_title("Original Instance Mask\n0 instances")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(binary_mask_original, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title("Original Binary Mask")
        axes[0, 2].axis('off')
        
        # Row 1: After transforms
        axes[1, 0].imshow(image, cmap='gray')
        axes[1, 0].set_title("Transformed Image")
        axes[1, 0].axis('off')
        
        if n_instances > 0:
            cmap = plt.cm.get_cmap('turbo', n_instances + 1)
            im = axes[1, 1].imshow(instance_mask, cmap=cmap, interpolation='nearest')
            axes[1, 1].set_title("Transformed Instance Mask")
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].imshow(instance_mask, cmap='gray')
            axes[1, 1].set_title("Transformed Instance Mask")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title("Transformed Binary Mask")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    else:
        # Inference mode - just show image
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"Image: {sample['sample_id']}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def analyze_dataset(dataset: CryoInstanceDataset) -> dict:
    """Analyze dataset statistics"""
    if dataset.mode != 'train':
        print("Dataset is in inference mode - no annotations to analyze")
        return {}
    
    stats = {
        'n_samples': len(dataset),
        'n_with_instances': 0,
        'total_instances': 0,
        'instances_per_image': [],
        'image_sizes': []
    }
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        n_instances = sample['n_instances']
        
        if n_instances > 0:
            stats['n_with_instances'] += 1
            stats['total_instances'] += n_instances
            stats['instances_per_image'].append(n_instances)
        
        # Get image size
        if isinstance(sample['image'], torch.Tensor):
            h, w = sample['image'].shape[-2:]
        else:
            h, w = sample['image'].shape[:2]
        stats['image_sizes'].append((h, w))
    
    # Calculate summary statistics
    if stats['instances_per_image']:
        stats['mean_instances'] = np.mean(stats['instances_per_image'])
        stats['median_instances'] = np.median(stats['instances_per_image'])
        stats['max_instances'] = np.max(stats['instances_per_image'])
        stats['min_instances'] = np.min(stats['instances_per_image'])
    
    return stats


def print_dataset_stats(stats: dict):
    """Print dataset statistics in a formatted way"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {stats['n_samples']}")
    print(f"Samples with instances: {stats['n_with_instances']}")
    print(f"Samples without instances: {stats['n_samples'] - stats['n_with_instances']}")
    print(f"Total instances: {stats['total_instances']}")
    
    if stats.get('mean_instances'):
        print(f"\nInstances per image:")
        print(f"  Mean: {stats['mean_instances']:.2f}")
        print(f"  Median: {stats['median_instances']:.1f}")
        print(f"  Min: {stats['min_instances']}")
        print(f"  Max: {stats['max_instances']}")
    
    # Image size statistics
    unique_sizes = set(stats['image_sizes'])
    print(f"\nImage sizes: {len(unique_sizes)} unique size(s)")
    for size in sorted(unique_sizes):
        count = stats['image_sizes'].count(size)
        print(f"  {size[0]}x{size[1]}: {count} images")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    """Test the dataset"""
    import sys
    
    # Example usage
    data_dir = r'C:\Users\Yifei\Documents\cryo\revised_annotation\split\val'
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        print("Please update the path in the script.")
        sys.exit(1)
    
    # Create dataset
    dataset = CryoInstanceDataset(data_dir, transform=get_light_train_transforms(), mode='train')
    
    # Analyze
    stats = analyze_dataset(dataset)
    print_dataset_stats(stats)
    
    # Visualize first few samples
    print("Visualizing first 3 samples...")
    for i in range(len(dataset)):
        visualize_sample(dataset, i, save_path=f'sample_{i}.png')