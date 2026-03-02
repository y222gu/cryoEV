import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size: int = 1024, p_augment: float = 0.8):
    """
    Augmentation pipeline for training cryo-EM instance segmentation.
    
    Args:
        image_size: Target size for resize
        p_augment: Probability of applying augmentation pipeline
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        # Resize to consistent size
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        
        # Geometric transforms
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Transpose(p=1.0),
        ], p=0.8),
        
        # Rotation and scaling
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.15,
            rotate_limit=45,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        
        # Elastic deformation (good for biological structures)
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT,
            p=0.25
        ),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT,
            p=0.25
        ),
        
        # Intensity transforms (important for EM images)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),
                p=1.0
            ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=1.0
            ),
        ], p=0.6),
        
        # Noise (common in EM imaging)
        A.OneOf([
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0,
                p=1.0
            ),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=1.0
            ),
        ], p=0.4),
        
        # Blur (can occur in EM due to defocus)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        # Normalize and convert to tensor
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], additional_targets={'instance_mask': 'mask'})  # Add instance_mask as additional target


def get_val_transforms(image_size: int = 1024):
    """
    Validation transform pipeline (no augmentation).
    
    Args:
        image_size: Target size for resize
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], additional_targets={'instance_mask': 'mask'})  # Add instance_mask


def get_test_transforms(image_size: int = 1024):
    """
    Test/inference transform pipeline (same as validation).
    
    Args:
        image_size: Target size for resize
        
    Returns:
        Albumentations Compose transform
    """
    return get_val_transforms(image_size)


def get_heavy_train_transforms(image_size: int = 1024):
    """
    Heavy augmentation for very small datasets (<100 samples).
    
    Args:
        image_size: Target size for resize
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        
        # More aggressive geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=90,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7
        ),
        
        # Multiple elastic/grid transforms
        A.OneOf([
            A.ElasticTransform(
                alpha=150,
                sigma=150 * 0.05,
                alpha_affine=150 * 0.03,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ),
            A.OpticalDistortion(
                distort_limit=0.5,
                shift_limit=0.5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ),
        ], p=0.5),
        
        # Aggressive intensity transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        
        A.OneOf([
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),
        
        # More noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8), p=1.0),
        ], p=0.6),
        
        # More blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.MotionBlur(blur_limit=9, p=1.0),
        ], p=0.5),
        
        # Coarse dropout (simulate occlusions)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
        
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], additional_targets={'instance_mask': 'mask'})  # Add instance_mask


def get_light_train_transforms(image_size: int = 1024):
    """
    Light augmentation for larger datasets or when training is unstable.
    
    Args:
        image_size: Target size for resize
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        
        # Basic flips only
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Minimal intensity adjustment
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.95
        ),
        
        # Light noise
        # A.GaussNoise(var_limit=(50, 80.0), p=0.99),
        
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], additional_targets={'instance_mask': 'mask'})  # Add instance_mask


if __name__ == '__main__':
    """Test transforms"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create dummy image and mask
    image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.float32)
    cv2.circle(mask, (256, 256), 100, 1, -1)
    
    # Test train transform
    transform = get_train_transforms(image_size=512)
    
    # Apply multiple times to see variation
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image'].numpy().squeeze()
        aug_mask = augmented['mask'].numpy()
        
        axes[0, i].imshow(aug_image, cmap='gray')
        axes[0, i].set_title(f'Augmented Image {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(aug_mask, cmap='gray')
        axes[1, i].set_title(f'Augmented Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/transform_test.png', dpi=150)
    print("Saved transform test to /tmp/transform_test.png")
    plt.close()
    
    print("Transform test complete!")