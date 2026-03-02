#!/usr/bin/env python3
"""
Script to clean up YOLO dataset filenames by removing .rf.<hash> portions
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple


def extract_clean_filename(filename: str) -> str:
    """
    Extract clean filename by removing .rf.<hash> portion.
    
    Examples:
        Grid4_91_0001_Jul30_jpg.rf.5ed3ea6ca63560131c3fb089821e88fa.jpg 
        -> Grid4_91_0001_Jul30.jpg
        
        grid-2_square97_hole115_0_hm_png.rf.ad57904876354ecae096adae934c68f8.jpg
        -> grid-2_square97_hole115_0_hm.jpg
    """
    # Pattern: matches .rf.<hash>.<extension>
    # The hash is typically 32 hexadecimal characters
    pattern = r'\.rf\.[a-f0-9]{32}\.(jpg|txt|png)$'
    
    # Check if pattern matches
    match = re.search(pattern, filename)
    if match:
        # Remove the .rf.<hash> part, keep only the extension
        extension = match.group(1)
        clean_name = re.sub(pattern, f'.{extension}', filename)
        
        # Handle cases where original extension was in the name (jpg -> _jpg, png -> _png)
        # Convert _jpg, _png back to proper extension
        clean_name = re.sub(r'_jpg\.jpg$', '.jpg', clean_name)
        clean_name = re.sub(r'_png\.jpg$', '.jpg', clean_name)
        clean_name = re.sub(r'_jpg\.txt$', '.txt', clean_name)
        clean_name = re.sub(r'_png\.txt$', '.txt', clean_name)
        
        return clean_name
    
    # If pattern doesn't match, return original filename
    return filename


def rename_files(base_dir: str, dry_run: bool = True) -> Tuple[int, int, Dict[str, str]]:
    """
    Rename all files in the images and labels directories.
    
    Args:
        base_dir: Base directory containing images/ and labels/ subdirectories
        dry_run: If True, only print what would be renamed without actually renaming
        
    Returns:
        Tuple of (success_count, skip_count, rename_mapping)
    """
    base_path = Path(base_dir)
    images_dir = base_path / 'images'
    labels_dir = base_path / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Directory structure invalid. Expected {images_dir} and {labels_dir}")
    
    rename_mapping = {}
    success_count = 0
    skip_count = 0
    
    # Process images
    print(f"\n{'=' * 60}")
    print(f"Processing images directory: {images_dir}")
    print(f"{'=' * 60}")
    
    for img_file in sorted(images_dir.glob('*.jpg')):
        old_name = img_file.name
        new_name = extract_clean_filename(old_name)
        
        if old_name != new_name:
            new_path = img_file.parent / new_name
            
            if new_path.exists():
                print(f"⚠️  SKIP: {new_name} already exists")
                skip_count += 1
                continue
            
            print(f"{'[DRY RUN] ' if dry_run else ''}✓ {old_name}")
            print(f"  -> {new_name}")
            
            if not dry_run:
                img_file.rename(new_path)
            
            rename_mapping[old_name] = new_name
            success_count += 1
        else:
            skip_count += 1
    
    # Process labels
    print(f"\n{'=' * 60}")
    print(f"Processing labels directory: {labels_dir}")
    print(f"{'=' * 60}")
    
    for label_file in sorted(labels_dir.glob('*.txt')):
        old_name = label_file.name
        new_name = extract_clean_filename(old_name)
        
        if old_name != new_name:
            new_path = label_file.parent / new_name
            
            if new_path.exists():
                print(f"⚠️  SKIP: {new_name} already exists")
                skip_count += 1
                continue
            
            print(f"{'[DRY RUN] ' if dry_run else ''}✓ {old_name}")
            print(f"  -> {new_name}")
            
            if not dry_run:
                label_file.rename(new_path)
            
            success_count += 1
        else:
            skip_count += 1
    
    return success_count, skip_count, rename_mapping


if __name__ == "__main__":
    # Set the base directory containing images/ and labels/
    base_directory = r'C:\Users\Yifei\Documents\cryo\revised_annotation'
    
    # Perform renaming (set dry_run=False to actually rename)
    renamed, skipped, mapping = rename_files(base_directory, dry_run=False)
    
    print(f"\nSummary:")
    print(f"Total files renamed: {renamed}")
    print(f"Total files skipped: {skipped}")