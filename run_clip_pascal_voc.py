#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract individual object crops from SODA-A images using polygon annotations.
Creates masked crops where only the target object is visible.
"""

import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

def categorize_size(rel_pct):
    if   rel_pct < 1.0:  return 'tiny'
    elif rel_pct < 5.0:  return 'small'
    elif rel_pct < 15.0: return 'medium'
    elif rel_pct < 40.0: return 'large'
    else:               return 'huge'

def polygon_area(poly_coords):
    """Calculate polygon area using shoelace formula."""
    if len(poly_coords) < 6:
        return 0
    coords = np.array(poly_coords).reshape(-1, 2)
    x, y = coords[:, 0], coords[:, 1]
    n = len(coords)
    return 0.5 * abs(sum(x[i]*y[(i+1)%n] - x[(i+1)%n]*y[i] for i in range(n)))

def create_polygon_mask(poly_coords, img_shape):
    """Create binary mask from polygon coordinates."""
    if len(poly_coords) < 6:
        return np.zeros(img_shape[:2], dtype=np.uint8)
    
    coords = np.array(poly_coords).reshape(-1, 2).astype(np.int32)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords], 255)
    return mask

def extract_object_crop(image, poly_coords, padding=10):
    """Extract bounding box crop with polygon mask applied."""
    if len(poly_coords) < 6:
        return None
    
    coords = np.array(poly_coords).reshape(-1, 2)
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    
    # Get bounding box with padding
    xmin = max(0, int(x_coords.min()) - padding)
    ymin = max(0, int(y_coords.min()) - padding)
    xmax = min(image.shape[1], int(x_coords.max()) + padding)
    ymax = min(image.shape[0], int(y_coords.max()) + padding)
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    # Crop image
    crop = image[ymin:ymax, xmin:xmax].copy()
    
    # Create mask for cropped region
    adjusted_coords = coords.copy()
    adjusted_coords[:, 0] -= xmin
    adjusted_coords[:, 1] -= ymin
    
    crop_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(crop_mask, [adjusted_coords.astype(np.int32)], 255)
    
    # Apply mask (black out non-object areas)
    crop[crop_mask == 0] = 0
    
    return crop

def process_soda_annotations(images_dir, ann_root, output_dir):
    """Extract object crops from SODA-A annotations."""
    images_dir = Path(images_dir)
    ann_root = Path(ann_root)
    output_dir = Path(output_dir)
    
    # SODA-A class mapping
    SODA_CLASSES = [
        "Car", "Truck", "Van", "Bus",
        "Cyclist", "Tricycle", "Motor", "Person", "Others"
    ]
    
    all_records = []
    
    for split in ['train', 'val', 'test']:
        ann_dir = ann_root / split
        if not ann_dir.exists():
            continue
            
        split_output_dir = output_dir / f"{split}_crops"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = list(ann_dir.glob("*.json"))
        print(f"Processing {split}: {len(json_files)} images")
        
        for json_file in tqdm(json_files, desc=f"Extracting {split}"):
            try:
                # Load annotation
                with open(json_file) as f:
                    data = json.load(f)
                
                image_id = json_file.stem
                img_path = images_dir / f"{image_id}.jpg"
                
                if not img_path.exists():
                    continue
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                h, w = image.shape[:2]
                img_area = h * w
                
                # Process each object
                objects = data.get('objects', data.get('annotations', []))
                
                for obj_idx, obj in enumerate(objects):
                    try:
                        # Get polygon and class
                        poly_coords = obj.get('poly', [])
                        if len(poly_coords) < 6:
                            continue
                        
                        cls_id = obj.get('category_id', 0)
                        cls_name = SODA_CLASSES[cls_id] if cls_id < len(SODA_CLASSES) else "Unknown"
                        
                        # Calculate size category
                        area = polygon_area(poly_coords)
                        rel_pct = (area / img_area) * 100
                        size_cat = categorize_size(rel_pct)
                        
                        # Extract object crop
                        crop = extract_object_crop(image, poly_coords)
                        if crop is None or crop.size == 0:
                            continue
                        
                        # Save crop
                        crop_filename = f"{image_id}_{obj_idx:02d}_{cls_name}_{size_cat}.jpg"
                        crop_path = split_output_dir / crop_filename
                        cv2.imwrite(str(crop_path), crop)
                        
                        # Record metadata
                        all_records.append({
                            'original_image_id': image_id,
                            'object_image_filename': crop_filename,
                            'object_image_path': str(crop_path),
                            'split': split,
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'size_category': size_cat,
                            'poly_area': area,
                            'relative_area_percent': round(rel_pct, 3)
                        })
                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
    
    # Save metadata CSV
    csv_path = output_dir / "soda_individual_objects.csv"
    if all_records:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
    
    print(f"\n✅ Extracted {len(all_records)} object crops")
    print(f"📄 Metadata saved to: {csv_path}")
    
    # Show statistics
    if all_records:
        from collections import Counter
        size_dist = Counter(r['size_category'] for r in all_records)
        class_dist = Counter(r['class_name'] for r in all_records)
        
        print(f"\n📊 Size distribution: {dict(size_dist)}")
        print(f"🏷️  Class distribution: {dict(class_dist.most_common(5))}")
    
    return csv_path

if __name__ == "__main__":
    # Configuration
    IMAGES_DIR = "/home/wirin/Ashish/VLM/data/Images"
    ANN_ROOT = "/home/wirin/Ashish/VLM/data/Annotations"
    OUTPUT_DIR = "/home/wirin/Ashish/VLM/data/soda_object_crops"
    
    print("🚀 Starting SODA-A object extraction...")
    csv_path = process_soda_annotations(IMAGES_DIR, ANN_ROOT, OUTPUT_DIR)
    print(f"✅ Complete! Now run CLIP evaluation with: {csv_path}")



