#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process SODA-A annotations (polygon-based) and produce a CSV listing
the largest object per class per image with size categories.
"""

import os
import json
import csv
import numpy as np
from collections import defaultdict
from pathlib import Path

def categorize_size(relative_area_percent):
    """Categorize object size based on relative area percentage"""
    if relative_area_percent < 1.0:
        return 'tiny'
    elif relative_area_percent < 5.0:
        return 'small'
    elif relative_area_percent < 15.0:
        return 'medium'
    elif relative_area_percent < 40.0:
        return 'large'
    else:
        return 'huge'

def polygon_area(poly):
    """Calculate polygon area using shoelace formula."""
    if len(poly) < 6:  # Need at least 3 points (x,y pairs)
        return 0
    
    coords = np.array(poly).reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Shoelace formula
    n = len(coords)
    area = 0.5 * abs(sum(x[i]*y[(i+1)%n] - x[(i+1)%n]*y[i] for i in range(n)))
    return area

def parse_soda_annotation(json_path, img_dir):
    """Parse SODA-A JSON with polygon annotations."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_id = Path(json_path).stem
    img_path = img_dir / f"{image_id}.jpg"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Get image dimensions
    from PIL import Image
    w, h = Image.open(img_path).size
    img_area = w * h

    # SODA-A class mapping (9 classes)
    class_map = {
        0: "Car", 1: "Truck", 2: "Van", 3: "Bus",
        4: "Cyclist", 5: "Tricycle", 6: "Motor", 7: "Person", 8: "Others"
    }

    objects = []
    
    # Handle different JSON structures
    objs_list = data.get('objects', data.get('annotations', []))
    
    for obj in objs_list:
        try:
            # Get polygon coordinates
            if 'poly' not in obj:
                continue
            
            poly = obj['poly']
            if not poly or len(poly) < 6:
                continue
            
            # Get class info
            cls_id = obj.get('category_id', obj.get('class_id', 0))
            cls_name = class_map.get(cls_id, f"class_{cls_id}")
            
            # Calculate polygon area
            poly_area = polygon_area(poly)
            if poly_area == 0:
                continue
            
            rel_pct = poly_area / img_area * 100
            size_cat = categorize_size(rel_pct)
            
            objects.append({
                'class_name': cls_name,
                'poly_area': poly_area,
                'relative_area_percent': rel_pct,
                'size_category': size_cat
            })
            
        except Exception as e:
            continue

    return image_id, img_area, objects

def process_soda_split(ann_dir, img_dir, split_name):
    """Process one split of SODA-A annotations."""
    results = []
    json_files = list(Path(ann_dir).glob("*.json"))
    print(f"Found {len(json_files)} annotations in split '{split_name}'")
    
    processed = 0
    errors = 0
    
    for jf in json_files:
        try:
            image_id, img_area, objs = parse_soda_annotation(jf, img_dir)
            if not objs:
                errors += 1
                continue
                
            processed += 1
            
            # Group by class, pick largest area per class
            by_class = defaultdict(list)
            for o in objs:
                by_class[o['class_name']].append(o)
            
            for cls, insts in by_class.items():
                largest = max(insts, key=lambda x: x['poly_area'])
                results.append({
                    'image_id': image_id,
                    'split': split_name,
                    'class_name': cls,
                    'size_category': largest['size_category'],
                    'poly_area': largest['poly_area'],
                    'relative_pct': round(largest['relative_area_percent'], 2)
                })
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Show first 5 errors
                print(f"⚠️  Skipping {jf.name}: {e}")
    
    print(f"✅ Processed: {processed}, Errors: {errors}")
    return results

def find_data_structure(data_root):
    """Auto-detect SODA-A data structure."""
    data_root = Path(data_root)
    
    # Common variations
    possible_img_dirs = ["images", "Images", "JPEGImages", "img"]
    possible_ann_dirs = ["annotations", "Annotations", "ann", "labels"]
    
    img_dir = None
    ann_root = None
    
    for img_name in possible_img_dirs:
        if (data_root / img_name).exists():
            img_dir = data_root / img_name
            break
    
    for ann_name in possible_ann_dirs:
        if (data_root / ann_name).exists():
            ann_root = data_root / ann_name
            break
    
    return img_dir, ann_root

def main():
    # Try to find the correct data paths
    possible_roots = [
        "/home/wirin/Ashish/VLM/data",
        "~/Ashish/VLM/data", 
        "./data",
        "/home/wirin/Ashish/VLM/ScaleVLM/data",
        "."
    ]
    
    DATA_ROOT = None
    for root in possible_roots:
        root_path = Path(root).expanduser()
        if root_path.exists():
            DATA_ROOT = root_path
            break
    
    if DATA_ROOT is None:
        print("❌ Could not find data directory. Please set DATA_ROOT manually.")
        return
    
    print(f"🔍 Using data root: {DATA_ROOT}")
    
    # Auto-detect structure
    IMG_DIR, ANN_ROOT = find_data_structure(DATA_ROOT)
    
    if IMG_DIR is None or ANN_ROOT is None:
        print(f"❌ Could not find images or annotations in {DATA_ROOT}")
        print(f"Contents: {list(DATA_ROOT.iterdir())}")
        return
    
    print(f"📁 Images: {IMG_DIR}")
    print(f"📄 Annotations: {ANN_ROOT}")
    
    OUTPUT_CSV = DATA_ROOT / "soda_object_sizes.csv"

    all_results = []
    
    # Check for split structure or flat structure
    splits_to_check = ["train", "val", "test"]
    found_splits = []
    
    for split in splits_to_check:
        split_dir = ANN_ROOT / split
        if split_dir.exists() and list(split_dir.glob("*.json")):
            found_splits.append(split)
    
    if not found_splits:
        # Flat structure - all JSONs in annotations root
        print("📂 Using flat annotation structure")
        res = process_soda_split(ANN_ROOT, IMG_DIR, "all")
        all_results.extend(res)
    else:
        # Split structure
        print(f"📂 Found splits: {found_splits}")
        for split in found_splits:
            ann_dir = ANN_ROOT / split
            res = process_soda_split(ann_dir, IMG_DIR, split)
            all_results.extend(res)

    # Write CSV
    if all_results:
        fieldnames = ['image_id', 'split', 'class_name', 'size_category', 'poly_area', 'relative_pct']
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)

        print(f"\n✅ Wrote {len(all_results)} rows to {OUTPUT_CSV}")

        # Statistics
        size_counts = defaultdict(int)
        class_counts = defaultdict(int)
        for r in all_results:
            size_counts[r['size_category']] += 1
            class_counts[r['class_name']] += 1

        print("\nSize category distribution:")
        for sz in ['tiny', 'small', 'medium', 'large', 'huge']:
            if sz in size_counts:
                print(f"  {sz}: {size_counts[sz]}")

        print("\nClass distribution:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {cnt}")
    else:
        print("❌ No results generated. Check file paths and JSON structure.")

if __name__ == "__main__":
    main()



