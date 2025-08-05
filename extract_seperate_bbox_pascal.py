#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

def categorize_size(rel_pct):
    if   rel_pct < 1.0:  return 'tiny'
    elif rel_pct < 5.0:  return 'small'
    elif rel_pct < 15.0: return 'medium'
    elif rel_pct < 40.0: return 'large'
    else:               return 'huge'

def poly_to_bbox_area(poly):
    """Convert polygon to bounding box and calculate area."""
    if len(poly) < 6:  # Need at least 3 points
        return 0, (0,0,0,0)
    
    coords = np.array(poly).reshape(-1, 2)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Bounding box
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()
    bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
    
    # Polygon area using shoelace formula
    n = len(coords)
    area = 0.5 * abs(sum(x_coords[i]*y_coords[(i+1)%n] - x_coords[(i+1)%n]*y_coords[i] 
                        for i in range(n)))
    
    return area, bbox

def parse_soda_json(json_path, img_dir):
    """Parse SODA-A JSON with polygon annotations."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_id = Path(json_path).stem
    img_path = img_dir / f"{image_id}.jpg"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    w, h = Image.open(img_path).size
    img_area = w * h
    
    objects = []
    
    # Handle different JSON structures
    if 'objects' in data:
        objs_list = data['objects']
    elif 'annotations' in data:
        objs_list = data['annotations']
    else:
        return image_id, []
    
    for idx, obj in enumerate(objs_list):
        try:
            # Get polygon coordinates
            if 'poly' not in obj:
                continue
            
            poly = obj['poly']
            if not poly or len(poly) < 6:
                continue
            
            # Get class ID
            class_id = obj.get('category_id', obj.get('class_id', 0))
            
            # Convert polygon to area and bbox
            poly_area, bbox = poly_to_bbox_area(poly)
            
            if poly_area == 0:
                continue
            
            rel_pct = poly_area / img_area * 100
            
            objects.append({
                'object_idx': idx,
                'class_id': class_id,
                'bbox': bbox,
                'poly_area': poly_area,
                'size_cat': categorize_size(rel_pct),
                'rel_pct': rel_pct
            })
            
        except Exception as e:
            continue
    
    return image_id, objects

def find_annotation_structure(ann_root):
    """Auto-detect SODA-A annotation structure."""
    ann_root = Path(ann_root)
    
    # Check for train/val/test subdirs
    if all((ann_root / split).exists() for split in ['train', 'val', 'test']):
        return 'split'
    
    # Check for flat structure with all JSONs
    json_files = list(ann_root.glob("*.json"))
    if json_files:
        return 'flat'
    
    # Check for other common structures
    subdirs = [d.name for d in ann_root.iterdir() if d.is_dir()]
    if subdirs:
        return subdirs
    
    return None

def extract_and_process_soda(images_dir, ann_root, output_dir):
    images_dir = Path(images_dir)
    ann_root = Path(ann_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Checking annotation structure in: {ann_root}")
    structure = find_annotation_structure(ann_root)
    print(f"📁 Found structure: {structure}")
    
    all_records = []
    error_count = 0
    success_count = 0
    
    if structure == 'split':
        splits = ['train', 'val', 'test']
    elif structure == 'flat':
        splits = ['.']  # Process all JSONs in ann_root directly
    elif isinstance(structure, list):
        splits = structure
    else:
        print(f"❌ No annotations found in {ann_root}")
        return None
    
    for split in splits:
        ann_dir = ann_root if split == '.' else ann_root / split
        
        if not ann_dir.exists():
            print(f"⚠️ Split {split} not found")
            continue
            
        json_files = list(ann_dir.glob("*.json"))
        print(f"📂 Processing {split}: {len(json_files)} JSONs")
        
        if not json_files:
            continue
        
        for jf in json_files:
            try:
                image_id, objs = parse_soda_json(jf, images_dir)
                if not objs:
                    error_count += 1
                    continue
                    
                success_count += 1
                
                # Group by class_id and pick largest instance
                by_cls = defaultdict(list)
                for o in objs:
                    by_cls[o['class_id']].append(o)
                
                for cls_id, insts in by_cls.items():
                    largest = max(insts, key=lambda x: x['poly_area'])
                    rec = {
                        'image_id': image_id,
                        'split': split if split != '.' else 'all',
                        'class_id': cls_id,
                        'size_category': largest['size_cat'],
                        'poly_area': largest['poly_area'],
                        'relative_pct': round(largest['rel_pct'], 2)
                    }
                    all_records.append(rec)
                    
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"⚠️ Error processing {jf.name}: {e}")
    
    print(f"✅ Successfully processed: {success_count} files")
    print(f"❌ Errors: {error_count} files")
    
    # Write CSV
    out_csv = output_dir / "soda_object_sizes.csv"
    keys = ['image_id', 'split', 'class_id', 'size_category', 'poly_area', 'relative_pct']
    
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)
    
    print(f"📄 Wrote {len(all_records)} rows to {out_csv}")
    
    if len(all_records) > 0:
        print("\n📊 Sample records:")
        for r in all_records[:3]:
            print(f"  {r}")
        
        # Show class distribution
        from collections import Counter
        class_dist = Counter(r['class_id'] for r in all_records)
        size_dist = Counter(r['size_category'] for r in all_records)
        print(f"\n📈 Class distribution: {dict(class_dist.most_common(5))}")
        print(f"📏 Size distribution: {dict(size_dist)}")
    
    return out_csv

if __name__ == "__main__":
    # Update these paths to match your actual directory structure
    IMG_DIR = "/home/wirin/Ashish/VLM/data/Images"  # Where your .jpg files are
    ANN_ROOT = "/home/wirin/Ashish/VLM/data/Annotations"  # Where your .json files are
    OUTPUT_DIR = "/home/wirin/Ashish/VLM/output"
    
    # Print current working directory for debugging
    print(f"🏠 Current directory: {os.getcwd()}")
    print(f"📁 Looking for images in: {IMG_DIR}")
    print(f"📄 Looking for annotations in: {ANN_ROOT}")
    
    extract_and_process_soda(IMG_DIR, ANN_ROOT, OUTPUT_DIR)