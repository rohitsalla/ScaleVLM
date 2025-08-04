#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and mask individual objects from SODA-A images.
For each JSON annotation (one per image) in train/val/test, 
produce masked crops where only the target object is visible.
Also generate a CSV metadata file.
"""

import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

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

def parse_soda_annotation(json_path, img_dir):
    """
    Parse one SODA-A JSON. Returns:
      image_id, filename, img_width, img_height, list of objects dicts with:
      class_name, bbox (rotated box), axis-aligned bbox, area, rel_area, size_category, idx
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_id = Path(json_path).stem
    filename = f"{image_id}.jpg"
    img_path = img_dir / filename
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    w, h = Image.open(img_path).size
    img_area = w * h

    objects = []
    for idx, obj in enumerate(data.get('objects', [])):
        cls = data['categories'][obj['category_id']] if 'categories' in data else str(obj['category_id'])
        cx, cy, bw, bh, _ = obj['bbox']
        # axis-aligned bounding box (xmin,ymin,xmax,ymax)
        xmin = int(cx - bw/2)
        ymin = int(cy - bh/2)
        xmax = int(cx + bw/2)
        ymax = int(cy + bh/2)
        bbox_area = bw * bh
        rel_pct = bbox_area / img_area * 100
        size_cat = categorize_size(rel_pct)
        objects.append({
            'class_name': cls,
            'bbox_rot': (cx, cy, bw, bh),
            'bbox': (xmin, ymin, xmax, ymax),
            'bbox_area': bbox_area,
            'relative_area_percent': rel_pct,
            'size_category': size_cat,
            'object_idx': idx
        })
    return image_id, filename, w, h, objects

def create_masked_image(image_path, target_bbox, all_bboxes):
    """Return image with only target_bbox visible; others blacked-out."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    masked = np.zeros_like(img)
    xmin, ymin, xmax, ymax = target_bbox
    masked[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]
    return masked

def extract_and_process_soda(images_dir, annotations_root, output_root):
    """
    For each split (train/val/test) under annotations_root:
      - parse JSONs
      - mask and save each object crop
      - record metadata
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_records = []

    for split in ['train', 'val', 'test']:
        ann_dir = Path(annotations_root) / split
        if not ann_dir.exists():
            print(f"Skipping missing split: {split}")
            continue
        img_dir = Path(images_dir)
        out_dir = output_root / f"{split}_objects"
        out_dir.mkdir(exist_ok=True)
        print(f"Processing split '{split}' with {len(list(ann_dir.glob('*.json')))} annotations")

        for jf in ann_dir.glob("*.json"):
            try:
                img_id, fn, w, h, objs = parse_soda_annotation(jf, img_dir)
            except Exception as e:
                print(f"⚠️  Skip {jf.name}: {e}")
                continue

            img_path = img_dir / fn
            bboxes = [o['bbox'] for o in objs]

            for o in objs:
                fname = f"{img_id}_{o['object_idx']}_{o['class_name']}_{o['size_category']}.jpg"
                save_path = out_dir / fname
                masked = create_masked_image(img_path, o['bbox'], bboxes)
                cv2.imwrite(str(save_path), masked)

                all_records.append({
                    'image_id': img_id,
                    'split': split,
                    'object_idx': o['object_idx'],
                    'class_name': o['class_name'],
                    'size_category': o['size_category'],
                    'bbox_coords': f"{o['bbox'][0]},{o['bbox'][1]},{o['bbox'][2]},{o['bbox'][3]}",
                    'relative_area_percent': round(o['relative_area_percent'],2),
                    'object_image_path': str(save_path)
                })

    # write CSV
    csv_path = output_root / "soda_individual_objects.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        for rec in all_records:
            writer.writerow(rec)

    print(f"\n✅  Saved {len(all_records)} object images under '{output_root}'")
    print(f"Metadata CSV: {csv_path}")
    return csv_path

if __name__ == "__main__":
    DATA_ROOT   = "/home/rohit/Downloads/ScaleVLM-master/data"
    IMAGES_DIR  = f"{DATA_ROOT}/images"
    ANNS_ROOT   = f"{DATA_ROOT}/annotations"
    OUT_ROOT    = f"{DATA_ROOT}/soda_extracted_objects"

    csv_metadata = extract_and_process_soda(IMAGES_DIR, ANNS_ROOT, OUT_ROOT)

    # Example: load the CSV for further use (e.g. CLIP benchmarking)
    # import pandas as pd
    # df = pd.read_csv(csv_metadata)
