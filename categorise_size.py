#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process SODA-A annotations (one JSON per image, split into train/val/test)
and produce a CSV listing, for each image & class, the largest object’s size category.
"""

import os
import json
import csv
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

def parse_soda_annotation(json_path, img_dir):
    """
    Parse a single SODA-A JSON annotation file.
    Returns image_id, image_area, list of objects with class, bbox area, relative area, size.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_id = Path(json_path).stem
    img_path = img_dir / f"{image_id}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # get image dimensions
    from PIL import Image
    w, h = Image.open(img_path).size
    img_area = w * h

    objects = []
    # SODA-A stores rotated boxes as [cx, cy, w, h, angle]
    for obj in data.get('objects', []):
        cls_id = obj['category_id']
        cls_name = data['categories'][cls_id] if 'categories' in data else str(cls_id)
        cx, cy, bw, bh, _ = obj['bbox']
        bbox_area = bw * bh
        rel_pct = bbox_area / img_area * 100
        size_cat = categorize_size(rel_pct)
        objects.append({
            'class_name': cls_name,
            'bbox_area': bbox_area,
            'relative_area_percent': rel_pct,
            'size_category': size_cat
        })

    return image_id, img_area, objects

def process_soda_split(ann_dir, img_dir, split_name):
    """Process one split (train/val/test) of SODA-A annotations."""
    results = []
    json_files = list(Path(ann_dir).glob("*.json"))
    print(f"Found {len(json_files)} annotations in split '{split_name}'")
    for jf in json_files:
        try:
            image_id, img_area, objs = parse_soda_annotation(jf, img_dir)
            # group by class, pick largest bbox_area per class
            by_class = defaultdict(list)
            for o in objs:
                by_class[o['class_name']].append(o)
            for cls, insts in by_class.items():
                largest = max(insts, key=lambda x: x['bbox_area'])
                results.append({
                    'image_id': image_id,
                    'split': split_name,
                    'class_name': cls,
                    'size_category': largest['size_category']
                })
        except Exception as e:
            print(f"⚠️  Skipping {jf.name}: {e}")
    return results

def main():
    # adjust these paths to your dataset root
    DATA_ROOT   = Path("/home/rohit/Downloads/ScaleVLM-master/data")
    IMG_DIR     = DATA_ROOT / "images"
    ANN_ROOT    = DATA_ROOT / "annotations"
    OUTPUT_CSV  = DATA_ROOT / "soda_object_sizes.csv"

    all_results = []
    for split in ["train", "val", "test"]:
        ann_dir = ANN_ROOT / split
        if not ann_dir.exists():
            print(f"❌  Annotation folder missing: {ann_dir}")
            continue
        res = process_soda_split(ann_dir, IMG_DIR, split)
        all_results.extend(res)

    # write CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=['image_id','split','class_name','size_category'])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n✅  Wrote {len(all_results)} rows to {OUTPUT_CSV}")

    # stats
    size_counts = defaultdict(int)
    class_counts = defaultdict(int)
    for r in all_results:
        size_counts[r['size_category']] += 1
        class_counts[r['class_name']]   += 1

    print("\nSize category distribution:")
    for sz, cnt in sorted(size_counts.items()):
        print(f"  {sz}: {cnt}")

    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 classes:")
    for cls, cnt in top_classes:
        print(f"  {cls}: {cnt}")

if __name__ == "__main__":
    main()
