#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --------------------------------------------------------------------------- #
#                1.  CONFIGURATION – change only the next line                #
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("/home/wirin/Ashish/VLM/data")   # ← your folder
# --------------------------------------------------------------------------- #

IMG_DIR     = DATA_ROOT / "Images"
ANN_ROOT    = DATA_ROOT / "Annotations"   # contains train / val / test

# SODA-A classes (9)
CLASSES = [
    "Car", "Truck", "Van", "Bus",
    "Cyclist", "Tricycle", "Motor", "Person", "Others"
]

SIZE_COLORS = dict(
    tiny='red', small='orange', medium='yellow',
    large='green', huge='blue'
)

# ------------- helpers ------------------------------------------------------ #
def polygon_from_coords(poly_coords):
    """Convert flat coordinate list to polygon array."""
    if len(poly_coords) < 6:  # Need at least 3 points
        return np.array([[0,0]])
    coords = np.array(poly_coords).reshape(-1, 2)
    return coords

def polygon_area(poly_coords):
    """Calculate polygon area using shoelace formula."""
    if len(poly_coords) < 6:
        return 0
    coords = np.array(poly_coords).reshape(-1, 2)
    x, y = coords[:, 0], coords[:, 1]
    n = len(coords)
    return 0.5 * abs(sum(x[i]*y[(i+1)%n] - x[(i+1)%n]*y[i] for i in range(n)))

def categorise(relative_area_pct):
    if   relative_area_pct < 1.0:  return 'tiny'
    elif relative_area_pct < 5.0:  return 'small'
    elif relative_area_pct < 15.0: return 'medium'
    elif relative_area_pct < 40.0: return 'large'
    else:                          return 'huge'

# ------------- main visualiser --------------------------------------------- #
def visualise(img_id: str, split: str = 'train'):
    """
    img_id = "00005"  (without extension)
    split  = 'train' | 'val' | 'test'
    """
    img_path = IMG_DIR / f"{img_id}.jpg"
    ann_path = ANN_ROOT / split / f"{img_id}.json"

    if not img_path.exists():
        print(f"❌ image not found   -> {img_path}")
        return
    if not ann_path.exists():
        print(f"❌ annotation missing -> {ann_path}")
        return

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    img_area = H * W

    with ann_path.open() as f:
        anno = json.load(f)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    details = []

    # Handle SODA-A polygon annotations
    objects = anno.get('objects', anno.get('annotations', []))
    
    for k, obj in enumerate(objects):
        try:
            # Get class info
            cls_id = obj.get('category_id', 0)
            cls = CLASSES[cls_id] if cls_id < len(CLASSES) else "Unknown"
            
            # Get polygon coordinates
            poly_coords = obj.get('poly', [])
            if not poly_coords or len(poly_coords) < 6:
                continue
                
            # Convert to polygon for visualization
            poly = polygon_from_coords(poly_coords)
            
            # Calculate area percentage
            area = polygon_area(poly_coords)
            pct = (area / img_area) * 100
            size = categorise(pct)
            color = SIZE_COLORS[size]

            # Draw polygon
            ax.add_patch(Polygon(poly, closed=True, fill=False,
                                linewidth=2.5, edgecolor=color))
            
            # Add label
            if len(poly) > 0:
                ax.text(poly[0,0], poly[0,1]-4,
                        f"{k}:{cls} ({size})", color=color,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec=color, lw=0.8),
                        fontsize=9)

            details.append((k, cls, size, pct))
            
        except Exception as e:
            print(f"⚠️ Error processing object {k}: {e}")
            continue

    ax.set_title(f"{img_id}.jpg  |  {split}  |  {len(details)} objects",
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    patches = [plt.Line2D([0],[0], color=SIZE_COLORS[s], lw=3,
                          label=s.capitalize())
               for s in SIZE_COLORS]
    ax.legend(handles=patches, loc='upper right')

    # Save visualization
    out_dir = DATA_ROOT / "viz"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{img_id}_{split}.jpg"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f"\nSaved visualisation   ➜  {out_path}")
    for k, cls, size, pct in details:
        print(f"[{k:02d}] {cls:<10} | {size:<6} | {pct:5.2f}% of image")

def explore_random_images(split='train', num_samples=3):
    """Visualize random images from a split."""
    ann_dir = ANN_ROOT / split
    if not ann_dir.exists():
        print(f"❌ Split {split} not found")
        return
    
    json_files = list(ann_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No annotations found in {split}")
        return
    
    sample_files = random.sample(json_files, min(num_samples, len(json_files)))
    print(f"🎲 Visualizing {len(sample_files)} random images from {split}:")
    
    for jf in sample_files:
        print(f"\n{'='*50}")
        visualise(jf.stem, split)

# ------------- quick demos -------------------------------------------------- #
if __name__ == "__main__":
    # Visualize specific image
    visualise("00005", "train")
    
    # Uncomment to explore random samples:
    # explore_random_images("train", 3)
    # explore_random_images("val", 2)



