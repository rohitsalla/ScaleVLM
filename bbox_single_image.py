#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.affinity import rotate
from shapely.geometry import box as shapely_box

# --------------------------------------------------------------------------- #
#                1.  CONFIGURATION – change only the next line                #
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("/home/rohit/Downloads/ScaleVLM-master/data")   # ← your folder
# --------------------------------------------------------------------------- #

IMG_DIR     = DATA_ROOT / "images"
ANN_ROOT    = DATA_ROOT / "annotations"   # contains train / val / test

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
def oriented_box_to_polygon(cx, cy, w, h, angle_deg):
    """Return the 4 corner points of an oriented rectangle."""
    rect = shapely_box(-w/2, -h/2, w/2, h/2)
    rect = rotate(rect, -angle_deg, use_radians=False)
    rect = shapely_box(*rect.bounds) if not rect.is_valid else rect
    poly = np.asarray(rect.exterior.coords[:-1])
    poly[:, 0] += cx
    poly[:, 1] += cy
    return poly

def categorise(relative_area_pct):
    if   relative_area_pct < 1.0:  return 'tiny'
    elif relative_area_pct < 5.0:  return 'small'
    elif relative_area_pct < 15.0: return 'medium'
    elif relative_area_pct < 40.0: return 'large'
    else:                          return 'huge'

# ------------- main visualiser --------------------------------------------- #
def visualise(img_id: str, split: str = 'train'):
    """
    img_id = "0000000"  (without extension)
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

    img      = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    H, W, _  = img.shape
    img_area = H * W

    with ann_path.open() as f:
        anno = json.load(f)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    details = []

    for k, obj in enumerate(anno['objects']):
        cls = CLASSES[obj['category_id']]
        cx, cy, w, h, ang = obj['bbox']
        poly  = oriented_box_to_polygon(cx, cy, w, h, ang)
        pct   = (w * h) / img_area * 100
        size  = categorise(pct)
        color = SIZE_COLORS[size]

        ax.add_patch(Polygon(poly, closed=True, fill=False,
                             linewidth=2.5, edgecolor=color))
        ax.text(poly[0,0], poly[0,1]-4,
                f"{k}:{cls} ({size})", color=color,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec=color, lw=0.8),
                fontsize=9)

        details.append((k, cls, size, pct))

    ax.set_title(f"{img_id}.jpg  |  {split}  |  {len(details)} objects",
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    patches = [plt.Line2D([0],[0], color=SIZE_COLORS[s], lw=3,
                          label=s.capitalize())
               for s in SIZE_COLORS]
    ax.legend(handles=patches, loc='upper right')

    out_dir = DATA_ROOT / "viz"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{img_id}_{split}.jpg"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f"\nSaved visualisation   ➜  {out_path}")
    for k, cls, size, pct in details:
        print(f"[{k:02d}] {cls:<10} | {size:<6} | {pct:5.2f}% of image")


# ------------- quick demos -------------------------------------------------- #
if __name__ == "__main__":
    visualise("0000005", "train")
    # Example for val:
    # ann_files = list((ANN_ROOT/'val').glob("*.json"))
    # random.shuffle(ann_files)
    # for jf in ann_files[:3]:
    #     visualise(jf.stem, "val")
