import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

# SODA-A classes (9)
SODA_CLASSES = [
    "Car", "Truck", "Van", "Bus",
    "Cyclist", "Tricycle", "Motor", "Person", "Others"
]

# Load CLIP model from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"  # or "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Configuration
DATA_ROOT = "/data/azfarm/siddhant/ICCV/dataset/SODA-A_clip"  # update as needed
CSV_PATH  = "/data/azfarm/siddhant/ICCV/soda_object_sizes.csv"

# Load dataset of extracted object crops
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(f"Size distribution: {df['size_category'].value_counts().to_dict()}")

# Pre-build text prompts
text_prompts = [f"a photo of a {cls.lower()}" for cls in SODA_CLASSES]

results = []
summary = {}

for size_category, group in df.groupby('size_category'):
    print(f"\nEvaluating {size_category} objects ({len(group)} samples)...")
    correct1 = 0
    correct5 = 0
    total_conf = 0.0
    preds = []

    batch_size = 32
    for start in tqdm(range(0, len(group), batch_size), desc=size_category):
        batch = group.iloc[start:start+batch_size]
        images = []
        true_labels = []
        valid_idxs = []

        # Load images
        for i, (_, row) in enumerate(batch.iterrows()):
            img_path = row['object_image_path']
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    true_labels.append(row['class_name'])
                    valid_idxs.append(i)
                except:
                    continue

        if not images:
            continue

        # Process batch with CLIP
        inputs = processor(text=text_prompts,
                           images=images,
                           return_tensors="pt",
                           padding=True,
                           truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # [B, 9]
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Evaluate predictions
        for idx, prob_vec in enumerate(probs):
            true_cls = true_labels[idx]
            if true_cls not in SODA_CLASSES:
                continue
            true_idx = SODA_CLASSES.index(true_cls)

            top5_idx = np.argsort(prob_vec)[-5:][::-1]
            top1_idx = top5_idx[0]
            conf = float(prob_vec[true_idx])

            correct1 += int(top1_idx == true_idx)
            correct5 += int(true_idx in top5_idx)
            total_conf += conf

            preds.append({
                'image_id': batch.iloc[valid_idxs[idx]]['original_image_id'],
                'filename': batch.iloc[valid_idxs[idx]]['object_image_filename'],
                'size_category': size_category,
                'true_class': true_cls,
                'pred_class': SODA_CLASSES[top1_idx],
                'confidence': conf,
                'top1_correct': top1_idx == true_idx,
                'top5_correct': true_idx in top5_idx,
                'top5_classes': [SODA_CLASSES[i] for i in top5_idx],
                'top5_probs': prob_vec[top5_idx].tolist()
            })

    n = len(preds)
    if n > 0:
        summary[size_category] = {
            'samples': n,
            'top1_accuracy': correct1 / n,
            'top5_accuracy': correct5 / n,
            'mean_confidence': total_conf / n
        }
        print(f"{size_category.capitalize()} → N={n}, Top-1={summary[size_category]['top1_accuracy']:.3f}, "
              f"Top-5={summary[size_category]['top5_accuracy']:.3f}, "
              f"MeanConf={summary[size_category]['mean_confidence']:.3f}")

    results.extend(preds)

# Final summary display
print("\n" + "="*40)
print("RESULTS BY SIZE CATEGORY")
for sz in ['tiny','small','medium','large','huge']:
    if sz in summary:
        m = summary[sz]
        print(f"{sz.upper():>6}: Top-1={m['top1_accuracy']:.3f}, Top-5={m['top5_accuracy']:.3f}, N={m['samples']}")

# Save detailed and summary CSVs
pd.DataFrame(results).to_csv('clip_soda_detailed.csv', index=False)
pd.DataFrame([
    {'size_category': sz, **metrics}
    for sz, metrics in summary.items()
]).to_csv('clip_soda_summary.csv', index=False)

print("\nSaved clip_soda_detailed.csv and clip_soda_summary.csv")
