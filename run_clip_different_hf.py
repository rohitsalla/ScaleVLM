import os
import csv
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm

class HFCLIPObjectSizeBenchmark:
    def __init__(self,
                 model_name="openai/clip-vit-base-patch32",
                 device=None):
        """
        Initialize CLIP benchmark using Hugging Face Transformers for SODA-A (9 classes).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name} on {self.device}")
        
        # Load CLIP model & processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        
        # SODA-A classes (9)
        self.soda_classes = [
            "Car", "Truck", "Van", "Bus",
            "Cyclist", "Tricycle", "Motor", "Person", "Others"
        ]
        # Build prompts
        self.prompts = [f"a photo of a {c.lower()}" for c in self.soda_classes]
        self.text_embeddings = self._compute_text_embeddings()
    
    def _compute_text_embeddings(self):
        """Compute & normalize text embeddings for SODA-A classes."""
        inputs = self.processor(
            text=self.prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            txt_feats = self.model.get_text_features(**inputs)
        return txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    
    def predict(self, image_path):
        """
        Predict class for a single image crop.
        Returns: (pred_class, confidence, top3_predictions, all_probs)
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_feats = self.model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        sims = (img_feats @ self.text_embeddings.T).squeeze(0)
        probs = sims.softmax(dim=0).cpu().numpy()
        
        # Top-3 for brevity
        topk = np.argsort(probs)[-3:][::-1]
        top3 = [(self.soda_classes[i], float(probs[i])) for i in topk]
        pred_idx = int(topk[0])
        return self.soda_classes[pred_idx], top3[0][1], top3, probs
    
    def benchmark_dataset(self,
                          csv_path,
                          output_dir="hf_clip_benchmark_soda"):
        """
        Benchmark CLIP on SODA-A object crops from CSV.
        CSV must include columns:
          original_image_id, object_image_filename, object_image_path,
          dataset, class_name, size_category, bbox_area, relative_area_percent
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} object images from {csv_path}")
        
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIP SODA-A"):
            img_p = row["object_image_path"]
            if not os.path.exists(img_p):
                continue
            true_cls = row["class_name"]
            size_cat = row["size_category"]
            
            pred_cls, conf, top3, _ = self.predict(img_p)
            correct = (pred_cls == true_cls)
            
            rec = {
                "orig_image_id": row["original_image_id"],
                "filename": row["object_image_filename"],
                "split": row["dataset"],
                "true_class": true_cls,
                "pred_class": pred_cls,
                "size_category": size_cat,
                "confidence": conf,
                "is_correct": correct
            }
            for i, (c,p) in enumerate(top3, 1):
                rec[f"top{i}_class"] = c
                rec[f"top{i}_prob"]  = p
            records.append(rec)
        
        res_df = pd.DataFrame(records)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = self.model_name.replace("/", "_")
        out_csv = os.path.join(output_dir, f"soda_results_{safe}_{ts}.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"Saved detailed results to {out_csv}")
        
        self._analyze(res_df, output_dir, safe, ts)
        return res_df
    
    def _analyze(self, df, output_dir, model_safe, ts):
        """Compute metrics and save summary & plot."""
        total = len(df)
        acc = df["is_correct"].mean()
        print(f"\nOverall Top-1 Accuracy: {acc:.3f} on {total} samples")
        
        # By size category
        size_stats = df.groupby("size_category")["is_correct"].agg(["count","mean"])
        size_stats = size_stats.rename(columns={"mean":"accuracy"}).round(3)
        print("\nAccuracy by size:\n", size_stats)
        
        # By class
        class_stats = df.groupby("true_class")["is_correct"].agg(["count","mean"])
        class_stats = class_stats[class_stats["count"]>=5]
        class_stats = class_stats.rename(columns={"mean":"accuracy"}).round(3)
        print("\nAccuracy by class (>=5 samples):\n", class_stats)
        
        # Save summary JSON
        summary = {
            "model": self.model_name,
            "timestamp": ts,
            "overall_accuracy": float(acc),
            "by_size": size_stats.to_dict(),
            "by_class": class_stats.to_dict()
        }
        summary_path = os.path.join(output_dir, f"soda_summary_{model_safe}_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")
        
        # Plot accuracy by size
        plt.figure(figsize=(6,4))
        size_stats["accuracy"].reindex(["tiny","small","medium","large","huge"], fill_value=0).plot.bar(
            color=["red","orange","gold","green","blue"])
        plt.title("Top-1 Accuracy by Size")
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"soda_acc_size_{model_safe}_{ts}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved plot to {plot_path}")

def main():
    CSV   = "soda_extracted_objects/soda_individual_objects.csv"
    OUT   = "hf_clip_benchmark_soda"
    bench = HFCLIPObjectSizeBenchmark()
    bench.benchmark_dataset(CSV, OUT)

if __name__ == "__main__":
    main()
