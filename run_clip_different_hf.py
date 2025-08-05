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
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """Initialize CLIP benchmark for SODA-A (9 classes)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name} on {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        
        # SODA-A classes (9)
        self.soda_classes = [
            "Car", "Truck", "Van", "Bus",
            "Cyclist", "Tricycle", "Motor", "Person", "Others"
        ]
        self.prompts = [f"a photo of a {c.lower()}" for c in self.soda_classes]
        self.text_embeddings = self._compute_text_embeddings()
    
    def _compute_text_embeddings(self):
        """Compute & normalize text embeddings for SODA-A classes."""
        inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            txt_feats = self.model.get_text_features(**inputs)
        return txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    
    def predict(self, image_path):
        """Predict class for a single image crop."""
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_feats = self.model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        sims = (img_feats @ self.text_embeddings.T).squeeze(0)
        probs = sims.softmax(dim=0).cpu().numpy()
        
        topk = np.argsort(probs)[-3:][::-1]
        top3 = [(self.soda_classes[i], float(probs[i])) for i in topk]
        pred_idx = int(topk[0])
        return self.soda_classes[pred_idx], top3[0][1], top3, probs
    
    def benchmark_dataset(self, csv_path, output_dir="hf_clip_benchmark_soda"):
        """Benchmark CLIP on SODA-A object crops."""
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} object images from {csv_path}")
        
        # Validate required columns
        required_cols = ['object_image_path', 'class_name', 'size_category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        records = []
        processed = 0
        errors = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIP SODA-A"):
            img_path = row["object_image_path"]
            
            if not os.path.exists(img_path):
                errors += 1
                continue
                
            try:
                true_cls = row["class_name"]
                size_cat = row["size_category"]
                
                pred_cls, conf, top3, _ = self.predict(img_path)
                correct = (pred_cls == true_cls)
                processed += 1
                
                rec = {
                    "orig_image_id": row.get("original_image_id", "unknown"),
                    "filename": row.get("object_image_filename", "unknown"),
                    "split": row.get("split", row.get("dataset", "unknown")),
                    "true_class": true_cls,
                    "pred_class": pred_cls,
                    "size_category": size_cat,
                    "confidence": conf,
                    "is_correct": correct,
                    "poly_area": row.get("poly_area", 0),
                    "relative_pct": row.get("relative_area_percent", 0)
                }
                for i, (c, p) in enumerate(top3, 1):
                    rec[f"top{i}_class"] = c
                    rec[f"top{i}_prob"] = p
                records.append(rec)
                
            except Exception as e:
                errors += 1
                continue
        
        print(f"✅ Processed: {processed}, ❌ Errors: {errors}")
        
        if not records:
            print("❌ No valid records generated")
            return None
        
        # Save results
        res_df = pd.DataFrame(records)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = self.model_name.replace("/", "_")
        out_csv = os.path.join(output_dir, f"soda_results_{safe}_{ts}.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"Saved detailed results to {out_csv}")
        
        self._analyze(res_df, output_dir, safe, ts)
        return res_df
    
    def _analyze(self, df, output_dir, model_safe, ts):
        """Compute metrics and save summary & visualizations."""
        total = len(df)
        acc = df["is_correct"].mean()
        print(f"\n{'='*50}")
        print(f"CLIP SODA-A BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"Total samples: {total}")
        print(f"Overall Top-1 Accuracy: {acc:.3f} ({acc*100:.1f}%)")
        
        # By size category
        print(f"\n📏 ACCURACY BY SIZE CATEGORY:")
        size_stats = df.groupby("size_category")["is_correct"].agg(["count","mean"])
        size_stats.columns = ["Count", "Accuracy"]
        size_stats = size_stats.round(3)
        
        size_order = ["tiny", "small", "medium", "large", "huge"]
        for size in size_order:
            if size in size_stats.index:
                count = size_stats.loc[size, "Count"]
                accuracy = size_stats.loc[size, "Accuracy"]
                print(f"  {size.upper():>6}: {accuracy:.3f} ({accuracy*100:.1f}%) - {count} samples")
        
        # By class
        print(f"\n🏷️  ACCURACY BY CLASS:")
        class_stats = df.groupby("true_class")["is_correct"].agg(["count","mean"])
        class_stats.columns = ["Count", "Accuracy"]
        class_stats = class_stats[class_stats["Count"] >= 5].sort_values("Accuracy", ascending=False)
        class_stats = class_stats.round(3)
        
        for cls, row in class_stats.iterrows():
            print(f"  {cls:>12}: {row['Accuracy']:.3f} ({row['Accuracy']*100:.1f}%) - {row['Count']} samples")
        
        # By split
        if "split" in df.columns:
            print(f"\n📂 ACCURACY BY SPLIT:")
            split_stats = df.groupby("split")["is_correct"].agg(["count","mean"]).round(3)
            split_stats.columns = ["Count", "Accuracy"]
            for split, row in split_stats.iterrows():
                print(f"  {split:>8}: {row['Accuracy']:.3f} - {row['Count']} samples")
        
        # Save summary JSON
        summary = {
            "model": self.model_name,
            "timestamp": ts,
            "total_samples": total,
            "overall_accuracy": float(acc),
            "by_size": size_stats.to_dict(),
            "by_class": class_stats.to_dict()
        }
        
        summary_path = os.path.join(output_dir, f"soda_summary_{model_safe}_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n💾 Saved summary to {summary_path}")
        
        # Create visualization
        self._create_plots(df, output_dir, model_safe, ts)
    
    def _create_plots(self, df, output_dir, model_safe, ts):
        """Create accuracy visualization plots."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Accuracy by size
        ax1 = axes[0]
        size_acc = df.groupby("size_category")["is_correct"].mean()
        size_counts = df.groupby("size_category").size()
        
        size_order = ["tiny", "small", "medium", "large", "huge"]
        colors = ["red", "orange", "gold", "green", "blue"]
        
        ordered_acc = [size_acc.get(s, 0) for s in size_order if s in size_acc.index]
        ordered_counts = [size_counts.get(s, 0) for s in size_order if s in size_counts.index]
        ordered_labels = [s for s in size_order if s in size_acc.index]
        ordered_colors = [colors[size_order.index(s)] for s in ordered_labels]
        
        bars = ax1.bar(range(len(ordered_acc)), ordered_acc, color=ordered_colors)
        ax1.set_xlabel("Object Size Category")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy by Object Size")
        ax1.set_xticks(range(len(ordered_labels)))
        ax1.set_xticklabels(ordered_labels, rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add count labels
        for bar, count in zip(bars, ordered_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Accuracy by class
        ax2 = axes[1]
        class_acc = df.groupby("true_class")["is_correct"].mean()
        class_counts = df.groupby("true_class").size()
        
        # Show all classes
        bars = ax2.bar(range(len(class_acc)), class_acc.values)
        ax2.set_xlabel("Object Class")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy by Object Class")
        ax2.set_xticks(range(len(class_acc)))
        ax2.set_xticklabels(class_acc.index, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"soda_analysis_{model_safe}_{ts}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved plots to {plot_path}")

def main():
    # Updated CSV path to match the object extraction output
    CSV_PATH = "/home/wirin/Ashish/VLM/data/soda_object_crops/soda_individual_objects.csv"
    OUTPUT_DIR = "/home/wirin/Ashish/VLM/results/hf_clip_benchmark_soda"
    
    print("🚀 Starting CLIP SODA-A benchmark...")
    print(f"📄 Using CSV: {CSV_PATH}")
    print(f"💾 Output dir: {OUTPUT_DIR}")
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        print("First run the object extraction script to create individual crops.")
        return
    
    bench = HFCLIPObjectSizeBenchmark()
    results = bench.benchmark_dataset(CSV_PATH, OUTPUT_DIR)
    
    if results is not None:
        print("✅ Benchmark complete!")
    else:
        print("❌ Benchmark failed. Check CSV format and image paths.")

if __name__ == "__main__":
    main()


