import json
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

class GenerativeVLMBenchmark:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device=None):
        """
        Initialize VLM benchmark for SODA-A VQA evaluation:
        - BLIP VQA: "Salesforce/blip-vqa-base" or "Salesforce/blip-vqa-capfilt-large"
        - GIT: "microsoft/git-base-vqav2"
        - ViLT: "dandelin/vilt-b32-finetuned-vqa"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Loading VLM: {model_name} on {self.device}")
        
        if "blip" in model_name.lower():
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        self.model.eval()
        print("✅ Model loaded successfully.")

    def answer_vqa_question(self, image_path, question):
        """Run VLM to answer a single question on an image."""
        try:
            img = Image.open(image_path).convert("RGB")
            if "blip" in self.model_name.lower():
                inputs = self.processor(img, question, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outs = self.model.generate(**inputs, max_length=20, num_beams=5)
                ans = self.processor.decode(outs[0], skip_special_tokens=True)
            else:
                inputs = self.processor(images=img, text=question, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outs = self.model.generate(**inputs, max_length=20, num_beams=5)
                ans = self.processor.decode(outs[0], skip_special_tokens=True)
            return ans.strip()
        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            return "error"

    def normalize_answer(self, answer):
        """Map raw VLM response to standardized yes/no/unknown format."""
        if not answer or answer == "error":
            return "unknown"
        
        a = answer.lower().strip()
        # Remove common VLM prefixes
        a = re.sub(r'^(answer:|the answer is|yes,|no,)\s*', "", a)
        
        # Yes patterns
        yes_patterns = ["yes", "true", "correct", "present", "visible", "there is", "i can see"]
        if any(pattern in a for pattern in yes_patterns):
            return "yes"
        
        # No patterns  
        no_patterns = ["no", "false", "absent", "not", "cannot", "can't", "don't see", "not visible"]
        if any(pattern in a for pattern in no_patterns):
            return "no"
        
        # Direct matches
        if a in ["yes", "y", "1", "true"]:
            return "yes"
        if a in ["no", "n", "0", "false"]:
            return "no"
        
        return "unknown"

    def evaluate_vqa_dataset(self, vqa_json, output_dir):
        """
        Evaluate VLM on SODA-A VQA dataset.
        Expected JSON fields: question_id, image_path, question, answer, target_class, target_size, question_type, split
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load VQA questions
        if not os.path.exists(vqa_json):
            print(f"❌ VQA file not found: {vqa_json}")
            return None
            
        with open(vqa_json) as f:
            questions = json.load(f)
        
        print(f"📄 Loaded {len(questions)} VQA questions")
        
        results = []
        correct = 0
        missing_images = 0
        
        for i, q in enumerate(tqdm(questions, desc="VLM Evaluation")):
            img_path = q["image_path"]
            
            # Check if image exists
            if not os.path.exists(img_path):
                missing_images += 1
                if missing_images <= 5:  # Show first 5 missing images
                    print(f"⚠️ Missing: {img_path}")
                continue
            
            # Get VLM prediction
            pred_raw = self.answer_vqa_question(img_path, q["question"])
            pred_norm = self.normalize_answer(pred_raw)
            gt = q["answer"].lower()
            
            is_correct = (pred_norm == gt)
            if is_correct:
                correct += 1
            
            # Store result
            results.append({
                "question_id": q.get("question_id", f"q_{i}"),
                "image_path": img_path,
                "split": q.get("split", "unknown"),
                "question": q["question"],
                "ground_truth": q["answer"],
                "pred_raw": pred_raw,
                "pred_normalized": pred_norm,
                "correct": is_correct,
                "target_class": q.get("target_class", "unknown"),
                "target_size": q.get("target_size", "unknown"),
                "question_type": q.get("question_type", "unknown")
            })
            
            # Progress update
            if (i + 1) % 100 == 0:
                current_acc = correct / len(results) if results else 0
                print(f"[{i+1}/{len(questions)}] Current accuracy: {current_acc:.3f}")
        
        if missing_images > 5:
            print(f"⚠️ Total missing images: {missing_images}")
        
        if not results:
            print("❌ No valid results generated")
            return None
        
        print(f"✅ Processed {len(results)} questions successfully")
        
        # Save results
        df = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = self.model_name.replace("/", "_").replace("-", "_")
        out_csv = os.path.join(output_dir, f"vlm_results_{safe}_{ts}.csv")
        df.to_csv(out_csv, index=False)
        print(f"💾 Detailed results saved to {out_csv}")
        
        # Analyze results
        self._analyze(df, output_dir, safe, ts)
        return df

    def _analyze(self, df, output_dir, model_safe, ts):
        """Analyze and visualize VLM performance metrics."""
        total = len(df)
        overall_acc = df["correct"].mean()
        
        print(f"\n{'='*60}")
        print(f"SODA-A VLM BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Total questions: {total:,}")
        print(f"Overall accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")
        
        # Accuracy by size category
        print(f"\n📏 ACCURACY BY SIZE CATEGORY:")
        size_stats = df.groupby("target_size")["correct"].agg(["count", "mean"]).round(3)
        size_stats.columns = ["Count", "Accuracy"]
        
        size_order = ["tiny", "small", "medium", "large", "huge"]
        for size in size_order:
            if size in size_stats.index:
                count = size_stats.loc[size, "Count"]
                acc = size_stats.loc[size, "Accuracy"]
                print(f"  {size.upper():>6}: {acc:.3f} ({acc*100:.1f}%) - {count:,} questions")
        
        # Accuracy by class
        print(f"\n🏷️ ACCURACY BY CLASS:")
        class_stats = df.groupby("target_class")["correct"].agg(["count", "mean"])
        class_stats = class_stats[class_stats["count"] >= 10].round(3)  # Min 10 samples
        class_stats.columns = ["Count", "Accuracy"]
        class_stats = class_stats.sort_values("Accuracy", ascending=False)
        
        for cls, row in class_stats.iterrows():
            print(f"  {cls:>12}: {row['Accuracy']:.3f} ({row['Accuracy']*100:.1f}%) - {row['Count']:,} questions")
        
        # Accuracy by answer type
        print(f"\n❓ ACCURACY BY ANSWER TYPE:")
        answer_stats = df.groupby("ground_truth")["correct"].agg(["count", "mean"]).round(3)
        answer_stats.columns = ["Count", "Accuracy"]
        for answer, row in answer_stats.iterrows():
            print(f"  {answer.upper():>3}: {row['Accuracy']:.3f} ({row['Accuracy']*100:.1f}%) - {row['Count']:,} questions")
        
        # Accuracy by question type
        print(f"\n🔍 ACCURACY BY QUESTION TYPE:")
        qtype_stats = df.groupby("question_type")["correct"].agg(["count", "mean"]).round(3)
        qtype_stats.columns = ["Count", "Accuracy"]
        for qtype, row in qtype_stats.iterrows():
            print(f"  {qtype:>20}: {row['Accuracy']:.3f} - {row['Count']:,} questions")
        
        # Save summary
        summary = {
            "model": self.model_name,
            "timestamp": ts,
            "total_questions": total,
            "overall_accuracy": float(overall_acc),
            "by_size": size_stats.to_dict(),
            "by_class": class_stats.to_dict(),
            "by_answer": answer_stats.to_dict(),
            "by_question_type": qtype_stats.to_dict()
        }
        
        summary_path = os.path.join(output_dir, f"vlm_summary_{model_safe}_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📊 Summary saved to {summary_path}")
        
        # Create visualization
        self._create_plots(df, output_dir, model_safe, ts)

    def _create_plots(self, df, output_dir, model_safe, ts):
        """Create accuracy visualization plots."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Accuracy by size
        ax1 = axes[0]
        size_acc = df.groupby("target_size")["correct"].mean()
        size_order = ["tiny", "small", "medium", "large", "huge"]
        colors = ["red", "orange", "gold", "green", "blue"]
        
        ordered_data = [(s, size_acc.get(s, 0)) for s in size_order if s in size_acc.index]
        if ordered_data:
            labels, values = zip(*ordered_data)
            colors_used = [colors[size_order.index(s)] for s in labels]
            ax1.bar(labels, values, color=colors_used)
        
        ax1.set_title("Accuracy by Object Size")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        
        # Plot 2: Accuracy by class
        ax2 = axes[1]
        class_acc = df.groupby("target_class")["correct"].mean().sort_values(ascending=False)
        ax2.bar(range(len(class_acc)), class_acc.values)
        ax2.set_title("Accuracy by Object Class")
        ax2.set_ylabel("Accuracy")
        ax2.set_xticks(range(len(class_acc)))
        ax2.set_xticklabels(class_acc.index, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Plot 3: Answer type accuracy
        ax3 = axes[2]
        answer_acc = df.groupby("ground_truth")["correct"].mean()
        colors_answer = ['lightgreen' if ans == 'Yes' else 'lightcoral' for ans in answer_acc.index]
        ax3.bar(answer_acc.index, answer_acc.values, color=colors_answer)
        ax3.set_title("Accuracy by Answer Type")
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"vlm_analysis_{model_safe}_{ts}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📈 Plots saved to {plot_path}")

def main():
    # Updated paths for SODA-A VQA evaluation
    MODEL_NAME = "Salesforce/blip-vqa-base"
    VQA_JSON = "/home/wirin/Ashish/VLM/data/vqa_soda/soda_all_questions.json"
    OUTPUT_DIR = "/home/wirin/Ashish/VLM/data/results/vlm_benchmark_soda"
    
    print("🚀 Starting SODA-A VLM benchmark...")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"📄 VQA file: {VQA_JSON}")
    print(f"💾 Output: {OUTPUT_DIR}")
    
    # Initialize and run benchmark
    benchmark = GenerativeVLMBenchmark(model_name=MODEL_NAME)
    results = benchmark.evaluate_vqa_dataset(VQA_JSON, OUTPUT_DIR)
    
    if results is not None:
        print("\n✅ VLM benchmark completed successfully!")
    else:
        print("\n❌ VLM benchmark failed. Check file paths and format.")

if __name__ == "__main__":
    main()



