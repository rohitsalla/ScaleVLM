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
        Initialize VLM benchmark:
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
        print("Model loaded.")

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
            print(f"Error on {image_path}: {e}")
            return "error"

    def normalize_answer(self, answer):
        """Map raw model text to yes/no/unknown."""
        if not answer or answer == "error":
            return "unknown"
        a = answer.lower().strip()
        a = re.sub(r'^(answer:|the answer is|yes,|no,)\s*', "", a)
        yes_kw = ["yes","true","correct","present","visible","there is"]
        no_kw  = ["no","false","absent","not","cannot","can't","don't see"]
        if any(w in a for w in yes_kw):
            return "yes"
        if any(w in a for w in no_kw):
            return "no"
        if a in ["yes","y","1","true"]:
            return "yes"
        if a in ["no","n","0","false"]:
            return "no"
        return "unknown"

    def evaluate_vqa_dataset(self, vqa_json, output_dir):
        """
        Evaluate the VQA JSON:
        expects fields: image_id, image_path, question, answer, target_class, target_size, question_type
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(vqa_json) as f:
            qs = json.load(f)
        print(f"Loaded {len(qs)} questions")
        results, correct, missing = [], 0, 0

        for i, q in enumerate(tqdm(qs, desc="Benchmarking")):
            img_path = q["image_path"]
            if not os.path.exists(img_path):
                missing += 1
                continue
            pred = self.answer_vqa_question(img_path, q["question"])
            pred_n = self.normalize_answer(pred)
            gt = q["answer"].lower()
            ok = (pred_n == gt)
            if ok: correct += 1

            results.append({
                "image_id":      q["image_id"],
                "split":         q.get("dataset",""),
                "question":      q["question"],
                "ground_truth":  q["answer"],
                "pred_raw":      pred,
                "pred_norm":     pred_n,
                "correct":       ok,
                "target_class":  q["target_class"],
                "target_size":   q.get("target_size",""),
                "question_type": q.get("question_type","")
            })
            if (i+1)%100==0:
                acc = correct/len(results)
                print(f"[{i+1}/{len(qs)}] Acc so far: {acc:.3f}")

        print(f"Done. {len(results)} processed, {missing} missing.")
        df = pd.DataFrame(results)
        # save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = self.model_name.replace("/","_")
        out_csv = os.path.join(output_dir, f"results_{safe}_{ts}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")
        # analyze
        self._analyze(df, output_dir, safe, ts)
        return df

    def _analyze(self, df, output_dir, model_safe, ts):
        """Print and plot key metrics."""
        total = len(df)
        acc = df["correct"].mean()
        print(f"\nOverall Accuracy: {acc:.3f}")
        # by size
        print("\nAccuracy by size:")
        for sz, grp in df.groupby("target_size"):
            print(f"  {sz}: {grp['correct'].mean():.3f} ({len(grp)})")
        # by class
        print("\nAccuracy by class (min 5 samples):")
        cls = df.groupby("target_class").filter(lambda x: len(x)>=5)
        for c, grp in cls.groupby("target_class"):
            print(f"  {c}: {grp['correct'].mean():.3f} ({len(grp)})")
        # by answer
        print("\nBy answer type:")
        print(df.groupby("ground_truth")["correct"].mean())
        # save summary
        summary = {
            "model": self.model_name,
            "timestamp": ts,
            "overall_acc": acc,
            "by_size": df.groupby("target_size")["correct"].mean().to_dict(),
            "by_answer": df.groupby("ground_truth")["correct"].mean().to_dict()
        }
        with open(os.path.join(output_dir, f"summary_{model_safe}_{ts}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        # optional: plot size-accuracy bar
        plt.figure(figsize=(6,4))
        df.groupby("target_size")["correct"].mean().reindex(["tiny","small","medium","large","huge"]).plot.bar(color=["red","orange","gold","green","blue"])
        plt.ylim(0,1); plt.title("Acc by Size"); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"plot_size_{model_safe}_{ts}.png"), dpi=200)
        plt.close()

def main():
    MODEL   = "Salesforce/blip-vqa-base"
    VQA_JSON= "/data/.../simple_vqa_questions.json"
    OUT_DIR = "/data/.../vlm_benchmark"
    bench = GenerativeVLMBenchmark(model_name=MODEL)
    bench.evaluate_vqa_dataset(VQA_JSON, OUT_DIR)

if __name__=="__main__":
    main()
