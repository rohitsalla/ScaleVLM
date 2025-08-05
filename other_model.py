import json
import pandas as pd
import torch
from PIL import Image
import os
import re
from datetime import datetime
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    BlipProcessor, Blip2Processor, Blip2ForConditionalGeneration
)
from tqdm import tqdm

VQA_JSON = "/home/wirin/Ashish/VLM/data/vqa_soda/soda_all_questions.json"
OUTPUT_DIR = "/home/wirin/Ashish/VLM/data/results/vlm_benchmark_soda"

MODEL_LIST = [
    "Salesforce/blip2-opt-2.7b",
    "liuhaotian/llava-v1.5-7b",
    "microsoft/mplug-owl",
    "HuggingFaceM4/idefics2-8b-base",
    "internlm/internvl-chat-v1-7b",
    "Qwen/Qwen-VL-Chat",
]

class GeneralVLMBenchmark:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading model: {model_name} on {self.device}")

        if "blip2" in model_name.lower():
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if self.device=="cuda" else torch.float32).to(self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device=="cuda" else torch.float32).to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def answer_vqa(self, image_path, question):
        try:
            image = Image.open(image_path).convert("RGB")
            if hasattr(self.processor, "__call__"):
                if isinstance(self.processor, (Blip2Processor, BlipProcessor)):
                    inputs = self.processor(image, question, return_tensors="pt")
                else:
                    inputs = self.processor(images=image, text=question, return_tensors="pt")
            else:
                inputs = self.processor(images=image, text=question, return_tensors="pt")
            # SAFEGUARD: Ensure input tokens are non-empty before generation
            if "input_ids" in inputs and inputs["input_ids"].shape[1] > 0:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=50)
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                return answer.strip()
            else:
                print(f"Empty input tokens, skipping generation for {image_path} and question '{question}'")
                return ""
        except Exception as e:
            print(f"Error on {image_path}: {e}")
            return "error"

    def normalize_answer(self, answer):
        if not answer or answer == "error":
            return "unknown"
        a = answer.lower().strip()
        a = re.sub(r'^(answer:|the answer is|yes,|no,)\s*', "", a)
        yes_kw = ["yes", "true", "correct", "present", "visible", "there is", "i can see"]
        no_kw  = ["no", "false", "absent", "not", "cannot", "can't", "don't see", "not visible"]
        if any(w in a for w in yes_kw):
            return "yes"
        if any(w in a for w in no_kw):
            return "no"
        if a in ["yes", "y", "1", "true"]:
            return "yes"
        if a in ["no", "n", "0", "false"]:
            return "no"
        return "unknown"

    def benchmark(self, vqa_json, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(vqa_json) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} SODA-A VQA questions")
        results, correct = [], 0
        for i, q in enumerate(tqdm(questions, desc=self.model_name)):
            img_path = q["image_path"]
            if not os.path.exists(img_path):
                continue
            pred = self.answer_vqa(img_path, q["question"])
            pred_norm = self.normalize_answer(pred)
            gt = q["answer"].lower()
            is_correct = (pred_norm == gt)
            if is_correct: correct += 1
            results.append({
                "image_id": q.get("question_id", f"q_{i}"),
                "question": q["question"],
                "gt": gt,
                "pred": pred,
                "pred_norm": pred_norm,
                "correct": is_correct,
                "target_class": q.get("target_class"),
                "target_size": q.get("target_size")
            })
        df = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.model_name.replace("/", "_")
        out_csv = os.path.join(output_dir, f"soda_vqa_{safe_name}_{ts}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Results saved: {out_csv}")
        print(f"Top-1 accuracy: {df['correct'].mean() if len(df)>0 else 0:.3f}")

if __name__ == "__main__":
    for model in MODEL_LIST:
        try:
            bench = GeneralVLMBenchmark(model)
            bench.benchmark(VQA_JSON, OUTPUT_DIR)
        except Exception as ex:
            print(f"Skipping {model} due to error: {ex}")


