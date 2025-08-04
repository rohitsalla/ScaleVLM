#  ScaleVLM

This repository contains code for benchmarking **Vision-Language Models (VLMs)** on their performance across **object sizes** — categorized as **tiny, small, medium,  large or huge** — using object detection annotations (e.g., from PASCAL VOC). The benchmark evaluates whether VLMs can accurately understand and describe objects of different scales.

---

## 🎯 Goal

Quantify how VLM performance (e.g., CLIP, BLIP, etc.) varies depending on **object size** within an image. This allows deeper insights into how well VLMs understand small or partially visible objects.

---

## 🧩 Folder Contents

### 📦 Preprocessing & Annotation

- `extract_seperate_bbox_pascal.py`  
  Extracts individual bounding boxes from Pascal VOC `.xml` annotation files. This approach extracts seperate image for seperate instance where all other objects are masked 

- `categorise_size.py`  
  Categorizes each object as `tiny`, `small`, `medium`,  `large` or `huge` based on the percentage area of its bounding box relative to the full image. If multiple objects are present, this falls back to the one of the larger size

- `bbox_single_image.py`  
  Visualizes and tests bounding box extraction and size categorization on a single image.

---

### ⚙️ Benchmark & Evaluation

- `run_clip_pascal_voc.py`  
  Evaluates CLIP’s classification ability on PASCAL VOC using extracted objects.

- `run_clip_different_hf.py`  
  Evaluates alternative CLIP variants from Hugging Face on the same task.

- `generate_vqa_csv.py`  
  Converts the object-level data into a VQA-style format for text–image prompting.

- `generative_vlm_benchmark.py`  
  Benchmarks generative VLMs (like BLIP) on the same object-level tasks with prompts.

---

## 📊 Object Size Categories

| Category | Area (% of image) |
|----------|-------------------|
| Tiny     | < 1%              |
| Small    | 1% – 5%           |
| Medium   | 5% – 15%          |
| Large    | > 15%             |

These thresholds can be customized in `categorise_size.py`.

## 🚀 How to Use

### 1️⃣ Extract and Categorize Objects

```bash
python extract_seperate_bbox_pascal.py
python categorise_size.py
# ScaleVLM
