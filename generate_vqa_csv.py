import pandas as pd
import numpy as np
import json
import random
from collections import defaultdict
import os

class SODAVQAConverter:
    def __init__(self, csv_path):
        """
        Initialize VQA converter with CSV data from SODA-A extraction.
        
        Assumes CSV has columns:
          image_id, split, object_idx, class_name, size_category,
          bbox_coords, relative_area_percent, object_image_path
        """
        self.df = pd.read_csv(csv_path)
        # base paths: extracted object crops for each split
        self.base_paths = {
            'train': None,   # not used, paths are absolute in CSV
            'val':   None,
            'test':  None
        }
    
    def generate_positive_questions(self):
        """
        Generate one positive VQA question per object-crop.
        (i.e. “Is there a <class> in this image?” → Yes)
        """
        questions = []
        for _, row in self.df.iterrows():
            img_path = row['object_image_path']
            cls     = row['class_name']
            size    = row['size_category']
            split   = row['split']
            img_id  = f"{row['image_id']}_{row['object_idx']}"
            
            q = f"Is there a {cls} in this image?"
            questions.append({
                'question_id': img_id,
                'image_path': img_path,
                'split': split,
                'question': q,
                'answer': 'Yes',
                'target_class': cls,
                'target_size': size,
                'question_type': 'presence_positive'
            })
        random.shuffle(questions)
        return questions

    def generate_negative_questions(self):
        """
        For each object-crop, sample one random other class → ask absence.
        (i.e. “Is there a <other_class> in this image?” → No)
        """
        questions = []
        classes = self.df['class_name'].unique().tolist()
        
        for _, row in self.df.iterrows():
            img_path = row['object_image_path']
            cls     = row['class_name']
            size    = row['size_category']
            split   = row['split']
            img_id  = f"{row['image_id']}_{row['object_idx']}"
            
            # pick a different class
            neg_cls = random.choice([c for c in classes if c != cls])
            q = f"Is there a {neg_cls} in this image?"
            questions.append({
                'question_id': img_id + "_neg",
                'image_path': img_path,
                'split': split,
                'question': q,
                'answer': 'No',
                'target_class': neg_cls,
                'target_size': None,
                'question_type': 'presence_negative'
            })
        random.shuffle(questions)
        return questions

    def save(self, questions, output_path, fmt='json'):
        """Save questions list to JSON or CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if fmt == 'json':
            with open(output_path, 'w') as f:
                json.dump(questions, f, indent=2)
        elif fmt == 'csv':
            pd.DataFrame(questions).to_csv(output_path, index=False)
        print(f"Saved {len(questions)} questions to {output_path}")

    def analyze(self, questions):
        """Print summary statistics."""
        df = pd.DataFrame(questions)
        total = len(df)
        print(f"\nTOTAL QUESTIONS: {total}")
        
        # answer dist
        ans_dist = df['answer'].value_counts(normalize=True) * 100
        print("\nANSWER DIST (%):")
        for a, p in ans_dist.items():
            print(f"  {a}: {p:.1f}%")
        
        # split dist
        sp_dist = df['split'].value_counts(normalize=True) * 100
        print("\nSPLIT DIST (%):")
        for s, p in sp_dist.items():
            print(f"  {s}: {p:.1f}%")
        
        # class dist
        cls_dist = df['target_class'].value_counts(normalize=True) * 100
        print("\nTARGET CLASS DIST (%):")
        for c, p in cls_dist.items():
            print(f"  {c}: {p:.1f}%")
        
        # size dist (positives only)
        sz_dist = df[df['answer']=='Yes']['target_size'].value_counts(normalize=True) * 100
        print("\nSIZE CATEGORY DIST (positives, %):")
        for sz, p in sz_dist.items():
            print(f"  {sz}: {p:.1f}%")

def main():
    CSV_PATH  = "/data/rohit/ScaleVLM-master/data/soda_extracted_objects/soda_individual_objects.csv"
    OUT_DIR   = "/data/rohit/ScaleVLM-master/data/vqa_soda"
    os.makedirs(OUT_DIR, exist_ok=True)

    converter = SODAVQAConverter(CSV_PATH)
    
    pos_qs = converter.generate_positive_questions()
    neg_qs = converter.generate_negative_questions()
    all_qs = pos_qs + neg_qs
    
    converter.save(pos_qs, os.path.join(OUT_DIR, "soda_pos_questions.json"), fmt='json')
    converter.save(neg_qs, os.path.join(OUT_DIR, "soda_neg_questions.json"), fmt='json')
    converter.save(all_qs, os.path.join(OUT_DIR, "soda_all_questions.csv"), fmt='csv')
    
    converter.analyze(all_qs)
    print("\nVQA dataset for SODA-A ready.")

if __name__ == "__main__":
    main()
