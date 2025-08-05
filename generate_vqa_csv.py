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
        
        Expected CSV columns:
          original_image_id, object_image_filename, object_image_path,
          split, class_id, class_name, size_category, poly_area, relative_area_percent
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} object records from CSV")
        
        # Validate required columns
        required_cols = ['object_image_path', 'class_name', 'size_category', 'split']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            print(f"Available columns: {list(self.df.columns)}")
    
    def generate_positive_questions(self):
        """Generate positive VQA questions: "Is there a <class> in this image?" → Yes"""
        questions = []
        for _, row in self.df.iterrows():
            img_path = row['object_image_path']
            cls = row['class_name']
            size = row['size_category']
            split = row['split']
            
            # Create unique question ID
            orig_id = row.get('original_image_id', 'unknown')
            filename = row.get('object_image_filename', f"{orig_id}.jpg")
            q_id = filename.replace('.jpg', '')
            
            questions.append({
                'question_id': q_id,
                'image_path': img_path,
                'split': split,
                'question': f"Is there a {cls.lower()} in this image?",
                'answer': 'Yes',
                'target_class': cls,
                'target_size': size,
                'question_type': 'presence_positive'
            })
        
        random.shuffle(questions)
        print(f"Generated {len(questions)} positive questions")
        return questions

    def generate_negative_questions(self):
        """Generate negative VQA questions: "Is there a <other_class> in this image?" → No"""
        questions = []
        classes = self.df['class_name'].unique().tolist()
        
        for _, row in self.df.iterrows():
            img_path = row['object_image_path']
            true_cls = row['class_name']
            size = row['size_category']
            split = row['split']
            
            # Pick a different class for negative question
            other_classes = [c for c in classes if c != true_cls]
            if not other_classes:
                continue  # Skip if only one class exists
            
            neg_cls = random.choice(other_classes)
            
            orig_id = row.get('original_image_id', 'unknown')
            filename = row.get('object_image_filename', f"{orig_id}.jpg")
            q_id = filename.replace('.jpg', '') + "_neg"
            
            questions.append({
                'question_id': q_id,
                'image_path': img_path,
                'split': split,
                'question': f"Is there a {neg_cls.lower()} in this image?",
                'answer': 'No',
                'target_class': neg_cls,
                'target_size': None,  # Not applicable for negative questions
                'question_type': 'presence_negative'
            })
        
        random.shuffle(questions)
        print(f"Generated {len(questions)} negative questions")
        return questions

    def generate_balanced_questions(self, pos_ratio=0.5):
        """Generate balanced mix of positive and negative questions."""
        pos_qs = self.generate_positive_questions()
        neg_qs = self.generate_negative_questions()
        
        # Balance the dataset
        total_desired = len(pos_qs) + len(neg_qs)
        pos_desired = int(total_desired * pos_ratio)
        neg_desired = total_desired - pos_desired
        
        if len(pos_qs) > pos_desired:
            pos_qs = random.sample(pos_qs, pos_desired)
        if len(neg_qs) > neg_desired:
            neg_qs = random.sample(neg_qs, neg_desired)
        
        all_qs = pos_qs + neg_qs
        random.shuffle(all_qs)
        
        print(f"Balanced dataset: {len(pos_qs)} positive + {len(neg_qs)} negative = {len(all_qs)} total")
        return all_qs, pos_qs, neg_qs

    def save(self, questions, output_path, fmt='json'):
        """Save questions to JSON or CSV format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if fmt == 'json':
            with open(output_path, 'w') as f:
                json.dump(questions, f, indent=2)
        elif fmt == 'csv':
            pd.DataFrame(questions).to_csv(output_path, index=False)
        
        print(f"💾 Saved {len(questions)} questions to {output_path}")

    def analyze(self, questions):
        """Print comprehensive statistics about the VQA dataset."""
        df = pd.DataFrame(questions)
        total = len(df)
        
        print(f"\n{'='*50}")
        print(f"SODA-A VQA DATASET ANALYSIS")
        print(f"{'='*50}")
        print(f"Total questions: {total:,}")
        
        # Answer distribution
        print(f"\n📊 ANSWER DISTRIBUTION:")
        ans_dist = df['answer'].value_counts()
        for answer, count in ans_dist.items():
            pct = count / total * 100
            print(f"  {answer}: {count:,} ({pct:.1f}%)")
        
        # Split distribution
        print(f"\n📂 SPLIT DISTRIBUTION:")
        split_dist = df['split'].value_counts()
        for split, count in split_dist.items():
            pct = count / total * 100
            print(f"  {split}: {count:,} ({pct:.1f}%)")
        
        # Question type distribution
        print(f"\n❓ QUESTION TYPE DISTRIBUTION:")
        type_dist = df['question_type'].value_counts()
        for qtype, count in type_dist.items():
            pct = count / total * 100
            print(f"  {qtype}: {count:,} ({pct:.1f}%)")
        
        # Class distribution
        print(f"\n🏷️ TARGET CLASS DISTRIBUTION:")
        class_dist = df['target_class'].value_counts()
        for cls, count in class_dist.items():
            pct = count / total * 100
            print(f"  {cls}: {count:,} ({pct:.1f}%)")
        
        # Size distribution (positive questions only)
        pos_df = df[df['answer'] == 'Yes']
        if len(pos_df) > 0:
            print(f"\n📏 SIZE DISTRIBUTION (positive questions):")
            size_dist = pos_df['target_size'].value_counts()
            for size, count in size_dist.items():
                pct = count / len(pos_df) * 100
                print(f"  {size}: {count:,} ({pct:.1f}%)")

def main():
    # Updated paths to match your directory structure
    CSV_PATH = "/home/wirin/Ashish/VLM/data/soda_object_crops/soda_individual_objects.csv"
    OUT_DIR = "/home/wirin/Ashish/VLM/data/vqa_soda"
    
    print("🚀 Starting SODA-A VQA dataset generation...")
    print(f"📄 Input CSV: {CSV_PATH}")
    print(f"💾 Output directory: {OUT_DIR}")
    
    # Check if input CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ Input CSV not found: {CSV_PATH}")
        print("First run the object extraction script to create individual crops.")
        return
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Initialize converter
    converter = SODAVQAConverter(CSV_PATH)
    
    # Generate balanced questions
    all_qs, pos_qs, neg_qs = converter.generate_balanced_questions(pos_ratio=0.5)
    
    # Save datasets
    converter.save(pos_qs, os.path.join(OUT_DIR, "soda_positive_questions.json"))
    converter.save(neg_qs, os.path.join(OUT_DIR, "soda_negative_questions.json"))
    converter.save(all_qs, os.path.join(OUT_DIR, "soda_all_questions.json"))
    converter.save(all_qs, os.path.join(OUT_DIR, "soda_all_questions.csv"), fmt='csv')
    
    # Analyze the dataset
    converter.analyze(all_qs)
    
    # Save a sample for inspection
    sample_size = min(20, len(all_qs))
    sample_qs = random.sample(all_qs, sample_size)
    converter.save(sample_qs, os.path.join(OUT_DIR, "soda_sample_questions.json"))
    
    print(f"\n✅ SODA-A VQA dataset generation complete!")
    print(f"📁 Files saved to: {OUT_DIR}/")
    print(f"🎯 Ready for VLM evaluation!")

if __name__ == "__main__":
    main()


