import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import nltk

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in nltk.data.path:
    nltk.data.path.append(current_dir)

# from pattern.text.en.inflect import singularize
import inflect
_chair_inflect_engine = inflect.engine()

def singularize(word):
    res = _chair_inflect_engine.singular_noun(word)
    return res if res else word


DEFAULT_SYNONYMS_FILE = "./synonyms.txt"

class CHAIR:
    def __init__(self, imids, coco_annotation_path, synonyms_file=DEFAULT_SYNONYMS_FILE):
        self.imid_to_objects = {imid: set() for imid in imids}
        self.coco_annotation_path = coco_annotation_path
        
        synonyms = [s.strip().split(", ") for s in open(synonyms_file).readlines()]
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for synonym_group in synonyms:
            self.mscoco_objects.extend(synonym_group)
            for s in synonym_group:
                self.inverse_synonym_dict[s] = synonym_group[0]
        
        coco_double_words = [
            "motor bike", "motor cycle", "air plane", "traffic light", "stop sign", 
            "parking meter", "sports ball", "baseball bat", "baseball glove", 
            "tennis racket", "wine glass", "hot dog", "cell phone", "teddy bear", 
            "hair drier", "potted plant"
        ]
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']

        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
            
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word

        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict['wine glas'] = 'wine glass' 
        
    def caption_to_words(self, caption):
        words = [singularize(w) for w in nltk.word_tokenize(caption.lower())]

        i, double_words = 0, []
        while i < len(words):
            if " ".join(words[i:i+2]) in self.double_word_dict:
                double_words.append(self.double_word_dict[" ".join(words[i:i+2])])
                i += 2
            else:
                double_words.append(words[i])
                i += 1

        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]
            
        words = [w for w in double_words if w in self.mscoco_objects]
        node_words = [self.inverse_synonym_dict[w] for w in words]
        return words, node_words

    def get_annotations(self):
        print("Loading ground truth annotations...")

        instances_train = json.load(open(os.path.join(self.coco_annotation_path, "instances_train2014.json")))
        instances_val = json.load(open(os.path.join(self.coco_annotation_path, "instances_val2014.json")))
        id_to_name = {cat["id"]: cat["name"] for cat in instances_train["categories"]}
        for ann in tqdm(instances_train["annotations"] + instances_val["annotations"], desc="Processing segmentations"):
            if ann["image_id"] in self.imid_to_objects:
                self.imid_to_objects[ann["image_id"]].add(self.inverse_synonym_dict[id_to_name[ann["category_id"]]])
        
        caps_train = json.load(open(os.path.join(self.coco_annotation_path, "captions_train2014.json")))
        caps_val = json.load(open(os.path.join(self.coco_annotation_path, "captions_val2014.json")))
        for ann in tqdm(caps_train["annotations"] + caps_val["annotations"], desc="Processing GT captions"):
            if ann["image_id"] in self.imid_to_objects:
                _, node_words = self.caption_to_words(ann["caption"])
                self.imid_to_objects[ann["image_id"]].update(node_words)

    def compute_chair(self, generated_captions):
        num_caps, num_hallucinated_caps = 0.0, 0.0
        hallucinated_word_count, coco_word_count = 0.0, 0.0

        for cap_eval in tqdm(generated_captions, desc="Computing CHAIR"):
            imid = cap_eval["image_id"]
            if imid not in self.imid_to_objects: continue
            
            words, node_words = self.caption_to_words(cap_eval["caption"])
            gt_objects = self.imid_to_objects[imid]
            
            coco_word_count += len(node_words)
            is_hallucinated_cap = False
            for node_word in node_words:
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    is_hallucinated_cap = True
            
            num_caps += 1
            if is_hallucinated_cap:
                num_hallucinated_caps += 1

        chair_s = num_hallucinated_caps / num_caps if num_caps > 0 else 0
        chair_i = hallucinated_word_count / coco_word_count if coco_word_count > 0 else 0
        
        return {"CHAIRs": chair_s, "CHAIRi": chair_i}

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def calculate_chair_metric(response_file: str, coco_path: str):
    """
    Pipeline of calculate CHAIR
    """
    print(f"--- Starting CHAIR calculation for: {os.path.basename(response_file)} ---")
    
    generated_data_raw = read_jsonl(response_file)
    generated_data = []

    for item in generated_data_raw:
        img_id = int(item.get('image_id') or item.get('question_id'))
        caption = item.get('text') or item.get('caption', "")
        
        generated_data.append({
            'image_id': img_id,
            'caption': caption
        })

    unique_generated_data = list({item['image_id']: item for item in generated_data}.values())
    generated_img_ids = sorted([item['image_id'] for item in unique_generated_data])
    
    print(f"Found {len(generated_img_ids)} unique generated captions.")
    annotation_dir = os.path.join(coco_path, "annotations")
    evaluator = CHAIR(imids=generated_img_ids, coco_annotation_path=annotation_dir)
    evaluator.get_annotations()
    print("Converting results to COCO format for standard metrics (e.g., BLEU)...")
    coco_gt = COCO(os.path.join(annotation_dir, "captions_val2014.json")) 
    coco_res = coco_gt.loadRes([{'image_id': item['image_id'], 'caption': item['caption']} for item in unique_generated_data])
    
    coco_eval = COCOEvalCap(coco_gt, coco_res)

    imgIds = generated_img_ids
    gts = {}
    res = {}
    
    print(f"Filtering annotations for {len(imgIds)} images...")
    for imgId in imgIds:
        if imgId in coco_eval.coco.imgToAnns:
            gts[imgId] = coco_eval.coco.imgToAnns[imgId]
            res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]
        else:
            print(f"Warning: Image ID {imgId} not found in COCO Ground Truth.")

    print('tokenization...')
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # if network is not used for download nltk.
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    coco_eval.eval = {}
    for scorer, method in scorers:
        print('computing %s score...' % (scorer.method()))
        try:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    coco_eval.eval[m] = sc
                    print("%s: %0.3f" % (m, sc))
            else:
                coco_eval.eval[method] = score
                print("%s: %0.3f" % (method, score))
        except Exception as e:
            print(f"Error computing {method}: {e}")

    chair_scores = evaluator.compute_chair(unique_generated_data)
    final_results = {
        "source_file": os.path.basename(response_file),
        "CHAIRs": round(chair_scores["CHAIRs"], 4),
        "CHAIRi": round(chair_scores["CHAIRi"], 4),
    }

    for metric, score in coco_eval.eval.items():
        final_results[metric] = round(score, 4)
        
    print("\n--- Evaluation Results ---")
    print(json.dumps(final_results, indent=4))
    

    output_path = response_file.replace('.jsonl', '_chair_results.json')
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CHAIR and other captioning metrics.")
    parser.add_argument(
        "--response_file", 
        type=str, 
        required=True,
        help="Path to the .jsonl file with generated captions."
    )
    parser.add_argument(
        "--coco_path", 
        type=str, 
        default='/data/coco2014/',
        help="Root directory of the COCO dataset, containing the 'annotations' folder."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"Response file not found: {args.response_file}")
    if not os.path.exists(os.path.join(args.coco_path, "annotations")):
        raise FileNotFoundError(f"COCO annotations folder not found in: {args.coco_path}")
        
    calculate_chair_metric(args.response_file, args.coco_path)