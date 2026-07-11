import os
import glob
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import shutil
import tempfile

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


VALID_IMAGE_EXTS = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']



class MMEMetricsCalculator:
    def divide_chunks(self, l, n=2):
        for i in range(0, len(l), n): yield l[i:i + n]

    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower().strip().replace('.', '')
        if pred_ans in ["yes", "no"]: return pred_ans
        prefix = pred_ans[:4]
        if "yes" in prefix: return "yes"
        if "no" in prefix: return "no"
        return "other"

    def compute_metric(self, gts, preds):
        label_map = {"yes": 1, "no": 0, "other": -1}
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]
        acc = accuracy_score(gts, preds) 
        clean_gts, clean_preds = [], []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        
        if len(clean_gts) > 0:
            conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
            tp, fn = conf_mat[0]
            fp, tn = conf_mat[1]
        else:
            tp, fn, fp, tn = 0, 0, 0, 0
        return {"acc": acc, "other_num": other_num}

    def process_result(self, results_dir):
        total_score = 0
        summary = {}
        print(f"\n=== Calculating MME Metrics from: {results_dir} ===")
        
        for eval_type, task_list in eval_type_dict.items():
            print(f"--- {eval_type} ---")
            cat_score = 0
            for task in task_list:
                txt_path = os.path.join(results_dir, f"{task}.txt")
                if not os.path.exists(txt_path):
                    continue
                
                lines = open(txt_path, 'r', encoding='utf-8').readlines()
                lines = [l for l in lines if l.strip()]
                chunks = list(self.divide_chunks(lines))
                
                chunks = [c for c in chunks if len(c) == 2]
                img_num = len(chunks)
                acc_plus_correct = 0
                gts, preds = [], []
                
                for items in chunks:
                    img_correct = 0
                    for item in items:
                        parts = item.strip().split("\t")
                        if len(parts) < 4: continue 
                        gt = parts[2].lower().strip()
                        pred = self.parse_pred_ans(parts[3])
                        gts.append(gt)
                        preds.append(pred)
                        if gt == pred: img_correct += 1
                    if img_correct == 2: acc_plus_correct += 1
                
                if not gts: continue
                metric = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct / img_num if img_num > 0 else 0
                score = (metric['acc'] + acc_plus) * 100
                cat_score += score
                summary[task] = round(score, 2)
                print(f"  {task}: {score:.2f}")
            
            total_score += cat_score
            print(f"  >>> {eval_type} Total: {cat_score:.2f}")
        
        print(f"\n=== Overall Score: {total_score:.2f} ===\n")
        return summary
def convert_jsonl_to_mme_format(jsonl_path, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_by_category = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
            except:
                continue
                
            category = item.get('category', 'unknown')
            pred = item.get('text', '').replace('\n', ' ').strip()
            gt = item.get('gt_answer', '').strip()
        
            img_path = item.get('image', '')
            img_name = os.path.basename(img_path)
            question = item.get('question', 'Q_Placeholder')

            if category not in data_by_category:
                data_by_category[category] = []
            
            data_by_category[category].append({
                "img_name": img_name,
                "question": question,
                "gt": gt,
                "pred": pred
            })
    
    for category, items in data_by_category.items():
        items.sort(key=lambda x: x['img_name'])
        
        txt_path = os.path.join(output_dir, f"{category}.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            for item in items:
                line = f"{item['img_name']}\t{item['question']}\t{item['gt']}\t{item['pred']}\n"
                f_out.write(line)
def eval_jsonl_file(jsonl_path):

    print(f"\n{'='*20} Evaluating: {os.path.basename(jsonl_path)} {'='*20}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        convert_jsonl_to_mme_format(jsonl_path, temp_dir)
        cal = MMEMetricsCalculator()

        summary = cal.process_result(temp_dir)

        total_score = sum(summary.values())
        return total_score, summary

def run_mme_inference(inference_fn, mme_data_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    print(f"Starting MME Inference.\nData Path: {mme_data_dir}")

    if not os.path.exists(mme_data_dir):
        print(f"Error: MME Data Directory not found: {mme_data_dir}")
        return
    
    all_subdirs = {d.lower(): d for d in os.listdir(mme_data_dir) if os.path.isdir(os.path.join(mme_data_dir, d))}

    for eval_type, task_list in eval_type_dict.items():
        for task_name in task_list:
            real_folder_name = all_subdirs.get(task_name.lower())
            if not real_folder_name:
                continue
            
            task_folder = os.path.join(mme_data_dir, real_folder_name)
            output_file = os.path.join(results_dir, f"{task_name}.txt")
            
            if os.path.exists(output_file):
                os.remove(output_file)

            all_txts = glob.glob(os.path.join(task_folder, "*.txt"))
            gt_files_to_process = [f for f in all_txts if "readme" not in f.lower() and "license" not in f.lower()]

            if not gt_files_to_process:
                continue

            print(f"Processing {task_name} (Found {len(gt_files_to_process)} txt files)...")
            
            results_lines = []
            
            for gt_file_path in tqdm(gt_files_to_process, desc=f"  Inferencing {task_name}", leave=False):
                txt_basename = os.path.basename(gt_file_path)

                img_id_from_txt = os.path.splitext(txt_basename)[0]

                with open(gt_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split("\t")
                    

                    if len(parts) == 2:
                        img_basename = img_id_from_txt
                        question = parts[0]
                        gt_ans = parts[1]
                    elif len(parts) >= 3:
                        img_basename = os.path.splitext(parts[0])[0]
                        question = parts[1]
                        gt_ans = parts[2]
                    else:
                        continue
                    
                    img_path = None
                    found_img_file = None 
                    
                    search_dirs = [task_folder, os.path.join(task_folder, "images")]
                    
                    for search_dir in search_dirs:
                        if not os.path.exists(search_dir): continue
                        
                        for ext in VALID_IMAGE_EXTS:
                            test_path = os.path.join(search_dir, img_basename + ext)
                            if os.path.exists(test_path):
                                img_path = test_path
                                found_img_file = img_basename + ext 
                                break
                        if img_path: break

                    # inference
                    if not img_path:
                        pred_ans = "image error"
                        found_img_file = img_basename + ".jpg" 
                    else:
                        try:
                            prompt = question
                            pred_ans = inference_fn(img_path, prompt)
                            pred_ans = pred_ans.replace("\n", " ").strip()
                        except Exception as e:
                            print(f"    Error on {img_basename}: {e}")
                            pred_ans = "error"
                    
                    out_line = f"{found_img_file}\t{question}\t{gt_ans}\t{pred_ans}\n"
                    results_lines.append(out_line)
            
            if results_lines:
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(results_lines)

def calculate_mme_metric(inference_fn, mme_data_dir, results_dir):
    run_mme_inference(inference_fn, mme_data_dir, results_dir)
    
    cal = MMEMetricsCalculator()
    metrics = cal.process_result(results_dir)
    
    with open(os.path.join(results_dir, "mme_scores.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics