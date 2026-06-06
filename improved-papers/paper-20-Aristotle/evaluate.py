import json
import argparse

def load_json_file(file_path):
    """Loads a JSON file and returns its content."""
    with open(file_path, 'r') as file:
        return json.load(file)
    
def normalize_answer(answer):
    """Normalize the final choice: if not 'A' or 'B', treat it as 'C'."""
    if answer == 'A' or answer == 'B' or answer == 'D':
        return answer
    return 'C'

def evaluate_instance(id_, instance1, instance2, ground_truth):
    """Evaluates two instances with the same 'id' according to the rules provided."""
    try:
        answer1 = instance1.get('final_choice', 'C' if 'No final answer found in the text.' in instance1.get('final_answer', '') else None)
    except:
        answer1 = None
        
    try:
        answer2 = instance2.get('final_choice', 'C' if 'No final answer found in the text.' in instance2.get('final_answer', '') else None)
    except:
        answer2 = None
        
    answer1 = normalize_answer(answer1)
    answer2 = normalize_answer(answer2)
                
    if ground_truth == 'A':
        if {answer1, answer2} in [{'A', 'C'}, {'A'}]:
            return True
    
    elif ground_truth == 'B':
        if {answer1, answer2} in [{'B', 'C'}, {'B'}]:
            return True
    
    elif ground_truth == 'C':
        if answer1 == 'C' and (answer2 == 'C' or answer2 is None):
            return True
        
    elif ground_truth == 'D':
        if (answer1 == 'A' and answer2 == 'B') or (answer1 == 'B' and answer2 == 'A'):
            return True
    
    return False

def evaluate_files(dataset_name, model_name):
    """Evaluates all matching instances from two JSON files."""
    file1_path = f'./results/{dataset_name}/{dataset_name}_{model_name}_search_negation_True.json'
    file2_path = f'./results/{dataset_name}/{dataset_name}_{model_name}_search_negation_False.json'
    
    file1 = load_json_file(file1_path)
    file2 = load_json_file(file2_path)
    
    file1_map = {}
    file2_map = {}
    for item in file1:
        try:
            file1_map[item['id']] = item
        except TypeError:
            print(f"Error: 'id' not found or invalid in file1 item: {item}")
    for item in file2:
        try:
            file2_map[item['id']] = item
        except TypeError:
            print(f"Error: 'id' not found or invalid in file2 item: {item}")
    
    total_instances = 0
    correct_instances = 0
    
    common_ids = set(file1_map.keys()).union(set(file2_map.keys()))
    valid_ids = [id_ for id_ in common_ids if file1_map.get(id_) is not None and file2_map.get(id_) is not None]
    error_id = []
    correct_id = []

    for id_ in valid_ids:
        try:
            instance1 = file1_map.get(id_)
            instance2 = file2_map.get(id_)
                        
            ground_truth = instance1.get('ground_truth')
            
            if ground_truth:
                total_instances += 1
                if evaluate_instance(id_, instance1, instance2, ground_truth):
                    correct_instances += 1
                    correct_id.append(id_)
                else:
                    error_id.append(id_)
        except Exception as e:
            print(f"Error processing instance with ID {id_}: {str(e)}")
            error_id.append(id_)
    
    accuracy = correct_instances / total_instances if total_instances > 0 else 0
    print(f"Total instances: {total_instances}")
    print(f"Accuracy: {accuracy:.2%}")
    print("Error id: ", error_id)
    print("Correct id: ", correct_id)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    evaluate_files(args.dataset_name, args.model_name)