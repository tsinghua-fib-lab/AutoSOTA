import json
import os
from tqdm import tqdm
from utils import OpenAIModel
import argparse
import re
import sys
import concurrent.futures
import threading
import traceback

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.negation = args.negation
        self.mode = args.mode
        self.search_round = args.search_round
        self.file_lock = threading.Lock()
        self.batch_num = args.batch_num
        if args.base_url:
            self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, base_url=args.base_url)
        else:
            self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
    
    def load_in_context_examples_complement(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'complement_search.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_logic_resolver(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'logic_resolver.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    
    def load_raw_dataset(self, split):
        if self.negation == 'True':
            file_path = f"./results/{self.dataset_name}/{self.dataset_name}_{self.model_name}_trans_decompose_negated_data.json"
        else:
            file_path = f"./results/{self.dataset_name}/{self.dataset_name}_{self.model_name}_trans_decompose_no_negation.json"
        print(f"Loading raw dataset from {file_path}")
        with open(file_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_a(self, record, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        context = record['context']
        question = record['question'].strip()
        full_prompt = full_prompt.replace('[[PREMISES]]', context)
        full_prompt = full_prompt.replace('[[CONJECTURE]]', question)
        return full_prompt

    def construct_prompt_b(self, responses_a, in_context_examples_decomposer):
        full_prompt = in_context_examples_decomposer
        full_prompt = full_prompt.replace('[[PREMISES]]', responses_a)
        return full_prompt

    def construct_prompt_c(self, responses_b, in_context_examples_search_init):
        full_prompt = in_context_examples_search_init
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_b)
        return full_prompt
    
    def construct_prompt_d(self, normalized_context, normalized_conjecture, negated_label, reasoning_step, sos_list, in_context_examples_search_router):
        full_prompt = in_context_examples_search_router
        full_prompt = full_prompt.replace('[[CONTEXT]]', normalized_context)
        full_prompt = full_prompt.replace('[[CONJECTURE]]', normalized_conjecture)
        full_prompt = full_prompt.replace('[[REASONING]]', reasoning_step)
        full_prompt = full_prompt.replace('[[SOS]]', sos_list)
        return full_prompt
    
    def construct_prompt_e(self, negated_label, conjecture, sos, selected_clause, in_context_examples_logic_resolver):
        full_prompt = in_context_examples_logic_resolver
        full_prompt = full_prompt.replace('[[SOS]]', sos)
        full_prompt = full_prompt.replace('[[SELECTED-CLAUSE]]', selected_clause)
        return full_prompt
    
    def construct_prompt_complement(self, sos, clause, in_context_examples_complement):
        full_prompt = in_context_examples_complement
        full_prompt = full_prompt.replace('[[SOS]]', sos)
        full_prompt = full_prompt.replace('[[CLAUSE]]', clause)
        return full_prompt
    
    def post_process_b(self, response_b):
        match = re.search(r"### Final Form:([\s\S]*)", response_b)
        if match:
            final_form_text = match.group(1)
            return final_form_text
        else:
            print("Text not found.")
            
    def extract_selected_clause(self, text):
        match = re.search(r'\[([^\[\]]+)\]', text)
        if match:
            selected_clause = match.group(1).strip()
            return selected_clause
        else:
            return "Selected clause not found."
        

    def post_process_c(self, response_c):
        sos_list = re.findall(r'\[(.*?)\]', response_c)
        negated_label = re.findall(r'\{(.*?)\}', response_c)
        str_sos_list = "".join(sos_list)
        str_negated_label = "".join(negated_label)
        return str_negated_label, str_sos_list
    
    def post_process_logic_solver(self, response_d):
        sufficiency_label_match = re.search(r'\[(.*?)\]', response_d)
        new_clause_match = re.search(r'\{(.*?)\}', response_d)
        
        new_clause = new_clause_match.group(1).strip() if new_clause_match else "New Clause not found."
        sufficiency_label = sufficiency_label_match.group(1).strip() if sufficiency_label_match else "Sufficiency Label not found."
        
        return {
            "new_clause": new_clause,
            "sufficiency_label": sufficiency_label,
        }
    
    def post_process_final_answer(self, response_c):
        pattern_bracket = r"Final answer: \{([A-E])\}"
        match = re.search(pattern_bracket, response_c)
        if match:
            answers =  match.group(1)
            return answers
        pattern_direct = r'\{(\w+)\}'
        match = re.search(pattern_direct, response_c, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "No final answer found in the text."

    
    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        elif final_answer == "unknown":
            final_answer = 'C'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer
    
    
    def clean_conjecture(self, conjecture):
        if isinstance(conjecture, dict):
            lines = []
            for key, value in conjecture.items():
                if not value:
                    continue

                if key == "Fact":
                    for line in value.splitlines():
                        if not line.strip():
                            continue
                        if ":::" in line:
                            left, right = line.split(":::", 1)
                            candidate = left if "(" in left else right
                            lines.append(candidate.strip())
                        elif "(" in line:
                            lines.append(line.strip())
                else:
                    lines.extend([l for l in value.splitlines() if l.strip()])

        else:
            lines = [l for l in conjecture.splitlines() if l.strip()]

        cleaned = []
        remove_list = ['Rules (others):', 'Rules (biconditional):', 'Rules (either_or):']
        for item in lines:
            if "(" in item and not any(rem in item for rem in remove_list):
                cleaned.append(item)

        return "\n".join(cleaned)
    
    def list_to_indexed_string(self, item_list):
        indexed_list = [f"{i + 1}. {item}" for i, item in enumerate(item_list)]
        return "\n".join(indexed_list)
    
    def extract_text_in_brackets(self, text):
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, text)
        
        if matches:
            last_match = matches[-1]
            
            cleaned_match = last_match.lower().strip()
            
            if 'true' in cleaned_match:
                return 'true'
            elif 'false' in cleaned_match:
                return 'false'
        
        return None
    
    def is_complementary(self, term1, term2):
        pattern = r'(\w+)\((.*?)\)'
        
        match1 = re.search(pattern, term1)
        match2 = re.search(pattern, term2)
        
        if not match1 or not match2:
            return False
        
        predicate1, args1 = match1.group(1), match1.group(2).split(',')
        predicate2, args2 = match2.group(1), match2.group(2).split(',')
        
        if predicate1 != predicate2:
            return False
        
        if len(args1) != len(args2):
            return False
        
        for arg1, arg2 in zip(args1[:-1], args2[:-1]): 
            if arg1.strip() != arg2.strip() and arg1.strip() != 'x' and arg2.strip() != 'x':
                return False
        
        bool1 = args1[-1].strip()
        bool2 = args2[-1].strip()
        
        return (bool1 == 'True' and bool2 == 'False') or (bool1 == 'False' and bool2 == 'True')


    def filter_complementary_context(self, normalized_context, conjecture):

        def clean_and_extract_predicates(logical_statement):
            cleaned_statement = re.sub(r'[0-9.]', '', logical_statement)

            operator_pattern = r'(\\lor|\\land|\u2228|\u2227|\\wedge|\\vee)'
            terms = re.split(operator_pattern, cleaned_statement)
            return [term.strip() for term in terms]

        complementary_context_indices = []

        for idx, context in enumerate(normalized_context):
            terms = clean_and_extract_predicates(context)

            for term in terms:
                if self.is_complementary(term, conjecture):
                    complementary_context_indices.append(idx)
                    break 

        return complementary_context_indices

    def negate_boolean(self, expression):
        return re.sub(r"(False|True)", lambda x: "True" if x.group() == "False" else "False", expression, count=1)
    
    def ensure_bool_inside_parentheses(self, match):
        inner_content = match.group(1)
        terms = [term.strip() for term in inner_content.split(',')]
        if terms[-1] in ["True", "False"]:
            return f"({inner_content})"
        else:
            return f"({inner_content}, True)"
        
    def remove_negations(self, context):
        print('Negation Removal...', context)
        neg_list = ['\u00ac', '¬', '\\neg', '\\lnot', 'eg']
        
        context = re.sub(r"\(([^)]+)\)", self.ensure_bool_inside_parentheses, context)
        
        print('Negation Removal...', context)
        for neg in neg_list:
            while neg in context:
                neg_pos = context.find(neg)
                rest = context[neg_pos + len(neg):]

                bool_match = re.search(r"(False|True)", rest)
                if bool_match:
                    bool_pos = bool_match.start()
                    bool_val = bool_match.group()
                    
                    new_bool = "True" if bool_val == "False" else "False"
                    
                    context = context[:neg_pos] + rest[:bool_pos] + new_bool + rest[bool_pos + len(bool_val):]
                else:
                    context = context[:neg_pos] + rest
        print('finish removal')
        return context
    
    def reasoning_graph_generation(self):
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        
        in_context_examples_logic_resolver = self.load_in_context_examples_logic_resolver()
        
        counter = 0
    
        def process_example(example, counter):
            try:
                print(f"Running example: {example['id']}")

                flag = 'true'
                reasoning_step = []
                search_round = 0
                
                normalized_context = example['normalized_context']
                cleaned_normalized_context = self.clean_conjecture(normalized_context)
                normalized_context_list = cleaned_normalized_context.splitlines()
                normalized_context_list = [
                    '\n'.join(self.remove_negations(line) for line in item.replace('\\textnormal', '').replace('\\textrm', '').replace('\\\\text', '\\text')
                            .replace('\\text', '').replace('-', '') 
                            .replace('{', '').replace('}', '').replace('\\right', '')
                            .replace('\\left', '').replace('\\newline', '\n').replace('$', '').split('\n'))
                    for item in normalized_context_list if "(" in item and ")" in item
                ]

                
                normalized_conjecture = self.clean_conjecture(example['normalized_conjecture'])
                negated_label = example['negated_label']
                sos_list = self.clean_conjecture(example['sos_list'])
                if '(' and ')' not in sos_list:
                    sos_list = normalized_conjecture
                    sos_list = self.remove_negations(sos_list)
                    if example['negated_label'] == 'True':
                        sos_list = self.negate_boolean(sos_list)
                        
                modified_context_list = []
                for item in normalized_context_list:
                    item = next((p.strip() for p in item.split(":::", 1) if "(" in p), "")
                    if '\\wedge' in item:
                        split_items = item.split('\\wedge')
                        modified_context_list.extend([sub_item.strip() for sub_item in split_items])
                    elif '\\land' in item:
                        split_items = item.split('\\land')
                        modified_context_list.extend([sub_item.strip() for sub_item in split_items])
                    elif '\u2227' in item:
                        split_items = item.split('\u2227')
                        modified_context_list.extend([sub_item.strip() for sub_item in split_items]) 
                    else:
                        modified_context_list.append(item)
                normalized_context_list = [item for item in modified_context_list if "(" in item and ")" in item]
                print("Normalized Context List: ", normalized_context_list)
                print("Normalized Conjecture: ", normalized_conjecture)
                print("sos_list: ", sos_list)
                
                normalized_context_list.append(sos_list)
                
                list_of_sos = []
                list_of_compelment = []
                
                selected_clause = None
                    
                while flag == 'true':
                    if search_round >= self.search_round:
                        final_answer = "No final answer found in the text."
                        break
                    
                    if selected_clause == None:
                        print("Search Router Operating...")
                        print("Running: ", example['id'])
                        print("Ground truth: ", example['ground_truth'])
                        
                        complement_indices = self.filter_complementary_context(normalized_context_list, sos_list)
                        
                        if complement_indices:
                            potential_clauses = [normalized_context_list[index] for index in complement_indices]
                            valid_clauses = sorted(
                                [clause for clause in potential_clauses if not any(step[0] == sos_list and step[1] == clause for step in reasoning_step)],
                                key=len
                            )
                            print("Potential Clauses: ", potential_clauses)
                            print("Valid Clauses: ", valid_clauses)
                            
                            if valid_clauses:
                                if len(valid_clauses) > 1:
                                    list_of_sos.append(sos_list)
                                    list_of_compelment.append(valid_clauses)
                                    print("Current SOS: ", sos_list, "Current Complement: ", valid_clauses)
                                    selected_clause = list_of_compelment[-1].pop(0)
                                else:
                                    selected_clause = valid_clauses[0]
                            else:
                                print("All potential clauses have been used before with this SOS list.")
                                print("Checking cached SOS and complement pairs.")
                                found_new_pair = False
                                if len(list_of_compelment) > 0:
                                    for i, complement_clauses in enumerate(list_of_compelment):
                                        if complement_clauses:
                                            sos_list = list_of_sos[i]
                                            selected_clause = complement_clauses.pop(0)
                                            found_new_pair = True
                                            break
                                    if not found_new_pair:
                                        print("No more sos and complement pairs found in cache.")
                                        final_answer = "cannot find sos with complement"
                                        break

                        
                        if not complement_indices:
                            if len(list_of_compelment) > 0:
                                all_empty = True
                                original_sos = sos_list
                                original_selected_clause = selected_clause
                                for i, clauses in enumerate(list_of_compelment):
                                    if len(clauses) > 0:
                                        new_selected_clause = list_of_compelment[i][0]
                                        new_sos_list = list_of_sos[i]
                                        if new_sos_list != original_sos or new_selected_clause != original_selected_clause:
                                            selected_clause = list_of_compelment[i].pop(0)
                                            sos_list = new_sos_list
                                            all_empty = False 
                                            print("Check cache: ", sos_list, "Current Complement: ", selected_clause)
                                            break
                                if all_empty:
                                    final_answer = "No complement found in both context and cache."
                                    break
                            else: 
                                final_answer = "No complement found in the context."
                                break

                    if any(step[0].strip() == sos_list.strip() and step[1].strip() == selected_clause.strip() for step in reasoning_step):
                        print("Skipping this search round as it has appeared before.")
                        found_new_pair = False
                        for i, complement_clauses in enumerate(list_of_compelment):
                            if complement_clauses:
                                new_sos_list = list_of_sos[i]
                                new_selected_clause = complement_clauses[0]
                                if new_sos_list != sos_list or new_selected_clause != selected_clause:
                                    sos_list = new_sos_list
                                    selected_clause = complement_clauses.pop(0)
                                    found_new_pair = True
                                    break

                        if not found_new_pair:
                            print("No more sos and complement pairs found in cache.")
                            final_answer = "No sos and complement found"
                            break 
                    else:
                        print("SOS: ", sos_list)
                        print("Selected Clause: ", selected_clause)
                        print("Search round: ", search_round)
                        
                    prompts_e = self.construct_prompt_e(negated_label, normalized_conjecture, sos_list, selected_clause, in_context_examples_logic_resolver)
                    responses_e, _ = self.openai_api.generate(prompts_e)
                    
                    logic_solver_result = self.post_process_logic_solver(responses_e)
                    new_clause = logic_solver_result['new_clause']
                    sufficiency_label = logic_solver_result['sufficiency_label']
                    
                    solve_step = [sos_list, selected_clause, new_clause]
                    
                    reasoning_step.append(solve_step)
                    print('Reasoning Steps:')
                    for step in reasoning_step:
                        sos_list, selected_clause, new_clause = step
                        solving_step = f"SOS clause: {sos_list}. Selected Clause: {selected_clause}. New Clause: {new_clause}"
                        print(solving_step)
                    
                    if not new_clause.strip() or new_clause == "New Clause not found.":
                        print("No new clause found. Searhing from the cache.")
                        all_empty = "True"
                        for i, clause in enumerate(list_of_compelment):
                            if len(clause) > 0:
                                selected_clause = list_of_compelment[i].pop(0)
                                sos_list = list_of_sos[i]
                                all_empty = "False"
                                print("Searching from cache：Current SOS: ", sos_list, "Current Complement: ", selected_clause)
                                break
                                
                        if len(list_of_compelment) > 0 and all_empty == "True": 
                            final_answer = "Unknown"
                            flag = 'false'
                        elif len(list_of_compelment) == 0:
                            final_answer = "Unknown"
                            flag = 'false'
                    else:
                        sos_list = new_clause
                        normalized_context_list.append(new_clause)
                        selected_clause = None
                    
                    if sufficiency_label == "True":
                        if new_clause.lower() == "contradiction" or "false":
                            if negated_label.lower() == "true":
                                final_answer = "True"
                            elif negated_label.lower() == "false":
                                final_answer = "False"
                        
                            flag = 'false'
                        
                        else: 
                            all_empty = "True"
                            for i, clause in enumerate(list_of_compelment):
                                if len(clause) > 0:
                                    selected_clause = list_of_compelment[i].pop(0)
                                    sos_list = list_of_sos[i]
                                    all_empty = "False"
                                    print("Check Cache: ", sos_list, "Current Complement: ", selected_clause)
                                    break
                                
                            if len(list_of_compelment) > 0 and all_empty == "True":
                                final_answer = "Unknown"
                                flag = 'false'
                            elif len(list_of_compelment) == 0:
                                final_answer = "Unknown"
                                flag = 'false'

                    search_round += 1
                    
                final_choice = self.final_process(final_answer)
                
                output = {'id': example['id'], 
                        'original_context': example['original_context'],
                        'question': example['question'], 
                        'translated_context': example['translated_context'],
                        'normalized_context': example['normalized_context'],
                        'normalized_conjecture': example['normalized_conjecture'],
                        'negated_label': negated_label,
                        'reasoning_step': self.list_to_indexed_string(reasoning_step),
                        'ground_truth': example['ground_truth'], 
                        'final_answer': final_answer,
                        'final_choice': final_choice,
                        'search_round': search_round}
                
                print(output)
                return output
        
            except Exception as e:
                print('Error in generating example: ', example['id'])
                print(e)
                error = {'id': example['id']}
                return error

        def save_output(output, is_error=False):
            if "llama" in self.model_name:
                model_name = 'llama'
            else:
                model_name = self.model_name
            file_name = f'{self.dataset_name}_{model_name}_search_negation_{self.negation}.json'
            file_path = os.path.join(self.save_path, self.dataset_name, file_name)
            print("Saving result with thread lock in path: ", file_path)
            
            with self.file_lock:
                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            file_content = f.read()
                            if file_content.strip():
                                existing_data = json.loads(file_content)
                            else:
                                print(f"File {file_path} is empty. Initializing with an empty list.")
                                existing_data = []
                    else:
                        existing_data = []
                    
                    existing_data.append(output)
                    
                    with open(file_path, 'w') as f:
                        json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error in saving {'error' if is_error else ''} output: {e}")

        counter = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            future_to_example = {executor.submit(process_example, example, counter): example for example in raw_dataset}
            for future in concurrent.futures.as_completed(future_to_example):
                example = future_to_example[future]
                try:
                    output = future.result()
                    if 'error' in output:
                        save_output(output, is_error=True)
                    else:
                        print(f"Saving output for example: {output}")
                        save_output(output)
                except Exception as exc:
                    print(f'{example["id"]} generated an exception: {exc}')
                    traceback.print_exc()
                counter += 1
                            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--negation', type=str, default='False')
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--search_round', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.reasoning_graph_generation()