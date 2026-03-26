import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import f1_score
from transformers import PreTrainedTokenizer, PreTrainedModel
# Download the necessary NLTK models (if not already downloaded)
nltk.download('punkt')


def set_pad_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, pad_token='<|pad|>'):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=pad_token))

    current_size = model.get_input_embeddings().num_embeddings

    if len(tokenizer) > current_size:
        model.resize_token_embeddings(len(tokenizer))
        num_new_tokens = len(tokenizer) - current_size
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    tokenizer.add_eos_token = True


def construct_inference_prompt(query, template=None):
    if template is None:
        template = """
            {{query}} Can you provide the explanation and output the final results? Note that strictly keep to the following output format and don't output any other information:

            Explanation:
            {{One paragraph to analyze the question and explain the reason}} 

            Answer:
            {{A few words. As brief as possible}}
        """
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message


def construct_weak_model_prompt(query, template=None):
    if template is None:
        template = """
            {{query}} Can you provide the explanation and output the final results? 
        """
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message


def construct_strong_model_prompt(query, weak_model_output, template=None):
    if template is None:
        template = """
            Given a question and the output of an expert model, please refer to this output if you think it's helpful and correct, otherwise generate the final result based on your own knowledge.
            
            Query:
            {{query}}

            
            Output of expert model:
            {{weak_model_output}}

            Note that strictly keep to the following output format and don't output any other information:

            Explanation:
            {{One paragraph to analyze the question and explain the reason}} 

            Answer:
            {{A few words. As brief as possible}}
            
        """
    text = template.replace("{{query}}", query)
    text = text.replace("{{weak_model_output}}", weak_model_output)
    text = text.strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    return message


def construct_evalution_prompt(model_output, groundtruth, template=None):
    if template is None:
        template = """
            Given the groundtruth and the model output (including the explanation and the answer), score the model output with respect to correctness on a scale from 1 to 10. Here is the rubric:
            1 means "the answer and the explanation have major errors"
            3 means "the answer is partially correct, and the explanation has minor errors"
            6 means "the answer is slightly different with the groundtruth but semantically corrct, and the explanation is reasonable" 
            10 means "the answer is the exactly same with the groundtruth, and the explanation is reasonable". 

            Please just output a number as the score. Don't output any other information. 
            
            Groundtruth:
            {{groundtruth}}

            
            Model output:
            {{model_output}}

            
            Please only output a integer from 1 to 10 as the score. Don't output any other information. 
        """

    text = template.replace("{{model_output}}", model_output)
    text = text.replace("{{groundtruth}}", groundtruth)
    text = text.strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    return message



'''
def construct_weak_model_prompt(query, template=None):
    if template is None:
        template = """
            {{query}} Can you provide the related background knowledge and output the final results? 
        """
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message




def construct_qa_prompt(query, template=None):
    if template is None:
        template = """
            Generate one paragraph to answer the following question.  
            {{query}} 
        """
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message


def construct_qa_prompt_with_knowledge(query, knowledge, template=None):
    if template is None:
        template = """
            Given a multi-choice question, four options, and some background knowledge, can you provide the explanation and output the final results?? 
            
            Background Knowledge:
            {{background knowledge}} 
            
            Question and Options:
            {{query}}
            
            Note that strictly keep to the following output format and don't output any other information:

            Explanation:
            {{One paragraph to explain why you choose this option}} 

            Answer:
            {{One letter (like A, B, C, D) as the right option}}

        """
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message

'''


def construct_weak_model_prompt_medmcqa(query, template=None):
    if template is None:
        template = """
            {{query}} Can you provide the explanation and output the final results (one letter like A, B, C, or D)? 
        """
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message



def construct_inference_prompt_medmcqa(query, template=None):
    if template is None:
        # template = """{{query}} Can you output the final results, like A, B, C, D?}}\n"""
        template = """{{query}} Can you provide the explanation and output the final results? Note that strictly keep to the following output format and don't output any other information:\n\nExplanation:\n{{One paragraph to explain why you choose this option}}\n\nAnswer:\n{{One letter (like A, B, C, D) as the right option}}\n"""
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message
'''

def construct_inference_prompt_medmcqa(query, template=None):
    if template is None:
        template = """{{query}}"""
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message
'''

def construct_strong_model_prompt_medmcqa(query, weak_model_output, template=None):
    if template is None:
        template = """
            Given a question and the output of an expert model, please refer to this output if you think it's helpful and correct, otherwise generate the final result based on your own knowledge.
            
            Query:
            {{query}}

            
            Output of expert model:
            {{weak_model_output}}

            Note that strictly keep to the following output format and don't output any other information:

            Explanation:
            {{One paragraph to explain why you choose this option}} 

            Answer:
            {{One letter (like A, B, C, D) as the right option}}
            
        """
    text = template.replace("{{query}}", query)
    text = text.replace("{{weak_model_output}}", weak_model_output)
    text = text.strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    return message







def construct_weak_model_prompt_prosocial(query, template=None):
    if template is None:
        template = """
            {{query}} Can you provide the explanation and output the final results (a label like casual, needs caution, or needs intervention)? 
        """
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message



def construct_inference_prompt_prosocial(query, template=None):
    if template is None:
        # template = """{{query}} Can you output the final results, like A, B, C, D?}}\n"""
        template = """{{query}} Can you provide the explanation and output the classification label? Note that strictly keep to the following output format and don't output any other information:\n\nExplanation:\n{{One paragraph to explain why you choose this label}}\n\nAnswer:\n{{a label (casual, needs caution, or needs intervention)}}\n"""
    template = template.replace("{{query}}", query).strip()
    
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    '''
    message = [
        {"role": "system", "content": "You are an assistant on dialogue safety. For a dialogue, it needs caution if situations that may be problematic, unethical, rude, toxic, or biased, requiring careful responses to maintain prosocial interactions. It needs intervention if serious contexts needing human intervention, such as medical emergencies or imminent danger, where professional help (e.g., calling 911) is necessary. It is casual if it involves non-problematic situations, including everyday interactions, casual chat, and positive or empathetic exchanges."},
        {"role": "user", "content": template},
    ]
    '''
    return message
'''

def construct_inference_prompt_medmcqa(query, template=None):
    if template is None:
        template = """{{query}}"""
    template = template.replace("{{query}}", query).strip()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": template},
    ]
    return message
'''



def construct_strong_model_prompt_prosocial(query, weak_model_output, template=None):
    if template is None:
        template = """
            For a dialogue safety classification task and the output of an expert model, please refer to this output if you think it's helpful and correct, otherwise classify the dialogue by yourself. 
            
            Task:
            {{query}}

            
            Output of expert model:
            {{weak_model_output}}


            Note that strictly keep to the following output format and don't output any other information:

            Explanation:
            {{A sentence to explain why you choose this label}} 

            Answer: 
            {{A label (casual, needs caution, or needs intervention)}}
            
        """
    # dialogue safety usually should be handled with caution. S
    text = template.replace("{{query}}", query)
    text = text.replace("{{weak_model_output}}", weak_model_output)
    text = text.strip()
    message = [
        {"role": "system", "content": "You are an expert on dialogue safety. For a dialogue, it needs caution if situations that may be problematic, unethical, rude, toxic, or biased, requiring careful responses to maintain prosocial interactions. It needs intervention if serious contexts needing human intervention, such as medical emergencies or imminent danger, where professional help (e.g., calling 911) is necessary. It is casual if it involves non-problematic situations, including everyday interactions, casual chat, and positive or empathetic exchanges."},
        {"role": "user", "content": text},
    ]
    return message











def calculate_accuracy(pred_answers, true_answers):
    if len(pred_answers) != len(true_answers):
        raise ValueError("Length of pred_answers and true_answers must be the same.")

    num_correct = 0
    total = len(pred_answers)

    for pred, true in zip(pred_answers, true_answers):
        if pred == true:
            num_correct += 1

    accuracy = num_correct / total
    return accuracy





def calculate_f1(pred_answers, true_answers):
    if len(pred_answers) != len(true_answers):
        raise ValueError("Length of pred_answers and true_answers must be the same.")
    # Get the set of unique classes (possible answers)
    unique_classes = set(true_answers)

    # Initialize variables to store F1 scores for each class
    f1_scores = []

    for cls in unique_classes:
        # Calculate true positives, false positives, and false negatives for this class
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for pred, true in zip(pred_answers, true_answers):
            if pred == cls and true == cls:
                true_positive += 1
            elif pred == cls and true != cls:
                false_positive += 1
            elif pred != cls and true == cls:
                false_negative += 1

        # Calculate precision and recall for this class
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        # Calculate F1 score for this class
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # Store the F1 score for this class
        f1_scores.append(f1)

    # Calculate the macro F1 score (average of F1 scores for all classes)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return macro_f1




def calculate_f1_score(pred_tokens, true_tokens):
    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)
    # Calculate the number of common tokens between prediction and ground truth
    common_tokens = pred_counter & true_counter
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0.0
    # Precision: proportion of predicted tokens that are correct
    precision = num_same / len(pred_tokens)
    # Recall: proportion of ground truth tokens that are predicted
    recall = num_same / len(true_tokens)
    # F1 score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_max_f1_for_question(pred, ground_truths):
    pred_tokens = word_tokenize(pred)
    max_f1 = 0
    for true_answer in ground_truths:
        true_tokens = word_tokenize(true_answer)
        f1 = calculate_f1_score(pred_tokens, true_tokens)
        max_f1 = max(max_f1, f1)
    return max_f1

def calculate_f1_over_questions(predictions, all_ground_truths):
    total_f1 = 0
    num_questions = len(predictions)
    for i in range(num_questions):
        max_f1 = calculate_max_f1_for_question(predictions[i], all_ground_truths[i])
        total_f1 += max_f1
    average_f1 = total_f1 / num_questions
    return average_f1




def calculate_exact_match(pred, ground_truths):
    # Check if the prediction matches exactly with any of the ground truth answers
    return int(pred.strip().lower() in [gt.strip().lower() for gt in ground_truths])

def calculate_exact_match_for_questions(predictions, all_ground_truths):
    total_exact_match = 0
    num_questions = len(predictions)
    for i in range(num_questions):
        exact_match = calculate_exact_match(predictions[i], all_ground_truths[i])
        total_exact_match += exact_match
    # Calculate the average exact match score
    average_exact_match = total_exact_match / num_questions
    return average_exact_match


# Example usage
# predictions = ["the quick brown fox", "jumps over", "the lazy dog"]
# ground_truths = [["the quick brown fox jumps", "a quick brown fox"], ["jumps over the lazy", "over"], ["the lazy dog", "lazy dog sleeps"]]
# print("Average F1 Score:", average_f1_over_questions(predictions, ground_truths))


'''
import requests
import time

def generate_chat_completion(prompt, model="gpt-3.5-turbo", temperature=1, max_tokens=None):
    API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    messages = [{"role": "user", "content": prompt}]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    
    while True: 
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            time.sleep(20)
            # raise Exception(f"Error {response.status_code}: {response.text}")
'''
