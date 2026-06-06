import os
import json
import time
from tqdm import tqdm
from joblib import delayed, Parallel
from openai import OpenAI
from .logging_utils import get_logger

logger = get_logger("eval_utils")


def run_single_api_prediction(example, model, max_request, generation_config):
    prompt = example['input']
    repeat = 0
    pred = ''

    while not pred and repeat < max_request:
        if isinstance(prompt, str):
            in_text = prompt
        else:
            raise TypeError(f"Unrecognized type for example input: {type(prompt)}")

        try:
            pred = model.generate(in_text, **generation_config)
        except Exception as e:
            repeat += 1
            logger.error(f"Cool down for try {repeat}. Exception: {e}")
            time.sleep(model.API_cooldown)

    if repeat == max_request:
        pred = "EMPTY PREDICTION DUE TO REPEADTED EXCEPTIONS"
    
    return {'id': example['id'], 'prompt': in_text, 'prediction': pred, 'answers': example['answers']}


def run_api_predictions(data, model, max_request, generation_config, disable_tqdm=False, n_jobs=1):
    preds = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(run_single_api_prediction)(
            example, model, max_request, generation_config)
        for example in tqdm(data, disable=disable_tqdm))

    return preds


class Evaluator():
    def __init__(self, model, data, max_request=12, inference_batchsize=4, generation_config={}, 
                 skip_metrics=False, model_eval_name=None, output_path="", num_passages=10,
                 mbe_eval_type='standard'):
        self.model = model
        self.data = data
        self.max_request = max_request
        self.inference_batchsize= inference_batchsize
        self.generation_config = generation_config
        self.skip_metrics = skip_metrics
        self.model_eval_name = model_eval_name
        self.model_based_output_path = output_path
        self.num_passages = num_passages
        self.mbe_eval_type = mbe_eval_type

    def run_predictions(self, metric):
        preds = run_api_predictions(data=self.data, model=self.model, max_request=self.max_request,
                                    generation_config=self.generation_config, n_jobs=self.inference_batchsize)
        if self.skip_metrics:
            metrics = {}
            logger.info("Metrics calculation is skipped due to config from dataloader or CLI.")
        else:
            predictions, references = [p['prediction'] for p in preds], [p['answers'] for p in preds]
            metrics = metric.compute(predictions=predictions, references=references)

        logger.info(metrics)
        return preds, metrics

    def save_predictions(self, predictions, metrics, output_path, suffix=""):
        with open(f"{output_path}/eval_predictions{suffix}.json", 'w') as outfile:
            json.dump(predictions, outfile, indent=2)
        
        if not self.skip_metrics:
            with open(f"{output_path}/eval_metrics{suffix}.json", 'w') as outfile:
                json.dump(metrics, outfile, indent=2)



class OpenAIAPIModel:
    API_cooldown = 1
    def __init__(self, model_name, max_new_tokens):
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            self.model = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        else:
            self.model = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_remap = {
            'chatgpt': 'gpt-3.5-turbo',
            'deepseek-chat': 'deepseek/deepseek-chat',
        }
        self.model_name = model_remap.get(model_name, model_name)

        self.max_token = max_new_tokens

    def generate(self, input, **kwargs):
        messages = []
        messages.append({"role": "user", "content": input})
        ans = self.model.chat.completions.create(model=self.model_name, 
                                                 messages=messages,
                                                 max_tokens=self.max_token,
                                                 top_p=kwargs['top_p'],
                                                 temperature=kwargs['temperature'])
        return ans.choices[0].message.content


def load_model(model_name, max_new_tokens):
    # Accept any model name for OpenAI-compatible APIs (gpt-*, deepseek/*, etc.)
    return OpenAIAPIModel(model_name=model_name, max_new_tokens=max_new_tokens)