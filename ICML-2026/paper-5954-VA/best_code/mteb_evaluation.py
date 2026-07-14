from mteb import MTEB
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, LlamaConfig, AutoModelForCausalLM
import torch
from torch import nn
from transformers import BertLMHeadModel, BertTokenizer, RobertaTokenizer, RobertaForMaskedLM, RobertaModel, BertModel
from tqdm import tqdm
import json
from transformers import LlamaTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from models import Value_Aggregation_Eval
from transformers import (
    AdamW,
    HfArgumentParser,
    get_scheduler,
)
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F
import os
#import deepspeed
from datasets import Dataset
from arguments_va_eval import ModelArguments, DataTrainingArguments, TrainingArguments
# from mteb.models.text_formatting_utils import corpus_to_texts
from mteb.encoder_interface import PromptType
import torch

use_auth_token = os.getenv("HUGGING_FACE_TOKEN")
task_to_prompt = {
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim:",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question:",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim:",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query:",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia:",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question:",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question:",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper:",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query:",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question:",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim:",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question:",
    "ArguAna": "Given a claim, find documents that refute the claim:",
    "CQADupstackTexRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWebmastersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackEnglishRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGisRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackMathematicaRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackStatsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackPhysicsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackProgrammersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackAndroidRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWordpressRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question:",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum:",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum:",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers:",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history:",
    "Banking77Classification": "Given a online banking query, find the corresponding intents:",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise:",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral:",
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or notcounterfactual:",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset:",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents:",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios:",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation:",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation:",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic:",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment:",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category:",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles:",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles:",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles:",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts:",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles:",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts:",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts:",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles:",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts:",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles:",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs:",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet:",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet:",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum:",
    "STS12": "Retrieve semantically similar text:",
    "STS13": "Retrieve semantically similar text:",
    "STS14": "Retrieve semantically similar text:",
    "STS15": "Retrieve semantically similar text:",
    "STS16": "Retrieve semantically similar text:",
    "STS17": "Retrieve semantically similar text:",
    "STS22": "Retrieve semantically similar text:",
    "BIOSSES": "Retrieve semantically similar text:",
    "SICK-R": "Retrieve semantically similar text:",
    "STSBenchmark": "Retrieve semantically similar text:",
    "SummEval": "Given a news summary, retrieve other semantically similar summaries:"
}
class MyModel():
    def __init__(self, model, tokenizer, data_args) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.gpu_count = torch.cuda.device_count() 
        self.max_length = data_args.max_length
        self.batch_size = 16
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()
        self.prompt = task_to_prompt[data_args.now_task]
        self.whole_length = 0

    def encode(self, sentences, **kwargs):
        if 'prompt_type' in kwargs and kwargs['prompt_type']==PromptType.query:
            sentences = [self.prompt + '\n' + text for text in sentences]
        if 'prompt_type' not in kwargs:
            sentences = [self.prompt + '\n' + text for text in sentences]
        with torch.no_grad():
            output_embedding = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches"):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                output_embedding.append(self.value_aggregation(sentences_batch).cpu())
            return torch.cat(output_embedding, dim=0)
    
    def value_aggregation(self, sentences_batch):
        inputs = self.tokenizer(sentences_batch, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
        inputs = {k:v.cuda() for k,v in inputs.items()}
        embedding = torch.cat(self.model(inputs['input_ids'], inputs['attention_mask']), dim=0)
        return embedding.to(torch.float32)
    
   
    

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=use_auth_token)
if 'llama' in model_args.model_name.lower(): 
    tokenizer.pad_token = tokenizer.eos_token
model = Value_Aggregation_Eval(training_args.local_rank)
model.to(torch.bfloat16) 
checkpoint_path = model_args.checkpoint_path
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['module'], strict=True)
for each_task in ['Banking77Classification', 'EmotionClassification','ArguAna', 'SciFact', 'STS17', 'NFCorpus', 'SICK-R', 'STSBenchmark', 'MedrxivClusteringS2S', 'StackOverflowDupQuestions', 'TwentyNewsgroupsClustering', 'BiorxivClusteringS2S', 'SciDocsRR', 'SprintDuplicateQuestions']:
    mymodel = MyModel(model, tokenizer, data_args, each_task)
    evaluation = MTEB(tasks=[each_task])
    results = evaluation.run(mymodel, output_folder=f"12_3_finetuned_VA", eval_splits=["test"])