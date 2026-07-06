import re
from collections import Counter

import backoff
import openai
import torch
from bert_score import BERTScorer
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else "cpu"
from parrot import Parrot

# stops = set(stopwords.words('english'))
stops = []

_scorer = None

def _get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = BERTScorer(model_type = "microsoft/deberta-xlarge-mnli", rescale_with_baseline=True, device=device, lang = "en")
    return _scorer

def run_bert_score(gen_sents, para_sents):
    P, R, F1 = _get_scorer().score(gen_sents, para_sents)
    return torch.mean(F1).item()

# We slightly modify the Parrot class to make it more suitable for our use case
class SParrot(Parrot):
    def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False):
        super().__init__(model_tag, use_gpu)
    
    def augment(self, input_phrase, use_gpu=False, diversity_ranker="levenshtein", do_diverse=False, max_return_phrases = 10, max_length=32, adequacy_threshold = 0.90, fluency_threshold = 0.90):
      if use_gpu:
        device= "cuda:0"
      else:
        device = "cpu"

      self.model     = self.model.to(device)

      import re

      save_phrase = input_phrase
      if len(input_phrase) >= max_length:
         max_length += 32	
			
      input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
      input_phrase = "paraphrase: " + input_phrase
      input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
      input_ids = input_ids.to(device)

      if do_diverse:
        for n in range(2, 9):
          if max_return_phrases % n == 0:
            break 
        #print("max_return_phrases - ", max_return_phrases , " and beam groups -", n)            
        preds = self.model.generate(
              input_ids,
              do_sample=False, 
              max_length=max_length, 
              num_beams = max_return_phrases,
              num_beam_groups = n,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=max_return_phrases,
              trust_remote_code=True)
      else: 
        preds = self.model.generate(
                input_ids,
                do_sample=True, 
                max_length=max_length, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=max_return_phrases) 
        

      paraphrases= set()

      for pred in preds:
        gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
        gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
        paraphrases.add(gen_pp)


      # Skip quality filters if not available (patched for performance)
      if hasattr(self, 'adequacy_score') and hasattr(self, 'fluency_score') and hasattr(self, 'diversity_score'):
          adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold, device )
          if len(adequacy_filtered_phrases) == 0 :
            adequacy_filtered_phrases = paraphrases
          fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold, device )
          if len(fluency_filtered_phrases) == 0 :
              fluency_filtered_phrases = adequacy_filtered_phrases
          diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
          para_phrases = []
          for para_phrase, diversity_score in diversity_scored_phrases.items():
              para_phrases.append((para_phrase, diversity_score))
          para_phrases.sort(key=lambda x:x[1], reverse=True)
          para_phrases = [x[0] for x in para_phrases]
      else:
          para_phrases = list(paraphrases)
          if not para_phrases:
              para_phrases = [save_phrase]
      return para_phrases


def tokenize(tokenizer, text):
    return tokenizer(text, return_tensors='pt').input_ids[0].to(device)

def build_bigrams(input_ids):
    bigrams = []
    for i in range(len(input_ids) - 1):
        bigram = tuple(input_ids[i:i+2].tolist())
        bigrams.append(bigram)
    return bigrams

def extract_list(text):
    p = re.compile("^[0-9]+[.)\]\*·:] (.*(?:\n(?![0-9]+[.)\]\*·:]).*)*)", re.MULTILINE)
    return p.findall(text)

def compare_ngram_overlap(input_ngram, para_ngram):
    input_c = Counter(input_ngram)
    para_c = Counter(para_ngram)
    intersection = list(input_c.keys() & para_c.keys())
    overlap = 0
    for i in intersection:
        overlap += para_c[i]
    return overlap

def accept_by_bigram_overlap(sent, para_sents, tokenizer, bert_threshold = 0.03):
    input_ids = tokenize(tokenizer, sent)
    input_bigram = build_bigrams(input_ids)
    para_ids = [tokenize(tokenizer, para) for para in para_sents]
    para_bigrams = [build_bigrams(para_id) for para_id in para_ids]
    min_overlap = len(input_ids)

    bert_scores = [run_bert_score([sent], [para_sent]) for para_sent in para_sents]
    max_score = bert_scores[0]
    best_paraphrased = para_sents[0]
    score_threshold = bert_threshold * max_score
    for i in range(len(para_bigrams)):
        para_bigram = para_bigrams[i]
        overlap = compare_ngram_overlap(input_bigram, para_bigram)
        bert_score = bert_scores[i]
        diff = max_score - bert_score
        if overlap < min_overlap and len(para_ids[i]) <= 1.5 * len(input_ids) and (diff <= score_threshold):
            min_overlap = overlap
            best_paraphrased = para_sents[i]
    return best_paraphrased

def gen_prompt(sent, context):
  prompt = f'''Previous context: {context} \n Current sentence to paraphrase: {sent}'''
  return prompt

def gen_prompt_together(text):
  prompt = f'''Please paraphrase the following text while preserving its original meaning. You may choose to merge or split a few sentences, but you should be quite cautious when doing this in order to preserving the original meaning. Output only the rewritten text, nothing else:
{text}'''
  return prompt

def gen_bigram_prompt(sent, context, num_beams):
  prompt = f'''Previous context: {context} \n Paraphrase in {num_beams} different ways and return a numbered list : {sent}'''
  return prompt
  
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai(client, prompt):
  while True:
    try:
      response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except openai.APIError:
      continue
    break
  return response.choices[0].message.content

# use long context
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai_bigram(client, prompt):
  while True:
    try:
      response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except openai.APIError:
      continue
    break
  return response.choices[0].message.content

# pick the paraphrases by openai
def pick_para(sent_list, tokenizer, all_paras, thres):
    # all_paras is a list shape: num of texts X num of paraphrases X number of beams
    data_len, para_texts, para_texts_bigram = [], [], [], []
    data_len = [len(t) for t in sent_list]

    for i in trange(len(sent_list), desc = "Picking paraphrases"):
        sents = sent_list[i]
        for j in range(len(sents)):
            sent = sents[j]
            # each sent has num_beams paraphrases
            paraphrases = all_paras[i][j] # all beams
            para = accept_by_bigram_overlap(sent, paraphrases, tokenizer, bert_threshold=thres)
            para_texts_bigram.append(para)
            para_texts.append(paraphrases[0])
    output_no_bigram = []
    output_bigram = []
    # new_texts = []
    start_pos = 0
    for l in data_len:
        output_no_bigram.append(para_texts[start_pos: start_pos+l])
        output_bigram.append(para_texts_bigram[start_pos: start_pos+l])
        start_pos+=l
    return output_no_bigram, output_bigram