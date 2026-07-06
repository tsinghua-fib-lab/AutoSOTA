import openai
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import (AutoTokenizer, PegasusForConditionalGeneration,
                          PegasusTokenizer)

# from dipper import DipperParaphraser
from paraphraser.semstamp_paraphrasers import (SParrot,
                                               accept_by_bigram_overlap,
                                               extract_list, gen_bigram_prompt,
                                               gen_prompt, gen_prompt_together,
                                               query_openai,
                                               query_openai_bigram)

PUNCTS = '!.?'
from itertools import groupby
from string import punctuation


def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]

def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)

def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return clean_text(sent)

class WatermarkAttackSemStamp:
    def __init__(self, watermark_model_name="facebook/opt-1.3b", device="cuda"):
        self._device = device
        self._num_beams = 10
        self._max_iter = 10

        self._parrot = SParrot()
        self._pegasus = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase").to(self._device)
        self._pegasus_tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")

        self._tokenizer = AutoTokenizer.from_pretrained(watermark_model_name)

    def _pegasus_helper(self, sents):
        '''
        Arguments:
            sents: list of sentences (max len under 60!)
        Returns:
            paraphrased: list of paraphrased sents
        '''
        batch = self._pegasus_tokenizer(
            sents, truncation=True, padding='longest', return_tensors="pt", max_length=60).to(self._device) # modified max_length to 512

        paraphrased_ids = self._pegasus.generate(
            **batch, max_length=60, num_beams=self._num_beams, num_return_sequences=self._num_beams, temperature=2.0, do_sample=True, repetition_penalty=1.03)
        # batch decode and return the first one
        paraphrased = [self._pegasus_tokenizer.decode(paraphrased_ids[i*self._num_beams], skip_special_tokens=True) for i in range(len(paraphrased_ids) // self._num_beams)]
        # breakpoint()

        return paraphrased
    
    def pegasus_paraphrase(self, text, bigram=False, together=False):
        if together:
            sents = [text]
        else:
            sents = sent_tokenize(text)
        paras = []
        for sent in tqdm(sents):
            paraphrased = self._pegasus_helper([sent])
            paraphrased = [well_formed_sentence(para) for para in paraphrased]
            if bigram:
                para = accept_by_bigram_overlap(sent, paraphrased, self._tokenizer, bert_threshold=0.0)
            else: 
                if together:
                    para = ' '.join(paraphrased)
                else:
                    para = paraphrased[0]
            paras.append(para)
        
        paras = ' '.join(paras)
        return paras
        
    def _parrot_helper(self, text):
        para_phrases = self._parrot.augment(input_phrase=text,
                                      use_gpu=True,
                                      diversity_ranker="levenshtein",
                                      do_diverse=True,
                                      max_return_phrases=10,
                                      max_length=60,
                                      adequacy_threshold=0.8,
                                      fluency_threshold=0.8)
        return para_phrases

    def parrot_paraphrase(self, text, bigram=False, together=False):
        if together:
            sents = [text]
        else:
            sents = sent_tokenize(text)
        paras = []
        for sent in tqdm(sents):
            paraphrased = self._parrot_helper(sent)
            paraphrased = [well_formed_sentence(
                para, end_sent=True) for para in paraphrased]
            if bigram:
                para = accept_by_bigram_overlap(sent, paraphrased, self._tokenizer, bert_threshold=0.0)
            else:
                if together:
                    para = ' '.join(paraphrased)
                else:
                    para = paraphrased[0]
            paras.append(para)
        paras = ' '.join(paras)
        return paras


if __name__ == "__main__":
    attacker = WatermarkAttackSemStamp()
    # sample_text = "This is a sample text to be paraphrased. It contains multiple sentences."
    sample_text = "In the realm of natural language processing, the ability to paraphrase text effectively is a crucial skill. This involves not only changing the words used but also maintaining the original meaning and context. Advanced models like GPT-3.5 have shown remarkable capabilities in this area, allowing for nuanced and contextually appropriate rewording of sentences. The challenge lies in ensuring that the paraphrased text remains coherent and retains the intent of the original message, which is essential for applications such as content creation, summarization, and translation."
    print("Parrot")
    parrot_paraphrased = attacker.parrot_paraphrase(sample_text, bigram=False)
    parrot_bigram_paraphrased = attacker.parrot_paraphrase(sample_text, bigram=True)
    print("Pegasus")
    pegasus_paraphrased = attacker.pegasus_paraphrase(sample_text, bigram=False)
    pegasus_bigram_paraphrased = attacker.pegasus_paraphrase(sample_text, bigram=True)

    print("Original Text:", sample_text)
    print("Parrot Paraphrased Text:\n", parrot_paraphrased, "\n")
    print("Parrot Bigram Paraphrased Text:\n", parrot_bigram_paraphrased, "\n")
    print("Pegasus Paraphrased Text:\n", pegasus_paraphrased, "\n")
    print("Pegasus Bigram Paraphrased Text:\n", pegasus_bigram_paraphrased, "\n")
