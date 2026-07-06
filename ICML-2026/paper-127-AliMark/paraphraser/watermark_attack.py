import os
import random

import openai
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from transformers import T5ForConditionalGeneration, T5Tokenizer

from paraphraser.evaluation.tools.text_editor import DipperParaphraser
from paraphraser.watermark_attack_semstamp import WatermarkAttackSemStamp


class WatermarkAttack:
    def __init__(self, 
                 watermark_model_name, 
                 load_dipper_paraphraser=True, 
                 load_semstamp_paraphraser=True
                 ):
        super(WatermarkAttack, self).__init__()

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self._openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
        else:
            print("WARNING: OPENROUTER_API_KEY not set. GPT-3.5 attacks will be unavailable.")
            self._openrouter_client = None

        if load_dipper_paraphraser:
            try:
                self._dipper_paraphraser = DipperParaphraser(
                    tokenizer=T5Tokenizer.from_pretrained("google/t5-v1_1-xxl"),
                    model=T5ForConditionalGeneration.from_pretrained("kalpeshk2011/dipper-paraphraser-xxl", device_map='auto'),
                    lex_diversity=60, order_diversity=0, sent_interval=1, 
                    max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)
            except Exception as e:
                print(f"WARNING: Failed to load DIPPER: {e}")
                print("DIPPER attacks unavailable. Continuing without DIPPER.")
                self._dipper_paraphraser = None
        
        if load_semstamp_paraphraser:
            self._semstamp_paraphraser = WatermarkAttackSemStamp(watermark_model_name=watermark_model_name)

    def pegasus_paraphrase_attack(self, text: str, bigram: bool) -> str:
        return self._semstamp_paraphraser.pegasus_paraphrase(text, bigram=bigram)

    def parrot_paraphrase_attack(self, text: str, bigram: bool) -> str:
        return self._semstamp_paraphraser.parrot_paraphrase(text, bigram=bigram)
    
    def dipper_paraphrase_attack(self, text: str, mode="all") -> str:
        assert self._dipper_paraphraser is not None, "DIPPER paraphraser not initialized."

        if mode == "all":
            return self._dipper_paraphraser.edit(text, "")
        elif mode == "sep":
            sentences = sent_tokenize(text)
            edited_sentences = [self._dipper_paraphraser.edit(sent, "") for sent in sentences]
            return " ".join(edited_sentences)
        else:
            raise ValueError("Invalid mode. Choose 'all' or 'sep'.")

    def gpt35_turbo_paraphrase_attack(self, text: str) -> str:
        if self._openrouter_client is None:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set. Required for GPT-3.5 attacks.")
        user_prompt = f"""
        Please rewrite the following text, avoiding the use of same words or phrases as the original text as much as possible. You are able to merge or split sentences but must preserve the original meaning:
        {text}
"""
        while True:
            try: 
                response = self._openrouter_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant to rewrite the text.",
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                            }
                    ],
                    temperature=1,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            except openai.APIError as e:
                print(f"OpenRouter API error occurred: {e} | Retrying...")
                continue
            break
        return response.choices[0].message.content

    def probing_insert(self, text: str, rate: float) -> str:
        # insert random [. ? !] to between two sentence based on the rate
        
        sentences = sent_tokenize(text)
        # insert rate * (len(sentences)) times of [. ? !] between sentences
        num_inserts = int(rate * len(sentences))
        insert_positions = random.sample(range(1, len(sentences)), num_inserts)
        new_sentences = []
        for i in range(len(sentences)):
            new_sentences.append(sentences[i])
            if i in insert_positions:
                new_sentences.append(random.choice([" . ", " ? ", " ! "]))
        return " ".join(new_sentences)
    
    def probing_delete(self, text: str, rate: float) -> str:
        # delete the sentences based on the rate
        sentences = sent_tokenize(text)

        # insert rate * (len(sentences)) times of [. ? !] between sentences
        num_deletes = int(rate * len(sentences))
        if num_deletes >= len(sentences):
            num_deletes = len(sentences) - 1
        delete_positions = set(random.sample(range(len(sentences)), num_deletes))

        new_sentences = []
        for i in range(len(sentences)):
            if i not in delete_positions:
                new_sentences.append(sentences[i])
        
        if not new_sentences:
            return sentences[-1]

        return " ".join(new_sentences)

    def probing_reorder(self, text: str, rate: float) -> str:
        # reorder the sentences based on the rate
        sentences = sent_tokenize(text)

        num_reorders = int(rate * len(sentences))
        if num_reorders >= len(sentences):
            num_reorders = len(sentences) - 1
        reorder_positions = set(random.sample(range(len(sentences)), num_reorders))

        sentences_to_reorder = [sentences[i] for i in reorder_positions]
        random.shuffle(sentences_to_reorder)

        new_sentences = []
        reorder_index = 0
        for i in range(len(sentences)):
            if i in reorder_positions:
                new_sentences.append(sentences_to_reorder[reorder_index])
                reorder_index += 1
            else:
                new_sentences.append(sentences[i])

        return " ".join(new_sentences)
    

if __name__ == "__main__":
    attack = WatermarkAttack("facebook/opt-1.3b", load_dipper_paraphraser=False, load_semstamp_paraphraser=False)
    text = """
Pujols has 3,865 hits in his career -- a total he eclipsed during Wednesday night's 9-7 victory over the Orioles at Comerica Park. His 712th was a two-out, 
single homer that cut the O's lead to 6-4 going into the ninth in Friday night's final regular-season contest.                                              
                                                                                                                                                            
"I don't think anyone's come this far because somebody hit them out of the park," Pujols said. "I think it's something you had to grind and grind and grind.
"                                                                                                                                                           
                                                                                                                                                            
Pitchers and catchers report to camp on Feb. 14.                                                                                                            
                                                                                                                                                            
The Angels' outfield is locked up, with second baseman David Freese (right hamstring) and outfielder Michael Young (neck) returning from the disabled list t
his week.                                                                                                                                                   
                                                                                                                                                            
Third baseman Mike Trout will be going into his ninth major league season.                                                                                  
                                                                                                                                                            
He was a key contributor during the Angels' American League West championship in 2010.                                                                      
                                                                                                                                                            
Trout, who was voted MVP, hit .285 this season with 27 homers and 103 RBIs in 131 games. 
"""

    print(len(sent_tokenize(text)))
    result = attack.gpt35_turbo_paraphrase_attack(text)
    print("GPT-3.5-turbo-paraphrase: \n", result)
    print(len(sent_tokenize(result)))