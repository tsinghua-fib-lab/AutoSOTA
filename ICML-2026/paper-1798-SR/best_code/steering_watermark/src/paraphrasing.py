from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
from transformers import BitsAndBytesConfig

import torch
import numpy as np
import time

from data_processing import cached_function2


def paraphrasing_data_ordering(params):
    # Get the parameters as input and give back the path to the according data
    data_name = params['data_arguments']['input_version']
    model_name = params['model_arguments']['model_id'].split('/')[-1]
    generation_style = f"gen_{params['generation_arguments']['max_new_tokens']}_temp{params['generation_arguments']['temperature']}"
    if float(params['steering_arguments']['noise_max']) == 0.0:
        steering_style = params["comparison_arguments"]["compared_text_type"]
    else:
        steering_style = f"steering_noise{params['steering_arguments']['noise_type']}_{params['steering_arguments']['noise_max']}_layers{'-'.join(map(str, params['steering_arguments']['steering_layers']))}"
    path = f"data/{data_name}/{model_name}/{generation_style}/{steering_style}/"
    return path


def paraphrase_texts(df, params):
    # Curate the usefull parameters for the caching system
    curated_params = {
        "parameters_type": "text_paraphrasing",
        "data_ordering_function": paraphrasing_data_ordering(params),
        "model_arguments": params["model_arguments"],
        "data_arguments": params["data_arguments"],
        "generation_arguments": params["generation_arguments"],
        "steering_arguments": params["steering_arguments"],
        "robustness_arguments": {"paraphrasing": params["robustness_arguments"]["paraphrasing"]},
        "verbose": params.get("verbose", True),
        "hf_token": params["hf_token"],
        "split_labels": params["split_labels"] if "split_labels" in params else None,
    }
    return paraphrase_texts_cached(df, curated_params)



class DipperParaphraser(object):
    def __init__(self, params):
        time1 = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.model_name = params["robustness_arguments"]["paraphrasing"]["model_id"]
        if params["robustness_arguments"]["paraphrasing"].get("use_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False, #True  # Nested quantization for better accuracy
                bnb_4bit_quant_type="fp4", #"nf4",       # NormalFloat4
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )

        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
            # load_in_8bit=False,
            # load_in_4bit=True
        ) #.to(self.device)
        if params.get("verbose", True):
            print(f"{self.model_name} model loaded in {time.time() - time1}")
        # self.model.to(self.device)
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text, language="english")
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt").to(self.device)
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text
    
    def paraphrase_batch(self, input_texts):
        paraphrased_texts = []
        for text in input_texts:
            # Default parameters for diversity
            paraphrased_text = self.paraphrase(
                text, 
                lex_diversity=self.params["robustness_arguments"]["paraphrasing"]["lexical_diversity"], 
                order_diversity=self.params["robustness_arguments"]["paraphrasing"]["order_diversity"], 
                prefix="", 
                do_sample=True, 
                # top_p=self.params["robustness_arguments"]["paraphrasing"]["top_p"], 
                # top_k=self.params["robustness_arguments"]["paraphrasing"]["top_k"], 
                max_length=self.params["robustness_arguments"]["paraphrasing"]["max_length"]
            )
            paraphrased_texts.append(paraphrased_text)
        return paraphrased_texts

    

class AteeqqParaphraser(object):
    def __init__(self, params, verbose=True):
        time1 = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.model_name = params["robustness_arguments"]["paraphrasing"]["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(params["robustness_arguments"]["paraphrasing"]["model_id"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(params["robustness_arguments"]["paraphrasing"]["model_id"]).to(device)

        if verbose:
            print(f"{self.model_name} model loaded in {time.time() - time1}")
        self.model.to(self.device)
        self.model.eval()

    def paraphrase_batch(self, input_texts):
        input_texts = [f'paraphraser: {text}' for text in input_texts]
        input_ids_list = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding="longest", 
            truncation=False, 
            max_length=self.params["robustness_arguments"]["paraphrasing"]["max_length"]
        ).input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids_list,
            num_beams=self.params["robustness_arguments"]["paraphrasing"]["num_beams"],
            num_beam_groups=self.params["robustness_arguments"]["paraphrasing"]["num_beam_groups"],
            num_return_sequences=self.params["robustness_arguments"]["paraphrasing"]["num_return_sequences"],
            repetition_penalty=self.params["robustness_arguments"]["paraphrasing"]["repetition_penalty"],
            diversity_penalty=self.params["robustness_arguments"]["paraphrasing"]["diversity_penalty"],
            no_repeat_ngram_size=self.params["robustness_arguments"]["paraphrasing"]["no_repeat_ngram_size"],
            temperature=self.params["robustness_arguments"]["paraphrasing"]["temperature"],
            max_length=self.params["robustness_arguments"]["paraphrasing"]["max_length"]
        )
        print("Raw out:", outputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)






@cached_function2()
def paraphrase_texts_cached(df, params):
    # Populate a df containing generated text with a paraphrased version of the given text.
    if params["robustness_arguments"]["paraphrasing"]["model_id"].startswith("Ateeqq"):
        paraphrasing_model = AteeqqParaphraser(params)
    elif params["robustness_arguments"]["paraphrasing"]["model_id"].startswith("kalpeshk2011"):
        paraphrasing_model = DipperParaphraser(params)
    else:
        raise ValueError("Unknown paraphrasing model id.")
    
    # paraphrased_texts = []
    # # Sequential processing; can be optimized with batch processing if needed
    # bsz = params["robustness_arguments"]["paraphrasing"].get("batch_size", 1)
    # num_passes = np.ceil(len(df) / bsz)
    # # Iterate over the dataset in batches
    # for k_pass in range(int(num_passes)):
    #     texts_in_batch = []
    #     for i in range(bsz):
    #         idx = k_pass * bsz + i
    #         if idx >= len(df):
    #             break

    #         text = df["output_text"].iloc[idx]
    #         # Set up the input prompt for paraphrasing
    #         texts_in_batch.append(text)

    #     # Generate paraphrases for the batch
    #     paraphrases = paraphrasing_model.paraphrase_batch(texts_in_batch)
    #     paraphrased_texts.extend([p for p in paraphrases]) # Assuming num_return_sequences=1
    #     if params["verbose"]:
    #         for original, paraphrased in zip(texts_in_batch, paraphrases):
    #             print(f"+Original: {original}\n-Paraphrased: {paraphrased}\n")

    #################
    ## Optimized batch processing
    #################
    paraphrased_texts = []
    bsz = params["robustness_arguments"]["paraphrasing"].get("batch_size", 1)

    if params["split_labels"] is None:
        # Paraphrase all texts
        to_paraphrase_df = df.reset_index(drop=True)
        indices_to_paraphrase = df.index.tolist()
    else:
        df["split_label"] = df["input_text_id"].apply(lambda x: params["split_labels"][x])
        to_paraphrase_df = df[df["split_label"] > 0].reset_index(drop=True)
        indices_to_paraphrase = df[df["split_label"] > 0].index.tolist()

    if params["robustness_arguments"]["paraphrasing"].get("with_prompt", True):
        texts_to_paraphrase = to_paraphrase_df["input_text"] + " " + to_paraphrase_df["output_text"]
    else:
        texts_to_paraphrase = to_paraphrase_df["output_text"]

    # Cleaner batching approach
    for batch_start in range(0, len(texts_to_paraphrase), bsz):
        batch_end = min(batch_start + bsz, len(texts_to_paraphrase))
        texts_in_batch = texts_to_paraphrase.iloc[batch_start:batch_end].tolist()
        
        # Generate paraphrases for the batch
        paraphrases = paraphrasing_model.paraphrase_batch(texts_in_batch)
        paraphrased_texts.extend(paraphrases)  # Remove redundant comprehension
        
        if params["verbose"]:
            for original, paraphrased in zip(texts_in_batch, paraphrases):
                print(f"+Original: {original}\n-Paraphrased: {paraphrased}\n")

    del paraphrasing_model  # Free up memory



    # Simply replace the output_text with the paraphrased version to keep same pipeline structure dowstream
    df["original_text"] = df["output_text"]
    df["paraphrased_text"] = "non-paraphrased"
    df.loc[indices_to_paraphrase, "paraphrased_text"] = paraphrased_texts

    # TODO: Add any additional metadata, such as token string values input/output
    print("***********************")
    print("Todo: Add any additional metadata, such as token string values input/output, etc.")
    print("Todo: check proper datapipeline flow -> Original_text vs Paraphrased_text")
    print("***********************")

    return df

    

if __name__ == "__main__":

    params = {
        "robustness_arguments": {
            "paraphrasing": {
                "model_id": "kalpeshk2011/dipper-paraphraser-xxl"
            }
        },
        "verbose": True
    }

    dp = DipperParaphraser(params)

    prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote valley."
    input_text = "They have never been known to mingle with humans. Today, it is believed these unicorns live in an unspoilt environment which is surrounded by mountains. Its edge is protected by a thick wattle of wattle trees, giving it a majestic appearance. Along with their so-called miracle of multicolored coat, their golden coloured feather makes them look like mirages. Some of them are rumored to be capable of speaking a large amount of different languages. They feed on elk and goats as they were selected from those animals that possess a fierceness to them, and can \"eat\" them with their long horns."


    print(f"Input = {prompt} <sent> {input_text} </sent>\n")
    output_l60_sample = dp.paraphrase(
        input_text, 
        lex_diversity=60, 
        order_diversity=0, 
        prefix=prompt, 
        do_sample=True, 
        top_p=0.75, 
        top_k=None, 
        max_length=512
    )
    print(f"Output (Lexical diversity = 60, Sample p = 0.75) = {output_l60_sample}\n")
