from pydoc import text
import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
except ImportError:
    Mistral3ForConditionalGeneration = None
    FineGrainedFP8Config = None
try:
    from transformers import MistralCommonBackend, AutoConfig
except ImportError:
    MistralCommonBackend = None
    AutoConfig = None
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from datasets import Column

import copy


class LLMWrapper:
    """
        A Wrapper for LLM model from Hugging face library with simple interface.
        It also enable Hook usage to call-back activation at inference.
    """
    def __init__(self, **kwargs):
        self.model_name = kwargs["model_id"]
        self.arguments = kwargs
        self.embedding_dim = None
        
        self.model = None
        self.tokenizer = None
        self.load_model(self.arguments)
        # self.tokenizer = self.pipeline.tokenizer
        # Move to "not Mistral section" only
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.embedding_dim = self.pipeline.model.model.embed_tokens.embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name: str):
        """
            Load the model from Hugging Face Transformers library.
        """
        # If load in quantize way, No device option
        if self.model_name.startswith("mistralai/Ministral-3") :
            # Load model + tokenizer
            self.tokenizer = MistralCommonBackend.from_pretrained(self.model_name)
            # tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = torch.compile(
                transformers.Mistral3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    # torch_dtype=torch.bfloat16,
                    quantization_config=FineGrainedFP8Config(dequantize=True)
                )
            )

            # Set embedding dim
            self.embedding_dim = self.model.get_input_embeddings().embedding_dim
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        else:
            # # TODO: update according to new synthax quantize_args
            # if ( "load_in_8bit" in self.arguments and self.arguments["load_in_8bit"] ) or ( "load_in_4bit" in self.arguments and self.arguments["load_in_4bit"] ):
            #     pipeline = transformers.pipeline(
            #         "text-generation", 
            #         model=self.model_name, 
            #         model_kwargs={
            #             "torch_dtype": eval(self.arguments["torch_dtype"]),
            #             "load_in_8bit": self.arguments["load_in_8bit"] if "load_in_8bit" in self.arguments else False,
            #             "load_in_4bit": self.arguments["load_in_4bit"] if "load_in_4bit" in self.arguments else False,
            #         }, 
            #         device_map=None,  # disables offloading
            #         token=self.arguments["hf_token"],
            #     )
            # else :
            #     pipeline = transformers.pipeline(
            #         "text-generation", 
            #         model=self.model_name, 
            #         model_kwargs={
            #             "torch_dtype": eval(self.arguments["torch_dtype"]),
            #         }, 
            #         device_map=None,  # disables offloading
            #         token=self.arguments["hf_token"],
            #         device="cuda" if torch.cuda.is_available() else "cpu",
            #     )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=eval(self.arguments.get("torch_dtype", "torch.bfloat16")),
                device_map="cuda:0",
                token=self.arguments["hf_token"]
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
                padding_side="left",
                token=self.arguments["hf_token"]
            )

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.embedding_dim = self.model.model.embed_tokens.embedding_dim

            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id



    def gathering_forward(self, input_batch, **gen_args):
        """
            input_batch: List of strings (generated text with prompt/start of sentence)
            Output: Model outputs with activations gathered by hooks
        """

        # Encoding inputs
        inputs = self.tokenizer(
            input_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate outputs
        outputs = self.model.generate(
            **inputs,
            **gen_args
        )
        return outputs


    def generate(self, input_batch, rich_output=False, **gen_args):
        """
            Input: List of strings (prompts)
            Output: List of strings (generated texts)
        """

        # print(">>> initial input batch:", input_batch)

        # Encoding inputs
        if isinstance(input_batch[0], list) :
            # Prepare inputs with chat template if needed. When generating on questions. Not when gathering activations


            text_prompts = [input_batch[i][1]["content"] for i in range(len(input_batch))]

            # 1) Chat template → strings
            tok_out = self.tokenizer.apply_chat_template(
                input_batch,
                return_tensors="pt",
                tokenize=True,
                padding=True,
                truncation=True,
                add_generation_prompt=True,
            )
            # Move to device (handle both tensor and dict cases)
            if isinstance(tok_out, dict):
                tokenized_prompts = {k: v.to(self.model.device) for k, v in tok_out.items()}
                input_ids_tensor = tokenized_prompts["input_ids"]
            else:
                input_ids_tensor = tok_out.to(self.model.device)
                tokenized_prompts = {"input_ids": input_ids_tensor}

        else :
            text_prompts = [input_batch[i] for i in range(len(input_batch))]

            tok_out = self.tokenizer(
                input_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            tokenized_prompts = {k: v.to(self.device) for k, v in tok_out.items()}
            input_ids_tensor = tokenized_prompts["input_ids"]

        # Generate outputs
        outputs = self.model.generate(
            **tokenized_prompts,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_args
        )

        # Decode outputs
        texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=False,
        )

        token_id_only_generated = outputs[:, input_ids_tensor.shape[1]:]
        # print(">>> generated outputs only (token ids):", token_id_only_generated)


        texts_gen_only = self.tokenizer.batch_decode(
            token_id_only_generated,
            skip_special_tokens=True,
        )

        tok_out = self.tokenizer(
            texts_gen_only,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        tokenized_output_only = {k: v.to(self.device) for k, v in tok_out.items()}

        for (user_input, gen_text) in zip(text_prompts, texts_gen_only):
            print(f"=== Input Prompt {user_input} ===\n")
            print(">> Answer: \n", gen_text)

        print()

        if rich_output:
            output_list = []
            # Loop over the batch to create detailed outputs
            for idx, (in_text, out_text) in enumerate(zip(text_prompts, texts_gen_only)):
                input_ids = tokenized_prompts["input_ids"][idx]
                output_ids = tokenized_output_only["input_ids"][idx]
                # Remove padding tokens from output_ids
                pad_id = self.tokenizer.pad_token_id
                if pad_id is not None:
                    output_ids = output_ids[output_ids != pad_id]
                output_dict = {
                    "input_text": in_text,
                    "generated_texts": out_text,
                    "encoded_inputs": input_ids,
                    "input_lengths": int((input_ids != pad_id).sum()) if pad_id is not None else len(input_ids),
                    "encoded_outputs": output_ids,
                    "output_lengths": len(output_ids),
                    "output_token_strings": [self.decode(id) for id in output_ids],
                }
                output_dict["input_token_strings"] = output_dict["output_token_strings"][:output_dict["input_lengths"]]
                output_list.append(output_dict)
            return output_list
        
        # Return simple list of generated texts
        else :
            return texts


    def __call__(self, input_dataset, rich_output=False, **kwargs):
        """
            Call the model with input text and additional arguments.
            
            Args:
                input_text (List[str]): Input text to generate output from the model.
                **kwargs: Additional arguments for the model.
                
            Returns:
                str: Generated text from the model.
        """

        # Create batches of the input text
        def chunks(lst, n):
            # Gives successive n-sized chunks from lst. Last part may be smaller.
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        with torch.inference_mode():
            # if self.model_name.startswith("mistralai/Ministral-3") or self.model_name.startswith("meta-llama/Meta-Llama-3-8B-Instruct") :
            output = []
            for batch in chunks(input_dataset, kwargs.pop("batch_size", 1)) :
                # kwargs.pop("batch_size", None)
                output.extend(self.generate(batch, rich_output=rich_output, **kwargs))
            
            return output


            # else :
            #     kwargs.pop("batch_size", 1)
            #     output = self.generate(input_dataset, rich_output=rich_output, **kwargs)
            #     return output
    


    def get_processed_output(self, model_output):
        return model_output[0]['generated_text']




    def encode(self, text, **kwargs):
        """
            Encode the input text (List[str]) into token IDs (List[List[int]]).
        """
        if self.model_name.startswith("mistralai/Ministral-3") :
            encoded = self.tokenizer(text, return_tensors="pt", padding=False, truncation=False).to(self.device)
            # encoded: dict{input_ids: tensor([[....]]), attention_mask: tensor([[....]])}
            # encoded["input_ids"]: tensor of shape (batch_size, [1, id] * seq_len)

            # print("Text to encode:", text)
            # print("Encoded:", encoded)
            # print("Encoded input IDs:", encoded['input_ids'])
            # print("Encoded input IDs shape:", encoded['input_ids'][0])

            encoded_outputs = [encoded['input_ids'][i] for i in range(len(encoded['input_ids']))]
            return encoded_outputs
        else:
            return self.tokenizer(text, **kwargs).to(self.device)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def get_layers(self):
        if self.model_name.startswith("mistralai/Ministral-3") :
            return self.model.model.language_model.layers
        else:
            return self.model.model.layers

    def get_shape_type(self):
        # Turns out by doing home made batches the dimentions are always same as 3d
        return "3d"
        # else:
        #     return "4d"


    def register_hooks(self, hook_type, layer_index, steering_vector=None):
        """
            Register hooks to the model for gathering activations or steering outputs.
            
            Args:
                hook_type (str): Type of hook to register, either "gather" or "steering".
                layer_index (list): List of layer indices to register hooks on.
                steering_vector (list, optional): Vector to steer the output if hook_type is "steering".
                
            Returns:
                list: List of registered hooks.
        """

        hooks = []
        if hook_type == "gather":
            for layer in layer_index:
                h = ActivationHook(layer_name=layer, embedding_shape_type=self.get_shape_type())
                handle = self.get_layers()[layer].register_forward_hook(h)
                h.handle = handle
                hooks.append(h)
        elif hook_type == "steering":
            for layer in layer_index:
                h = SteeringHook(layer_name=layer, steering_vector=steering_vector, embedding_shape_type=self.get_shape_type())
                handle = self.get_layers()[layer].register_forward_hook(h)
                h.handle = handle
                hooks.append(h)
        return hooks
    



#####################
#   Hook Objects    #
#####################
    

class ActivationHook:
    def __init__(self, layer_name=None, embedding_shape_type="4d"):
        self.layer_name = layer_name
        self.embedding_shape_type = embedding_shape_type
        self.activations = None  # Will hold the last forward pass
        self.handle = None  # Will hold the hook handle

    def __call__(self, module, input, output):
        # Unwrap tuple if needed (some models return (hidden_states, ...))
        if isinstance(output, tuple):
            output = output[0]
        if self.embedding_shape_type == "4d":
            output = output[0]
        self.activations = output[0].type(torch.float16).detach().cpu().numpy()  # safer to detach
        # Resulting shape: array(sequence_length, embedding_dim)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class SteeringHook:
    def __init__(self, layer_name=None, steering_vector=None, embedding_shape_type="4d"):
        self.layer_name = layer_name
        self.steering_vector = steering_vector
        self.embedding_shape_type = embedding_shape_type
        self.handle = None  # Will hold the hook handle

    def __call__(self, module, input, output):                                                          # For "3d" shape type (Mistral):
        # Unwrap if output is a tuple (some models return (hidden_states, ...))
        if isinstance(output, tuple):
            output = output[0]
        if self.embedding_shape_type == "4d":
            output = output[0]

        # Ensure steering vector is on the same device as the output
        sv = self.steering_vector.to(output.device)

        for b in range(len(output)):
            for i in range(len(output[0])):
                output[b][i] += sv


    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# #####################
# #   Hook Functions  #
# #####################

# def hook_act_gather(output, layer):
#     # if layer not in activation_memorry:
#     #     activation_memorry[layer] = {"hooked_output": []}

#     print("Hook Layer:", layer)
#     print("output:", output)

#     # activation_memorry[layer]["hooked_output"].extend(output)
#     # print("(hook_act_gather) Hook activation memory:", activation_memorry)

# def hook_act_steering(output, key_vector=None):
#     for i in range(len(output[0][0])):
#         output[0][0][i] = output[0][0][i] + key_vector








if __name__ == "__main__":
    model_arguments = {
        "do_sample": True,
        "temperature": 0.7,
        "max_new_tokens": 100, #35,
        "top_p": 0.9,
        "top_k": 50,
        # "pad_token_id": pipeline.tokenizer.eos_token_id,
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "torch_dtype": torch.bfloat16,
        "hf_token": os.environ.get("HF_TOKEN", ""),
    }

    llm_wrapper = LLMWrapper(**model_arguments)