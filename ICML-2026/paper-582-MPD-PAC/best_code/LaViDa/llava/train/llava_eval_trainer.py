import collections
import json
import math
import subprocess
import copy
from dataclasses import dataclass

from typing import Any, Dict, Literal, Union, Optional, List, Tuple, Callable, Sequence


from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import (
    # EvalPrediction, 
    # EvalLoopOutput,
    has_length,
)
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
import torch.nn as nn
import torch.distributed as dist
import torch
import PIL
from tqdm import tqdm

from llava.train.llava_trainer import LLaVATrainer
# from llava.train.config import TrainingArguments
from llava.utils import rank0_print
from llava.mm_utils import (
    process_images,
    tokenizer_image_token
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.conversation import conv_templates
from trl.models.utils import unwrap_model_for_generation

from loguru import logger as eval_logger
import os
EVAL_CONV_TEMPLATE = os.environ.get('EVAL_CONV_TEMPLATE','llada')
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)
try:
    from lmms_eval.tasks import (
        TaskManager,
        get_task_dict
    )
    task_manager = TaskManager()
except:
    rank0_print("Please install lmms_eval to use evaluation")
    raise ModuleNotFoundError

class LMMsEvalDataset(Dataset):
    def __init__(
        self, 
        hf_dataset, 
        task_obj, 
        model_config,
        image_processor,
        conv_template,
        task_type: Literal["loglikelihood", "generate_until"],
        tokenizer,
        limit=-1,
        ) -> None:
        super().__init__()
        self.hf_dataset = hf_dataset
        self.task_obj = task_obj
        self.model_config = model_config
        self.image_processor = image_processor
        self.conv_template = conv_template
        self.task_type = task_type
        self.generation_kwargs = task_obj.config.generation_kwargs
        self.tokenizer = tokenizer
        self.limit = limit
    
    def __getitem__(self, index):
        visual = self.task_obj.doc_to_visual(self.hf_dataset[index])
        context = self.task_obj.doc_to_text(self.hf_dataset[index])
        if visual is None or visual == []:
            visual = None
            task_type = "text"
            image_tensor = None
        else:
            if type(visual[0]) == PIL.Image.Image:
                image_tensor = process_images(visual, self.image_processor, self.model_config)
                if type(image_tensor) is list:
                    image_tensor = [_image for _image in image_tensor]
                else:
                    image_tensor = image_tensor

                task_type = "image"
        
        if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
            placeholder_count = len(visual) if isinstance(visual, list) else 1
            image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
            image_tokens = " ".join(image_tokens)
            prompts_input = image_tokens + "\n" + context
        else:
            prompts_input = context
        
        if "llama_3" in self.conv_template or 'llada' in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        conv.append_message(conv.roles[0], prompts_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        if type(self.task_obj.doc_to_target) == str:
            target = self.task_obj.doc_to_target
        else:
            target =self.task_obj.doc_to_target(self.hf_dataset[index])
        
        image_sizes = [visual[idx].size for idx in range(len(visual))]
        
        if self.task_type == "generate_until":
            return {
                "input_ids" : input_ids,
                "modalities" : ["image"] if task_type == "image" else ["text"],
                "images" : image_tensor,
                "image_sizes" : image_sizes,
                "index" : index,
                "prompt":prompt,
            }
        elif self.task_type == "loglikelihood":
            # Because caption tasts such as coco return a
            # list of answer, we pick the first one
            if isinstance(target, list):
                target = target[0]
            conv.messages[-1][1] = target
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            labels = full_input_ids.clone()
            labels[ : input_ids.shape[0]] = -100
            return {
                "input_ids" : full_input_ids,
                "modalities" : ["image"] if task_type == "image" else ["text"],
                "images" : image_tensor,
                "image_sizes" : image_sizes,
                "index" : index,
                "labels" : labels,
            }
        else:
            raise ValueError(f"Task type : {self.task_type} is not Supported, please choose between generate_until or loglikelihood")

    
    def __len__(self) -> int:
        if self.limit > 0:
            return self.limit
        return len(self.hf_dataset)
    
@dataclass
class DataCollatorForEvaluationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizerBase

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, modalities, images, image_sizes, index,prompt = tuple([instance[key] for instance in instances] for key in ("input_ids", "modalities", "images", "image_sizes", "index","prompt"))
            
        labels = []
        for instance in instances:
            if "labels" in instance:
                labels.append(instance["labels"])
        if len(labels) == 0:
            labels = None
        
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]

        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if labels is not None:
            labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
            labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids, 
            labels=labels.long() if labels is not None and labels.dtype == torch.int32 else labels, 
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
            modalities=modalities,
            image_sizes=image_sizes,
            index=index,
            prompt=prompt,
        )

def mean(arr):
    return sum(arr) / len(arr)

class LLaVAEvalTrainer(LLaVATrainer):
    def __init__(
        self, 
        *args,
        
        **kwargs
        # model: Union[PreTrainedModel, nn.Module] = None, 
        # args: TrainingArguments = None, 
        # data_collator: Optional[DataCollator] = None, 
        # train_dataset: Optional[Union[Dataset, IterableDataset]] = None, 
        # eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None, 
        # tokenizer: Optional[PreTrainedTokenizerBase] = None, 
        # model_init: Optional[Callable[[], PreTrainedModel]] = None, 
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None, 
        # callbacks: Optional[List[TrainerCallback]] = None, 
        # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None), 
        # preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # conv_template: str = "qwen_1_5",
        ):
        super().__init__(*args,**kwargs)
        generate_tasks = self.args.lmms_eval_generate_tasks.split(",")
        self.generate_task_dict = get_task_dict(generate_tasks, task_manager)
        # ppl_tasks = self.args.lmms_eval_ppl_tasks.split(",")
        # self.ppl_task_dict = get_task_dict(ppl_tasks, task_manager)
        self.ppl_task_dict = {}
        self.model_config = self.model.config
        self.image_processor = self.model.get_vision_tower().image_processor
        self.conv_template = EVAL_CONV_TEMPLATE
        self.eval_data_collator = DataCollatorForEvaluationDataset(self.tokenizer)

    def evaluate(
        self, 
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "eval"
        ) -> Dict[str, float]:

        
        log_dict = {}
        for task_name, task_obj in self.generate_task_dict.items():
            eval_dataset = LMMsEvalDataset(
                task_obj.test_docs(),
                task_obj,
                self.model_config,
                self.image_processor,
                self.conv_template,
                "generate_until",
                self.tokenizer,
            )

            eval_dataloader = self.get_lmms_eval_dataloader(
                eval_dataset,
                self.eval_data_collator
            )

            resps, correspond_index = self.generate_until_loop(
                eval_dataloader,
                description=task_obj.task_name,
            )
            # breakpoint()
            processed_results = self.process_results(
                resps,
                correspond_index,
                task_obj
            )

            # Because the resps are scattered in different ranks
            # We gather all the processed results and then merged
            all_processed_results = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_processed_results, processed_results)

            
            merged_processed_results = collections.defaultdict(list)
            for processed_result in all_processed_results:
                for metric_name, data_dict in processed_result.items():
                    merged_processed_results[metric_name].extend(data_dict)


            if self.accelerator.is_main_process:
                for metric_name, processed_result in merged_processed_results.items():
                    aggregation_list = task_obj.aggregation()
                    # Okay, to be honest, other tasks might also suffer from this, 
                    # but mme strictly follows pair evaluation so I kind of hard code this handle logic in this way. 
                    # data loader might contain duplicate tasks when preparing. 
                    # I am just keep it this way for now, 
                    # since it is just an inofficial evaluation during middle training. 
                    # At last, recommend you to use lmms_eval for a wholistic evaluation after the training ! :D
                    if task_name == "mme":
                        processed_result = self.handle_mme_duplicate_result(processed_result)
                    score = self.aggregation(aggregation_list, metric_name, processed_result)
                    log_dict[f"{task_name}/{metric_name}"] = score
            self.accelerator.wait_for_everyone()

        for task_name, task_obj in self.ppl_task_dict.items():
            eval_dataset = LMMsEvalDataset(
                task_obj.test_docs(),
                task_obj,
                self.model_config,
                self.image_processor,
                self.conv_template,
                "loglikelihood",
                self.tokenizer,
            )

            eval_dataloader = self.get_lmms_eval_dataloader(
                eval_dataset,
                self.eval_data_collator
            )

            losses = self.loglikelihood_loop(
                eval_dataloader,
                description=task_obj.task_name
            )

            all_losses = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_losses, losses)
            merged_losses = []
            for losses in all_losses:
                merged_losses.extend(losses)

            if self.accelerator.is_main_process:
                ppl = math.exp(-mean(merged_losses))
                log_dict[f"{task_name}/ppl"] = ppl
            
            self.accelerator.wait_for_everyone()


        self.log(log_dict)
        torch.cuda.empty_cache()

        return log_dict

    def get_lmms_eval_dataloader(
        self, 
        eval_dataset: Optional[Union[str, Dataset]] = None,
        data_collator = None,
        ) -> DataLoader:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        return self.accelerator.prepare_data_loader(eval_dataloader)
    
    def generate_until_loop(
        self,
        dataloader: DataLoader,
        description: str,
    ):
        # self.model.eval()
        model = self.unwrap_model_for_inference(dataloader)
        args = self.args
        batch_size = self.args.eval_batch_size
        num_examples = self.num_examples(dataloader)
        rank0_print(f"\n***** Running {description} *****")
        rank0_print(f"  Num examples = {num_examples}")
        rank0_print(f"  Batch size = {batch_size}")

        world_size = max(1, args.world_size)
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        resps = []
        correspond_index = []

        pbar = tqdm(total=len(dataloader), desc=description, disable=not self.accelerator.is_local_main_process)
        gen_kwargs = {}
        gen_kwargs["max_new_tokens"] = 16
        gen_kwargs['block_length'] = min(128,gen_kwargs["max_new_tokens"])
        gen_kwargs['prefix_lm']=True
        gen_kwargs['step_per_block'] = gen_kwargs['block_length']
        
        if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
          
        with torch.inference_mode():
         with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:      
          for step, inputs in enumerate(dataloader):
            # Because batch size is 1, so we unwrap the list from inside
            modalities = inputs.pop("modalities")[0]
            image_sizes = inputs.pop("image_sizes")[0]
            inputs["images"] = inputs["images"][0]
            inputs["images"] = inputs["images"].to(model.dtype)
            index = inputs.pop("index")
            prompt = inputs.pop("prompt")
                # model.generate(input_ids=inputs["input_ids"],images=inputs["images"],attention_mask=inputs["attention_mask"],modalities=modalities, image_sizes=image_sizes, pad_token_id=pad_token_ids)
                
                    # with open('/data1/jacklishufan/trainer.pt', 'wb') as f:
                    #     torch.save(unwrapped_model.state_dict(), f)
                    #     print('saved')
                    #     print(1/0)  
            cont = unwrapped_model.generate(
                inputs=inputs["input_ids"],
                images=inputs["images"],
                #attention_mask=inputs["attention_mask"],
                attention_mask=None,
                modalities=modalities, 
                image_sizes=image_sizes, 
                pad_token_id=pad_token_ids,
                use_cache=True,
                # temperature=0.0,
                # do_sample=False,
                **gen_kwargs
            )
            if hasattr(cont,'sequences'):
                cont = cont.sequences
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            text_outputs = [response.strip().lstrip('!') for response in text_outputs]
            resps.extend(text_outputs)
            # breakpoint()
            if DEBUG_PRINT_OUTPUT:
                print(f'\n--------Start of Sample {index}---------')
                print("Question: ",prompt[0])
                print("Answer: ",text_outputs)
                log_kwargs = dict(
                    **gen_kwargs,
                    image_sizes=image_sizes,
                )
                print("Answer: ",log_kwargs)
                print('--------End---------')

            correspond_index.extend(index)
            pbar.update(1)
        pbar.close()

        
        return resps, correspond_index

    def loglikelihood_loop(
        self,
        dataloader: DataLoader,
        description: str,
    ) -> List[float]:
        model = self.unwrap_model_for_inference(dataloader)
        args = self.args
        batch_size = self.args.eval_batch_size
        num_examples = self.num_examples(dataloader)
        rank0_print(f"\n***** Running {description} *****")
        rank0_print(f"  Num examples = {num_examples}")
        rank0_print(f"  Batch size = {batch_size}")

        world_size = max(1, args.world_size)

        losses = []

        pbar = tqdm(total=len(dataloader), desc=description, disable=not self.accelerator.is_local_main_process)
        for step, inputs in enumerate(dataloader):
            # Because batch size is 1, so we unwrap the list from inside
            modalities = inputs.pop("modalities")[0]
            image_sizes = inputs.pop("image_sizes")[0]
            inputs["images"] = inputs["images"][0]
            inputs["images"] = inputs["images"].to(model.dtype)
            index = inputs.pop("index")
            with torch.no_grad():
                output = model(
                    input_ids=inputs["input_ids"],
                    images=inputs["images"],
                    attention_mask=inputs["attention_mask"],
                    modalities=modalities, 
                    image_sizes=image_sizes, 
                    labels=inputs["labels"],
                    )
                loss = output["loss"]
                losses.append(float(loss.item()))
            pbar.update(1)
        return losses

    def process_results(
        self,
        resps: List[str],
        correspond_index: List[int],
        task_obj,
        ) -> Dict[str, List[Dict[str, Any]]]:
        # We retrive our test docs first
        # Notice that here is no image, so probably you
        # can't evaluate llava_wilder etc. :D
        test_docs_no_image = task_obj.dataset_no_image[task_obj.config.test_split]
        processed_results = collections.defaultdict(list)
        pbar = tqdm(total=len(resps), desc="Processed eval results", disable= not self.accelerator.is_main_process)
        for resp, index in zip(resps, correspond_index):
            doc = test_docs_no_image[index]
            result = [resp]
            data_dict = task_obj.process_results(doc, result)
            for metric_name, data in data_dict.items():
                processed_results[metric_name].append(data)
            pbar.update(1)
        pbar.close()
        
        return processed_results

    def aggregation(
        self,
        aggregation_list: List[Dict[str, Callable]],
        metric_name: str,
        results: Dict[str, List[Dict[str, Any]]],
    ) -> float:
        if metric_name == 'submission':
            return -1
        aggregation_fn = aggregation_list[metric_name]
        score = aggregation_fn(results)
        return score
    
    def handle_mme_duplicate_result(
        self,
        data_dict: List[Dict[str, Any]]
    ):
        exist_question_id = collections.defaultdict(int)
        fixed_data_dict = []
        # Each question id may contains at most 2 images
        for res in data_dict:
            question_id = res["question_id"]
            if exist_question_id[question_id] >= 2:
                continue
            else:
                fixed_data_dict.append(res)
                exist_question_id[question_id] += 1
        
        return fixed_data_dict

    def unwrap_model_for_inference(
        self,
        dataloader : DataLoader,
    ):
        args = self.args

        if not has_length(dataloader):
            raise ValueError("dataloader must implement a working __len__")

        # if eval is called w/o train, handle model prep here
        hf_deepspeed_config = self.accelerator.state.deepspeed_plugin.hf_ds_config

        # resume config update - some bits like `model` and `num_training_steps` only become available during train
        if not hf_deepspeed_config.is_zero3():
            pass
        elif self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model.eval()
        return model