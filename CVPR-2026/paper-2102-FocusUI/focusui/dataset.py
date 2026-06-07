import copy
import json
import math
import os
import random
import re
import ast
from typing import Dict, Optional

import torch
import transformers
import yaml
from PIL import Image

from qwen_vl_utils import extract_vision_info, fetch_image, fetch_video
from torch.utils.data import Dataset

from focusui.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    ACTION_PATTENS_XY,
    ADDITIONAL_SPECIAL_TOKENS,
    ADDITIONAL_SPECIAL_TOKENS_IMAGE_DROP,
    assistant_template,
    chat_template,
    chat_template_qwen3vl,
    grounding_system_message_guiactor_qwen25vl,
    grounding_system_message_guiactor_qwen3vl,
)
from focusui.trainer import rank0_print

from focusui.preprocess_focusui import preprocess_focusui_data


def reformat_coordinates(text):
    """
    (1) Find all the coordinates in the text.
    (2) Replace the coordinates with the special tokens.
    (3) Return the new text and the coordinates as a list of (x, y), where x in [0, 1] and y in [0, 1].
    """
    epsilon = 0.001
    def adjust_coord(c):
        """
        Adjust coordinate if it is too close to 0 or 1.
        """
        if abs(c) < epsilon:
            return epsilon
        elif abs(c - 1) < epsilon:
            return 1 - epsilon
        return c

    all_matches = []
    for pattern in ACTION_PATTENS_XY:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            all_matches.append((match.start(), match.groups()))
        if pattern == ACTION_PATTENS_XY[0]:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        elif pattern == ACTION_PATTENS_XY[1]:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}, {DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        elif pattern == ACTION_PATTENS_XY[2]:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        text = re.sub(
            pattern,
            target_text,
            text
        )
    
    coordinates = []
    all_matches.sort(key=lambda x: x[0])
    # Extract coordinates in order
    for _, groups in all_matches:
        # When two coordinate values are found, parse them as one (x, y) pair.
        if len(groups) == 2:
            x_str, y_str = groups
            x = adjust_coord(ast.literal_eval(x_str))
            y = adjust_coord(ast.literal_eval(y_str))
            coordinates.append((x, y))
        # When four coordinate values are found, parse them as two pairs.
        elif len(groups) == 4:
            x1_str, y1_str, x2_str, y2_str = groups
            x1 = adjust_coord(ast.literal_eval(x1_str))
            y1 = adjust_coord(ast.literal_eval(y1_str))
            x2 = adjust_coord(ast.literal_eval(x2_str))
            y2 = adjust_coord(ast.literal_eval(y2_str))
            coordinates.append((x1, y1))
            coordinates.append((x2, y2))
    
    return text, coordinates

def get_token_index(image_processor, image, point_x, point_y):
    """
    Get the index of the visual token that contains the point (x, y).
    Args:
        image_processor: the image processor
        image: the image in PIL format
        point_x: the x coordinate of the point, in [0, 1].
        point_y: the y coordinate of the point, in [0, 1].
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")
    
    # get the original image size and the resized image size
    image = image[0]
    w, h = image.size
    px, py = w * point_x, h * point_y
    # get the token index
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = math.floor(px / merge_patch_size)
    y_index = math.floor(py / merge_patch_size)
    
    visual_token_index = y_index * (w // merge_patch_size) + x_index

    # merge all above print into one line
    return visual_token_index

def get_multi_patch_labels(image_processor, image, bbox_gt):
    """
    Get the multi-patch labels for the bounding box.
    Args:
        image_processor: the image processor
        image: the image in PIL format
        bbox_gt: the bounding box in the format of (x_min, y_min, x_max, y_max) [0,1]
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")

    # Get the original image size and the resized image size
    image = image[0]
    w, h = image.size

    bbox_gt = [bbox_gt[0]*w, bbox_gt[1]*h, bbox_gt[2]*w, bbox_gt[3]*h]
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox_gt
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    assert w % merge_patch_size == 0 and h % merge_patch_size == 0, f"Image size {w}x{h} is not divisible by merge_patch_size {merge_patch_size} (={image_processor.patch_size}x{image_processor.merge_size})"
    grid_h, grid_w = h // merge_patch_size, w // merge_patch_size

    binary_mask = torch.zeros(grid_h * grid_w)
    # Iterate through all patches, check if they overlap with the bounding box
    for y_idx in range(grid_h):
        for x_idx in range(grid_w):
            # Calculate patch boundaries
            patch_x_min = x_idx * merge_patch_size
            patch_y_min = y_idx * merge_patch_size
            patch_x_max = patch_x_min + merge_patch_size
            patch_y_max = patch_y_min + merge_patch_size
            
            # Check if patch overlaps with the bounding box
            if not (patch_x_max <= x_min or patch_x_min >= x_max or 
                    patch_y_max <= y_min or patch_y_min >= y_max):
                # Calculate patch index in the flattened grid
                patch_idx = y_idx * grid_w + x_idx
                binary_mask[patch_idx] = 1

    return binary_mask

def token_index_to_coordinates(image_processor, visual_token_index, image_width, image_height):
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = visual_token_index % (image_width // merge_patch_size)
    y_index = visual_token_index // (image_width // merge_patch_size)
    px = x_index * merge_patch_size + merge_patch_size / 2
    py = y_index * merge_patch_size + merge_patch_size / 2
    return px, py


def process_vision_info_w_factor(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    image_factor: int = 28,  # 28 for Qwen2(.5)VL, 32 for Qwen3VL
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    """
    Process the vision info for both Qwen2VL and Qwen3VL (fit patch_size = 28 or 32).
    """

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_factor))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        data_path: str,
        data_args,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.list_data_dict = []
        self.list_image_path = []
        self.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
        self.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        self.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]

        if data_args.grounding_system_message == "qwen25vl":
            self.system_message = grounding_system_message_guiactor_qwen25vl
            rank0_print(f"Using Qwen2.5VL system message: {self.system_message}")
        elif data_args.grounding_system_message == "qwen3vl":
            self.system_message = grounding_system_message_guiactor_qwen3vl
            rank0_print(f"Using Qwen3VL system message: {self.system_message}")
        else:
            raise ValueError(f"Invalid grounding system message: {data_args.grounding_system_message}")


        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path) as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path) as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    images_folder = dataset.get("images_folder")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy.")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path) as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        # NOTE: we only use json_path with .json now
                        # Handle the images_folder in yaml
                        with open(json_path) as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(float(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "repeat" and sampling_number is not None:
                        cur_data_dict = cur_data_dict * sampling_number

                    rank0_print(f"  - Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
                    self.list_image_path.extend([images_folder] * len(cur_data_dict))
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path) as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)
                self.list_image_path.extend([""] * len(cur_data_dict))  # NOTE: the image subfolder is empty...

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        # Optional: shuffle dataset order while keeping data and image paths aligned
        if self.data_args.shuffle:
            seed = 42
            rng = random.Random(seed)
            indices = list(range(len(self.list_data_dict)))
            rng.shuffle(indices)
            self.list_data_dict = [self.list_data_dict[idx] for idx in indices]
            if len(self.list_image_path) == len(indices):
                self.list_image_path = [self.list_image_path[idx] for idx in indices]
            rank0_print(f"Shuffled dataset with seed={seed}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = (
                1200 * len(sample["image"]) if isinstance(sample["image"], list) else 1200 if "image" in sample else 0
            )
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"

            img_tokens = (
                1200 * len(sample["image"]) if isinstance(sample["image"], list) else 1200 if "image" in sample else 0
            )

            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len + img_tokens)
            else:
                length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # sample = self._get_item(i)
        # if sample is None:
        #     new_index = random.randint(0, len(self.list_data_dict) - 1)
        #     return self.__getitem__(new_index)
        # else:
        #     return sample
        try:
            sample = self._get_item(i)
            if sample is None:
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                return self.__getitem__(new_index)
        except Exception as e:
            print(f"Failed to fetch sample {i}. Exception:", e)
            new_index = random.randint(0, len(self.list_data_dict) - 1)
            return self.__getitem__(new_index)
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image_path = os.path.join(self.data_args.image_folder, self.list_image_path[i])

        if "image" in sources:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image_list = [os.path.join(image_path, image_file) for image_file in image_file]
            else:
                image_list = [os.path.join(image_path, image_file)]
            self.list_data_dict[i]["image_abs_path"] = image_list[0]

            sources = copy.deepcopy(sources["conversations"])
        elif "video" in sources:
            raise NotImplementedError("Video is not supported for Qwen2VL")
        else:
            sources = copy.deepcopy(sources["conversations"])

        item_id = self.list_data_dict[i].get("id", i)

        data_dict = self.preprocess_qwenvl(
            source=sources,
            tokenizer=self.tokenizer,
            processor=self.processor,
            image=image_list,
            system_message=self.system_message,
            chat_template=self.tokenizer.chat_template,  # chat_template
            id=item_id
        )

        element_image = Image.open(image_list[0])
        element_bbox_gt = data_dict["bbox_gt"]
        # element_bbox_gt = data_dict["bbox_gt"] if data_dict["bbox_gt"] is not None else [0, 0, 0, 0]
        element_query_text = sources[0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()

        focusui_data_dict = preprocess_focusui_data(
            ele_image=element_image,
            ele_bbox=element_bbox_gt,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
            # gt_bbox_weight=1.0,
            # gt_uigraph_weight=1.0,
            tokenizer=self.tokenizer,  # self.tokenizer,
            instruction=element_query_text,
            patch_size=self.processor.image_processor.patch_size,
            merge_size=self.processor.image_processor.merge_size,
            # None,  # self.tokenizer,
            # None,  # query_text
        )
        # focus_ui_data_dict = {
        #     "focus_ids": focus_ids,
        #     "focus_attention_mask": focus_attention_mask,
        #     "patch_score_bbox": torch.from_numpy(patch_score_bbox),
        #     "patch_score_uigraph": torch.from_numpy(patch_score_uigraph),
        #     "patch_scores_label_unmerged": torch.from_numpy(patch_scores_label),
        #     "patch_scores_label": torch.from_numpy(patch_score_merged_gt),
        # }

        # some instruction query may be empty, skip these samples
        if focusui_data_dict["focus_input_ids"] is None:
            return None

        data_dict = {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "coordinates": data_dict["coordinates"][0],
            "visual_token_indices_of_coordinates": data_dict["visual_token_indices_of_coordinates"][0],
            "bbox_gt": data_dict["bbox_gt"][0],
            "pixel_values": data_dict["pixel_values"],
            "image_grid_thw": data_dict["image_grid_thw"],
            "multi_patch_labels": data_dict["multi_patch_labels"][0],   # add multi_patch_labels   
            "patch_scores_label": focusui_data_dict["patch_scores_label"],
            "focus_input_ids": focusui_data_dict["focus_input_ids"],
            "focus_attention_mask": focusui_data_dict["focus_attention_mask"],
            "focusui_data_dict": focusui_data_dict,
            "metadata": self.list_data_dict[i],
        }

        data_dict["id"] = item_id

        # return None if the input_ids is longer than the model_max_length
        n_image_tokens = (
            data_dict["image_grid_thw"][0][0] * 
            data_dict["image_grid_thw"][0][1] * 
            data_dict["image_grid_thw"][0][2] / 
            self.processor.image_processor.merge_size / 
            self.processor.image_processor.merge_size
        )
        if (len(data_dict["input_ids"]) + n_image_tokens) > self.tokenizer.model_max_length:
            rank0_print(f"=== Removed data_dict {i} because it is longer than the model_max_length: {len(data_dict['input_ids'])} + {n_image_tokens} > {self.tokenizer.model_max_length}")
            return None
        
        return data_dict

    def preprocess_qwenvl(
        self,
        source, # conversations
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        image: list,
        system_message: str,
        agent_mode: bool = True,
        chat_template: str = chat_template,
        assistant_template: str = assistant_template,
        id: int = None,
    ) -> Dict:
        roles = {"human": "user", "gpt": "assistant", "system": "system"}
        assistant_template = assistant_template if agent_mode else chat_template
        processor.tokenizer = tokenizer
        assert tokenizer.additional_special_tokens == ADDITIONAL_SPECIAL_TOKENS+ADDITIONAL_SPECIAL_TOKENS_IMAGE_DROP

        # Apply prompt templates
        pixel_values, image_grid_thw = None, None

        input_id, target = [], []
        coordinates = []
        visual_token_indices_of_coordinates = []
        multi_patch_labels = []
        bbox_gt = None
        
        image_list = []
        image_index = 0

        ## prepare the system message
        if roles[source[0]["from"]] == "system":
            system_message = source[0]["value"]
            source = source[1:self.data_args.max_conv_turns]
        # else: use the constant system message
        system_input_id = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": [{"type": "text", "text": system_message}]}],
            chat_template=chat_template,
        )
        input_id += system_input_id
        target += [IGNORE_INDEX] * len(system_input_id)

        ## prepare user-assistant conversation
        for conv in source:
            # regularize the conversation format
            role = conv.get("role", conv.get("from", None))
            content = conv.get("content", conv.get("value", None))
            role = roles.get(role, role)

            # Count the number of <image> tokens in the content
            image_count = content.count(DEFAULT_IMAGE_TOKEN)
            if image_count > 0:
                assert role == "user", "Images are only supported for user messages"
                # include image information regarding to current conversation turn
                image_placeholders = []
                for _ in range(image_count):
                    image_placeholders.append({
                        "type": "image",
                        "image": image[image_index],
                        "min_pixels": self.processor.image_processor.min_pixels,
                        "max_pixels": self.processor.image_processor.max_pixels,
                    })
                    image_index += 1

                content = content.replace(DEFAULT_IMAGE_TOKEN, "")
                conv = {"role": role, "content": image_placeholders + [{"type": "text", "text": content}]}

                merge_patch_size = processor.image_processor.merge_size * processor.image_processor.patch_size
                
                image_inputs, _ = process_vision_info_w_factor(
                    conversations=[conv],
                    image_factor=merge_patch_size,
                ) # list of PIL.Image.Image

                image_list.extend(image_inputs)
                
                templated_conv = tokenizer.apply_chat_template(
                    conversation=[conv], chat_template=chat_template, tokenize=False
                )
                inputs = processor(text=[templated_conv], images=image_inputs, return_tensors="pt")

                if pixel_values is None and image_grid_thw is None:
                    pixel_values = inputs["pixel_values"]
                    image_grid_thw = inputs["image_grid_thw"]
                else:
                    pixel_values = torch.concat([pixel_values, inputs["pixel_values"]], dim=0)
                    image_grid_thw = torch.concat([image_grid_thw, inputs["image_grid_thw"]], dim=0)
            
            else:
                if role in ["user", "system"]:
                    conv = {
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    }
                else:  # assistant
                    conv = {
                        "role": role,
                        "content": [{"type": "text", "text": content}],
                        "recipient": conv.get("recipient", "os"),
                        "end_turn": conv.get("end_turn", True),
                        "bbox_gt": conv.get("bbox_gt", None),
                    }
                    if conv["recipient"] == "os":
                        if len(image_inputs) == 0:
                            raise ValueError("No image found for visual grounding")
                        # replace the coordinates with the special tokens
                        text, coord = reformat_coordinates(conv["content"][0]["text"])
                        conv["content"][0]["text"] = text

                        # get the visual token indices of the coordinates
                        coordinates.extend(coord)
                        for (point_x, point_y) in coord:
                            visual_token_index = get_token_index(
                                processor.image_processor,
                                image_list,
                                point_x,
                                point_y
                            )
                            visual_token_indices_of_coordinates.append(visual_token_index)

                            if conv["bbox_gt"] is not None:
                                patch_mask = get_multi_patch_labels(
                                    processor.image_processor,
                                    image_list,
                                    conv["bbox_gt"]
                                )  
                                multi_patch_labels.append(patch_mask)
                                # update the bbox_gt
                                if bbox_gt is None:
                                    bbox_gt = conv["bbox_gt"]

                templated_conv = tokenizer.apply_chat_template(
                    conversation=[conv],
                    chat_template=assistant_template,
                    tokenize=False,
                )
                inputs = processor(text=[templated_conv], return_tensors="pt")

            encode_id = inputs.input_ids[0].tolist()

            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        # make the labels of all pointer_end_token_id to be IGNORE_INDEX
        target = [IGNORE_INDEX if token == self.pointer_end_token_id else token for token in target]

        input_ids = torch.tensor([input_id], dtype=torch.long)
        targets = torch.tensor([target], dtype=torch.long)
        visual_token_indices_of_coordinates = torch.tensor([visual_token_indices_of_coordinates], dtype=torch.long) if len(visual_token_indices_of_coordinates) > 0 else [None]
        coordinates = [coordinates] if len(coordinates) > 0 else [None]

        # process multi_patch_labels
        if len(multi_patch_labels) > 0:
            multi_patch_labels = [torch.stack(multi_patch_labels)]
        else:
            multi_patch_labels = [None]

        data_dict = {
            "input_ids": input_ids,  # tensor(bs x seq_len)
            "labels": targets,  # tensor(bs x seq_len)
        }

        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
        
        data_dict["bbox_gt"] = bbox_gt
        data_dict["coordinates"] = coordinates
        data_dict["visual_token_indices_of_coordinates"] = visual_token_indices_of_coordinates
        data_dict["multi_patch_labels"] = multi_patch_labels
        return data_dict
