import json
import random
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os

from openai import OpenAI
import os
import base64
import pdb
from PIL import Image
import clip
import numpy as np
import dashscope
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)
    
    font = ImageFont.load_default()  # Uses PIL's basic font

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]
      
      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.show()
    return img


def plot_points(im, text, input_width, input_height):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  xml_text = text.replace('```xml', '')
  xml_text = xml_text.replace('```', '')
  data = decode_xml_points(xml_text)
  if data is None:
    img.show()
    return
  points = data['points']
  description = data['phrase']

  font = ImageFont.load_default()  # Uses PIL's basic font

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/input_width * width
    abs_y1 = int(point[1])/input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)
  
  img.show()
  

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output



#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def extract_bounding_box_regions(im, bounding_boxes, input_width, input_height):
    """
    Extracts regions inside bounding boxes from an image.

    Args:
        im (PIL.Image): The input image.
        bounding_boxes (list): A list of bounding boxes in normalized [y1, x1, y2, x2] format.
        input_width (int): Original width of the image before resizing.
        input_height (int): Original height of the image before resizing.

    Returns:
        list: A list of cropped image regions.
    """
    width, height = im.size
    cropped_images = []
    
    # Ensure bounding_boxes are properly parsed
    bounding_boxes = parse_json(bounding_boxes)

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    for i, bounding_box in enumerate(json_output):
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / input_width * width)

        # Ensure coordinates are correctly ordered
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Crop the bounding box area
        cropped_region = im.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        cropped_images.append(cropped_region)

    return cropped_images


class ReasonModelAPI:
    def __init__(self, system_prompt=None, model_id="qwen2.5-vl-32b-instruct"):
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. I will provide an implicit description of the task. "
            "Your task is to identify and outline the most appropriate region where the object to be replaced is located, "
            "and generate a label for the existing object. Then, generate a clear and explicit description of the new object that will replace it. "
            'Output must strictly follow JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "...", "new_object_description": "...", "reasoning_process": "..."}]'
        )
        self.model_id = model_id
        
        self.client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            # base_url="llama-4-maverick-17b-128e-instruct",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            
        )

    def reason(self, image, prompt, min_pixels=512*28*28, max_pixels=2048*28*28):
        base64_image = encode_image(image)

        # image = base64.b64encode(image.read()).decode("utf-8")
        

        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": self.sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                {"image":  f"data:image/jpeg;base64,{base64_image}"},
                {"text": prompt}]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            api_key = '<Your API Key>',
            # model = 'llama-4-scout-17b-16e-instruct',
            model = 'qwen-vl-plus',
            messages = messages,
            vl_high_resolution_images=True
        )

        print(response.output.choices[0].message.content[0]["text"])
        return response.output.choices[0].message.content[0]["text"]

    def reason_baseline(self, image_paths, prompt, min_pixels=512*28*28, max_pixels=2048*28*28):
        """
        Args:
            image_paths (List[str]): list of image file paths
            prompt (str): natural language prompt
        Returns:
            parsed JSON object: [{'selected_index': ..., 'bbox_2d': [...], 'label': '...', 'explanation': '...'}]
        """
        import base64
        import json

        def encode_image(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # 编码所有图片
        image_contents = [{"image": f"data:image/jpeg;base64,{encode_image(p)}"} for p in image_paths]

        # 构造消息内容
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.sys_prompt}]
            },
            {
                "role": "user",
                "content": image_contents + [{"text": prompt}]
            }
        ]

        # 调用 API
        response = dashscope.MultiModalConversation.call(
            api_key='<Your Key>',
            model='qwen-vl-plus',
            messages=messages,
            vl_high_resolution_images=True
        )

        # 获取回复内容（JSON格式字符串）
        response_text = response.output.choices[0].message.content[0]["text"]
        print("[Raw Response Text]:", response_text)
        return response.output.choices[0].message.content[0]["text"]

    def reason_text(self, prompt, min_pixels=512*28*28, max_pixels=2048*28*28):
        # base64_image = encode_image(image)

        # image = base64.b64encode(image.read()).decode("utf-8")
        

        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": self.sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                # {"image":  f"data:image/jpeg;base64,{base64_image}"},
                {"text": prompt}]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            #若没有配置环境变量， 请用百炼API Key将下行替换为： api_key ="sk-xxx"
            api_key = 'sk-e56763764b9c493f948f45d2862d49ef',
            model = 'qwen-vl-plus',
            messages = messages,
            vl_high_resolution_images=True
        )

        print(response.output.choices[0].message.content[0]["text"])
        return response.output.choices[0].message.content[0]["text"]

class ReasonModel:
    def __init__(self, system_prompt = 'I am conducting an image editing task. You are a helpful assistant. I will give you a implicit decription. You should outline the position of the most relevant target object that should be edited and output the coordinates, object to be edited, and the specific object label after edit. The object label after edit should be a specific name of certain object, e.g. a bottle of water. The output should be in JSON format: [{"bbox_2d": [], "label": "", "target": ""}].'):
        self.system_prompt = system_prompt
        self.setup_model()

    def setup_model(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        model_path = "/root/autodl-fs/Qwen2.5-VL-3B-Instruct"
        if not os.path.exists(model_path):
            raise ValueError(f"本地模型路径不存在: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,).cuda()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def reason(self, image, prompt):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "image": None
                    }
                   ]
            }
        ]
        image = tensor_to_pil(image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("input:\n",text)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')
        output_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("output:\n",output_text[0])

        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14

        return output_text[0], input_height, input_width


    # def reason(self, image, prompt):
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": self.system_prompt
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                 "type": "text",
    #                 "text": prompt
    #                 },
    #                 {
    #                 "image": None
    #                 }
    #                ]
    #         }
    #     ]
    #     image = tensor_to_pil(image)
    #     text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     print("input:\n",text)
    #     inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')
    #     output_ids = self.model.generate(**inputs, max_new_tokens=1024)
    #     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    #     output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     print("output:\n",output_text[0])

    #     input_height = inputs['image_grid_thw'][0][1]*14
    #     input_width = inputs['image_grid_thw'][0][2]*14

    #     return output_text[0], input_height, input_width


# def tensor_to_pil(image_tensor):
#     # 1. 确保张量是浮点类型且形状为CxHxW
#     if image_tensor.dtype != torch.float32:
#         image_tensor = image_tensor.float()
    
#     # 2. 分离计算图并转到CPU
#     image_np = image_tensor.cpu().detach()
    
#     # 3. 归一化处理（自动适应不同数值范围）
#     min_val = image_np.min()
#     max_val = image_np.max()
#     image_np = (image_np - min_val) / (max_val - min_val)  # 归一化到0-1
    
#     # 4. 转换到numpy并调整通道顺序
#     image_np = image_np.numpy()
#     image_np = np.transpose(image_np, (1, 2, 0))  # CHW→HWC
    
#     # 5. 转换为0-255的uint8
#     image_np = (image_np * 255).astype(np.uint8)
    
#     # 6. 创建PIL图像
#     return Image.fromarray(image_np)
def tensor_to_pil(image_tensor):
    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()
    
    image_np = image_tensor.cpu().detach().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # CHW→HWC

    # 假设图像在 [0, 1]，直接乘 255
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(image_np)