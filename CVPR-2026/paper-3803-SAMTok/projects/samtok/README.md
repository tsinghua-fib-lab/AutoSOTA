<p align="center">
  <img width="225" src="figs/logo.png">
</p>

<h2 align="center">SAMTok: Representing Any Mask with Two Words [CVPR-2026]</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2601.16093" target="_blank"><img src="https://img.shields.io/badge/arXiv-2601.16093-red"></a>
  <a href="https://zhouyiks.github.io/projects/SAMTok/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
  <a href="https://huggingface.co/collections/zhouyik/samtok" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-orange"></a>
  <a href="https://huggingface.co/spaces/insomnia7/SAMTok" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg"></a>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=dZikW2YAAAAJ&hl=en&oi=ao">Yikang Zhou</a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=3xu4a5oAAAAJ">Tao Zhang</a><sup>1</sup>, <a>Dengxian Gong</a><sup>1</sup>, <a>Yuanzheng Wu</a><sup>1</sup>, <a>Haochen Wang</a><sup>2</sup>, <a>Ye Tian</a><sup>2</sup>, <a>Haobo Yuan</a><sup>1</sup>, <a>Jiacong Wang</a><sup>2</sup>,  <a>Lu Qi</a><sup>1</sup>, <a>Hao Fei</a><sup>1</sup>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=FjoRmF4AAAAJ">Shunping Ji</a><sup>1,✉️</sup>, <a>Anran Wang</a><sup>2</sup>, <a>Zhuochen Wang</a><sup>2</sup>, <a>Yujing Wang</a><sup>2</sup>, <a>Cheng CHEN</a><sup>2</sup>, <a href="https://lxtgh.github.io/">Xiangtai Li</a><sup>2</sup>
  <p align="center"><sup>1</sup>Wuhan University <sup>2</sup>ByteDance</p>
</p>

<p align="center"><img width="750" src="figs/teaser.png"></p>

**SAMTok** provides a unified mask-token interface for MLLMs. (Left) SAMTok compresses region masks into two discrete tokens and faithfully reconstructs them across diverse visual domains. (Middle) Injecting these mask tokens into MLLMs enables a wide range of region-level mask generation and understanding tasks. (Right) The text-based representation of region masks allows a purely textual answer-matching reward for the GRPO of the mask generation task.

## Opensource progress

- ✅ Release model weights.
- ✅ Release mask tokenizer codes.
- ✅ Release SAMTok training instruction.
- ✅ Release evaluation codes.
- ✅ Release demo codes.
- ✅ Release gradio demo.
- ✅ Release RL codes & instruction.


## 📦 Model Zoo
| Model | Checkpoint | Note |
|:-:|:-:|:-:|
| Qwen3-VL-8B-SAMTok | [🤗 Link](https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok) | This model is trained on a mixture of [general VQA datasets](https://huggingface.co/datasets/Open-Bee/Honey-Data-1M), Mask Generation, and Mask Understanding tasks, and it is recommended for use. |
| Qwen3-VL-4B-SAMTok | [🤗 Link](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok) | This model is trained on a mixture of [general VQA datasets](https://huggingface.co/datasets/Open-Bee/Honey-Data-1M), Mask Generation, and Mask Understanding tasks, and it is recommended for use. |
| Qwen2.5-VL-7B-SAMTok-co | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-7B-SAMTok-co) | This model is trained using Mask Generation and Mask Understanding data. It corresponds to the method Qwen25VL-SAMTok (7B) in Table 1, Table 2, Table 3, Table 9, and Table 10 of the paper. |
| Qwen2.5-VL-3B-SAMTok-co | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-3B-SAMTok-co) | This model is trained using Mask Generation and Mask Understanding data. It corresponds to the method Qwen25VL-SAMTok (3B) in Table 1, Table 2, Table 3, Table 4, Table 5, Table 10, and Table 11 of the paper. |
| Qwen3-VL-4B-SAMTok-co | [🤗 Link](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok-co) | This model is trained using Mask Generation and Mask Understanding data. It corresponds to the method Qwen3VL-SAMTok (4B) in Table 1, Table 2, Table 3, Table 4, and Table 11 of the paper. |
| PLM-1B-SAMTok-co | [🤗 Link](http://huggingface.co/zhouyik/PLM-1B-SAMTok-co) | This model is trained using Mask Generation and Mask Understanding data. It corresponds to the method PLM-SAMTok (1B) in Table 11 of the paper. |
| Qwen3-VL-4B-SAMTok-dam | [🤗 Link](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok-dam) | This model is trained using Mask Understanding data. It corresponds to the method Qwen3VL-SAMTok (4B) in Table 6, Table 7, and Table 8 of the paper. |
| Qwen2.5-VL-3B-SAMTok-gcg-rl | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-3B-SAMTok-gcg-rl) | This model is post-trained on the GCG task using GRPO, based on Qwen2.5-VL-3B-SAMTok-co. It corresponds to the method Qwen25VL-SAMTok (rl) (3B) in Table 1 of the paper. |
| Qwen2.5-VL-3B-SAMTok-gres-rl | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-3B-SAMTok-gres-rl) | This model is post-trained on the GRES task using GRPO, based on Qwen2.5-VL-3B-SAMTok-co. It corresponds to the method Qwen25VL-SAMTok (rl) (3B) in Table 3 of the paper. |
| Qwen2.5-VL-3B-SAMTok-gcg-ft | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-3B-SAMTok-gcg-ft) | This model is further fine-tuned on the GCG task based on Qwen2.5-VL-3B-SAMTok-co. It corresponds to the method Qwen25VL-SAMTok (ft) (3B) in Table 1 of the paper. |
| Qwen2.5-VL-7B-SAMTok-gcg-ft | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-7B-SAMTok-gcg-ft) | This model is further fine-tuned on the GCG task based on Qwen2.5-VL-7B-SAMTok-co. It corresponds to the method Qwen25VL-SAMTok (ft) (7B) in Table 1 of the paper. |
| Qwen2.5-VL-3B-SAMTok-gres-ft | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-3B-SAMTok-gres-ft) | This model is further fine-tuned on the GRES task based on Qwen2.5-VL-3B-SAMTok-co. It corresponds to the method Qwen25VL-SAMTok (ft) (3B) in Table 3 of the paper. |
| Qwen2.5-VL-7B-SAMTok-gres-ft | [🤗 Link](https://huggingface.co/zhouyik/Qwen2.5-VL-7B-SAMTok-gres-ft) | This model is further fine-tuned on the GRES task based on Qwen2.5-VL-7B-SAMTok-co. It corresponds to the method Qwen25VL-SAMTok (ft) (7B) in Table 3 of the paper. |

## 🎯 Inference Demo
1. **Environment.**
SAMTok provides a unified mask interface for VLMs without requiring structural changes to the base model, so you only need an environment that supports the base model's inference to use SAMTok. For example, to provide a mask interface for Qwen3VL, you simply need to follow the [official instructions](https://github.com/QwenLM/Qwen3-VL) to set up the environment.

2. **Model.**
```
hf download zhouyik/Qwen3-VL-8B-SAMTok --local-dir zhouyik/Qwen3-VL-8B-SAMTok
```
3. **Mask Generation.**
Mask generation means the VLM outputs responses containing mask tokens, and SAMTok decodes these textual mask tokens into 2D arrays.
[Please refer to this script.](demo/qwen3vl_samtok_infer.py)
```python
import re
import numpy as np

def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list

def find_first_index(arr, value):
    indices = np.where(arr == value)[0]
    
    return indices[0] if len(indices) > 0 else -1

def fix_mt_format_comprehensive(text):
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)

    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)

    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from projects.samtok.models import DirectResize, VQ_SAM2, VQ_SAM2Config, SAM2Config

# build VLM
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "zhouyik/Qwen3-VL-8B-SAMTok", torch_dtype="auto"
).cuda().eval()
processor = AutoProcessor.from_pretrained("zhouyik/Qwen3-VL-8B-SAMTok")

# build SAMTok
CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2
sam2_config = SAM2Config(
    ckpt_path="zhouyik/Qwen3-VL-8B-SAMTok/sam2.1_hiera_large.pt",
)
vq_sam2_config = VQ_SAM2Config(
    sam2_config=sam2_config,
    codebook_size=CODEBOOK_SIZE,
    codebook_depth=CODEBOOK_DEPTH,
    shared_codebook=False,
    latent_dim=256,
)
vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
state = torch.load("zhouyik/Qwen3-VL-8B-SAMTok/mask_tokenizer_256x2.pth", map_location="cpu")
vq_sam2.load_state_dict(state)
sam2_image_processor = DirectResize(1024)

# message
image_path = "figs/totoro.jpg"
question = "Could you please give me a detail description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer."
image = Image.open(image_path).convert('RGB')
ori_width, ori_height = image.size
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": question},
        ],
    }
]

# VLM inferece
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(
    **inputs, 
    max_new_tokens=512,
    do_sample=False,
    top_p=1.0,
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# decode mask
quant_ids = extract_mt_token_ids_v1(output_text[0])
if len(quant_ids) % CODEBOOK_DEPTH != 0:
    output_text = [fix_mt_format_comprehensive(output_text[0])]
    quant_ids = extract_mt_token_ids_v2(output_text[0])

batch_size = len(quant_ids) // CODEBOOK_DEPTH
remap_quant_ids = []
tags = []
for bs_id in range(batch_size):
    chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
    tags.append(f"{chunk_quant_ids[0]}-{chunk_quant_ids[1]}")
    remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
    code1 = remap_chunk_quant_ids[0]
    code2 = remap_chunk_quant_ids[1]
    if not (code2 >= 0 and code2 < CODEBOOK_SIZE):
        code2 = -1
    remap_chunk_quant_ids_error_handle = [code1, code2]
    remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

batch_size = len(remap_quant_ids)
sam2_image = np.array(image)
sam2_image = sam2_image_processor.apply_image(sam2_image)
sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

with torch.no_grad():
    _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
_pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
_pred_masks = _pred_masks > 0.5
_pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
text_token_2d_mask_mapping = {tag: _pred_mask for tag, _pred_mask in zip(tags, _pred_masks)}
```

4. **Mask Understanding.**
Mask understanding means encoding 2D masks into mask tokens using SAMTok and incorporating them into user instructions, requiring the VLM to understand the specific image regions referred to by the mask tokens and answer the user's questions. Region captioning is a typical task of mask understanding. [Please refer to this script.](evaluation/qwen3vl/qwen3vl_dam_infer.py)

## 🕹️ Gradio Demo

## 🤖 Training
Please refer to [TRAIN_TOKENIZER.md](docs/TRAIN_TOKENIZER.md) for training tokenizer.

Please refer to [TRAIN_VLM.md](docs/TRAIN_VLM.md) for training VLM.

We implement the RL code based on [Easy-R1](https://github.com/hiyouga/EasyR1), with the only modification lying in the reward design. Please refer to rl/reward_function/text2mask.py for details. The script to launch RL training is available at rl/qwen25vl_3b_mt_gres.sh.

## 🔬 Evaluation
[Please refer to this folder](evaluation)

## 📖 Citation

Please kindly cite our paper if you find this project helpful.

```bibtex
@article{samtok,
  title={SAMTok: Representing Any Mask with Two Words},
  author={Zhou, Yikang and Zhang, Tao and Gong, Dengxian and Wu, Yuanzheng and Tian, Ye and Wang, Haochen and Yuan, Haobo and Wang, Jiacong and Qi, Lu and Wang, Anran and Wang, Zhuochen and Wang, Yujing and Chen, Cheng and Ji, Shunping and Li, Xiangtai},
  journal={CVPR},
  year={2026}
}
```
