"""
Model Configuration File
Defines model configurations for all tasks
"""

MODEL_CONFIG = {
    "image_classification": {
        "blip2": {
            "model_name": "Salesforce/blip2-opt-2.7b",
            "model_type": "blip2",
            "device": "cuda",
        },
        "qwen3-vl-8b": {
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            "model_type": "qwen3_vl",
            "device": "cuda",
        },
        "metaclip2": {
            "model_name": "facebook/metaclip-2-worldwide-huge-quickgelu",
            "model_type": "metaclip2",
            "device": "cuda",
        },
        "gpt-4o-mini": { 
            "model_name": "gpt-4o-mini",  
            "model_type": "openai_vision",
            "api_key": None,  
        },
        "clip-vit-h": {
            "model_name": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "model_type": "clip",
            "device": "cuda"
        },
        "internvl3.5-8b": {
            "model_name": "OpenGVLab/InternVL3_5-8B",
            "model_type": "internvl",
            "device": "cuda"
        },
        "sailvl-8b": {
            "model_name": "BytedanceDouyinContent/SAIL-VL2-8B",
            "model_type": "sailvl",
            "device": "cuda"
        },
        "ministral3-vl-8b": {
            "model_name": "mistralai/Ministral-3-8B-Instruct-2512",
            "model_type": "ministral3_vl",
            "device": "cuda"
        },
        "pixtral-12b": {
            "model_name": "mistral-community/pixtral-12b",
            "model_type": "pixtral",
            "device": "cuda"
        },
        "glm-4.6v-flash": {
            "model_name": "zai-org/GLM-4.6V-Flash",
            "model_type": "glm4v",
            "device": "cuda",
            "use_flash_attn": False,  # Flash Attention 2 (requires compatible CUDA environment)
            "use_sdpa": True,         # SDPA acceleration (more stable alternative)
            "use_compile": True       # torch.compile optimization
        },
        "idefics3-8b": {
            "model_name": "HuggingFaceM4/Idefics3-8B-Llama3",
            "model_type": "idefics3",
            "device": "cuda"
        },
        "gemma3-4b": {
            "model_name": "google/gemma-3-4b-it",
            "model_type": "gemma3",
            "device": "cuda"
        },
        "step3-vl-10b": {
            "model_name": "stepfun-ai/Step3-VL-10B",
            "model_type": "step3vl",
            "device": "cuda"
        }
         # Can add more later
        # "resnet50": {...},
        # "vit": {...},
    },   
    "text_classification": {
        "llama3.1-8b": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "model_type": "llama",
            "device": "cuda",
        },
        "gpt-4o-mini": {
            "model_name": "gpt-4o-mini",
            "model_type": "openai",
            "api_key_env": "OPENAI_API_KEY",
        },
        "qwen3-8b": {
            "model_name": "/models/Qwen3-8B",
            "model_type": "qwen3",
            "device": "cuda",
        },
        "ministral-8b": {
            "model_name": "mistralai/Ministral-8B-Instruct-2410",
            "model_type": "ministral",
            "device": "cuda",
        },
        # Can add more later
        # "bert": {...},
    },

    "llm_generation": {
        "llama3.1-8b": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "model_type": "llama",
            "device": "cuda",
            "max_new_tokens": 32,
            "temperature": 0
        },
        "qwen3-8b": {
            "model_name": "/models/Qwen3-8B",
            "model_type": "qwen3",
            "device": "cuda",
            "max_new_tokens": 32,
            "temperature": 0
        },
        "gpt-4o-mini": {
            "model_name": "gpt-4o-mini",
            "model_type": "openai",
            "api_key": None,
            "max_tokens": 32,
            "temperature": 0
        },
        "ministral-8b": {
            "model_name": "mistralai/Ministral-8B-Instruct-2410",
            "model_type": "ministral",
            "device": "cuda",
            "max_new_tokens": 32,
            "temperature": 0
        }
    },
    
    "vlm_tagging": {
        "blip2": {
            "model_name": "Salesforce/blip2-opt-2.7b",
            "model_type": "blip",
            "device": "cuda",
        },
        "qwen3-vl-8b": {                                    # 
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            "model_type": "qwen3_vl",
            "device": "cuda"
        },
        "metaclip2": {                                   
            "model_name": "facebook/metaclip-2-worldwide-huge-quickgelu",
            "model_type": "metaclip2",
            "device": "cuda"
        },
        "gpt-4o-mini": {                                 
            "model_name": "gpt-4o-mini",
            "model_type": "openai",
            "api_key": None
        },
        "llama-vision": {
            "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "model_type": "llama-vision",
            "device": "cuda"
        },
        "flan-t5-xxl": {
            "model_name": "Salesforce/blip2-flan-t5-xxl",
            "model_type": "blip2",
            "device": "cuda"
        },
        "internvl3.5-8b": {
            "model_name": "OpenGVLab/InternVL3_5-8B",
            "model_type": "internvl",
            "device": "cuda"
        },
        "sailvl-8b": {
            "model_name": "BytedanceDouyinContent/SAIL-VL2-8B",
            "model_type": "sailvl",
            "device": "cuda"
        },
        "ministral3-vl-8b": {
            "model_name": "mistralai/Ministral-3-8B-Instruct-2512",
            "model_type": "ministral3_vl",
            "device": "cuda"
        },
        "pixtral-12b": {
            "model_name": "mistral-community/pixtral-12b",
            "model_type": "pixtral",
            "device": "cuda"
        },
        "glm-4.6v-flash": {
            "model_name": "zai-org/GLM-4.6V-Flash",
            "model_type": "glm4v",
            "device": "cuda",
            "use_flash_attn": False,  # Flash Attention 2 (requires compatible CUDA environment)
            "use_sdpa": True,         # SDPA acceleration (more stable alternative)
            "use_compile": True       # torch.compile optimization
        },
        "idefics3-8b": {
            "model_name": "HuggingFaceM4/Idefics3-8B-Llama3",
            "model_type": "idefics3",
            "device": "cuda"
        },
        "step3-vl-10b": {
            "model_name": "stepfun-ai/Step3-VL-10B",
            "model_type": "step3vl",
            "device": "cuda"
        },
        # Can add more later
        # "llava": {...},
    }
}
