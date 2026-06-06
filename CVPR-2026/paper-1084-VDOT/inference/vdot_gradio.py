# -*- coding: utf-8 -*-
import argparse
import time
import os
import sys
import warnings
import random
import logging
import queue
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
import gradio as gr

warnings.filterwarnings('ignore')

JOB_QUEUE = queue.Queue()
RESULT_QUEUE = queue.Queue()

wan = None
WanVace = None
WAN_CONFIGS = None
SIZE_CONFIGS = None
cache_video = None

def setup_distributed():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=30))
        
    return rank, world_size, local_rank

def load_model_distributed(model_name, ckpt_dir, rank, world_size, local_rank):
    logging.info(f"Rank {rank}: Loading model {model_name}...")
    cfg = WAN_CONFIGS[model_name]
    
    ulysses_size = world_size
    ring_size = 1
    t5_fsdp = True 
    dit_fsdp = True 
    
    if ulysses_size > 1 or ring_size > 1:
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        init_distributed_environment(rank=rank, world_size=world_size)
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=ring_size,
            ulysses_degree=ulysses_size,
        )

    wan_vace = WanVace(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=t5_fsdp,
        dit_fsdp=dit_fsdp,
        use_usp=(ulysses_size > 1 or ring_size > 1),
        t5_cpu=False,
    )
    
    #wan_vace.eval()
    logging.info(f"Rank {rank}: Model loaded successfully (Ulysses={ulysses_size}).")
    return wan_vace

def gradio_inference_trigger(
    prompt, model_name_ui, ckpt_dir, size_key, frame_num, 
    sample_steps, sample_shift, guide_scale, seed, 
    src_video_path, src_mask_path, src_ref_images_path, use_prompt_extend
):
    p_src_video = src_video_path.strip() if src_video_path and src_video_path.strip() else None
    p_src_mask = src_mask_path.strip() if src_mask_path and src_mask_path.strip() else None
    p_src_ref = src_ref_images_path.strip() if src_ref_images_path and src_ref_images_path.strip() else None

    logging.info(f"Received Task -> Video: {p_src_video}")

    payload = {
        "type": "inference",
        "params": {
            "prompt": prompt,
            "model_name": model_name_ui,
            "ckpt_dir": ckpt_dir,
            "size_key": size_key,
            "frame_num": int(frame_num),
            "sample_steps": int(sample_steps),
            "sample_shift": float(sample_shift),
            "guide_scale": float(guide_scale),
            "seed": int(seed),
            "src_video": p_src_video,
            "src_mask": p_src_mask,
            "src_ref_images": p_src_ref,
            "use_prompt_extend": use_prompt_extend
        }
    }
    
    JOB_QUEUE.put(payload)
    result = RESULT_QUEUE.get() 
    
    if result["status"] == "success":
        return result["video_path"], result["final_prompt"], result["msg"]
    else:
        return None, None, f"Error: {result['msg']}"

def main():
    custom_temp_dir = os.path.join(os.getcwd(), "gradio_temp_cache")
    os.makedirs(custom_temp_dir, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = custom_temp_dir
    print(f"🔧 Gradio temp dir set to: {custom_temp_dir}")
    
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.ERROR)

    global wan, WanVace, WAN_CONFIGS, SIZE_CONFIGS, cache_video
    logging.info(f"Rank {rank}: Importing Wan libraries...")
    import wan
    from wan.utils.utils import cache_video
    from models.wan import WanVace
    from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS

    DEFAULT_MODEL_NAME = "vace-14B"
    DEFAULT_CKPT_DIR = 'VDOT'
    
    try:
        model = load_model_distributed(DEFAULT_MODEL_NAME, DEFAULT_CKPT_DIR, rank, world_size, local_rank)
    except Exception as e:
        logging.error(f"Rank {rank} failed to load model: {e}")
        return

    if rank == 0:
        with gr.Blocks(css='style.css', title=f"VDOT-14B ({world_size} GPUs)") as demo:
            gr.Markdown(f"## VDOT-14B (Running on {world_size} GPUs)")
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(label="Prompt", lines=6, value="在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏...")
                    with gr.Accordion("Input Paths", open=True):
                        video_in = gr.Textbox(label="Src Video Path", placeholder="Path (Optional)")
                        mask_in = gr.Textbox(label="Src Mask Path", placeholder="Path (Optional)")
                        img_in = gr.Textbox(label="Ref Images Paths", placeholder="Path (Optional)")
                    with gr.Accordion("Advanced Settings", open=False):
                        ckpt_input = gr.Textbox(label="Ckpt Path", value=DEFAULT_CKPT_DIR, visible=False)
                        model_name_input = gr.Textbox(label="Model Name", value=DEFAULT_MODEL_NAME, interactive=False)
                        size_input = gr.Dropdown(choices=list(SIZE_CONFIGS.keys()), value="480p", label="Size")
                        frame_num = gr.Number(value=81, label="Frames")
                        steps = gr.Slider(1, 40, 4, step=1, label="Steps")
                        shift = gr.Slider(1.0, 32.0, 16.0, label="Shift")
                        guide = gr.Slider(1.0, 20.0, 1.0, label="Guidance")
                        seed = gr.Number(value=42, label="Seed")
                        extend = gr.Dropdown(['plain', 'wan_zh', 'wan_en'], value='plain', label="Prompt Extend")
                    btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    out_vid = gr.Video(label="Result")
                    out_prompt = gr.Textbox(label="Extended Prompt")
                    out_log = gr.Textbox(label="Log")

            btn.click(gradio_inference_trigger, 
                      [prompt_input, model_name_input, ckpt_input, size_input, frame_num, steps, shift, guide, seed, video_in, mask_in, img_in, extend], 
                      [out_vid, out_prompt, out_log])
        
        logging.info("Starting Gradio server...")
        demo.queue(max_size=10, api_open=False)
        demo.launch(
            server_name='0.0.0.0', 
            server_port=10023, 
            share=True, 
            prevent_thread_lock=True 
        )

    logging.info(f"Rank {rank}: Ready for tasks...")
    
    while True:
        broadcast_list = [None]
        
        if rank == 0:
            try:
                task = JOB_QUEUE.get(timeout=1.0)
                broadcast_list = [task]
            except queue.Empty:
                broadcast_list = [None]
            except Exception as e:
                logging.error(f"Queue error: {e}")
                broadcast_list = [None]
        
        try:
            dist.broadcast_object_list(broadcast_list, src=0)
        except Exception as e:
            logging.error(f"Rank {rank} Broadcast Error: {e}")
            continue

        task = broadcast_list[0]
        if task is None: continue
        if task.get("type") == "exit": break
            
        if task.get("type") == "inference":
            p = task["params"]
            try:
                if rank == 0: logging.info(f"Start generating...")
                
                current_seed = p['seed']
                if current_seed < 0: current_seed = random.randint(0, sys.maxsize)
                
                device = local_rank
                size_cfg = SIZE_CONFIGS[p['size_key']]
                src_ref_list = p['src_ref_images'].split(',') if p['src_ref_images'] else None
                
                src_video, src_mask, src_ref_images = model.prepare_source(
                    [p['src_video']], [p['src_mask']], [src_ref_list], 
                    p['frame_num'], size_cfg, device
                )

                video = model.generate(
                    p['prompt'], src_video, src_mask, src_ref_images,
                    size=size_cfg, frame_num=p['frame_num'], shift=p['sample_shift'],
                    sample_solver='unipc', sampling_steps=p['sample_steps'],
                    guide_scale=p['guide_scale'], seed=current_seed, offload_model=True 
                )
                
                if rank == 0:
                    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                    save_dir = os.path.join('results_gradio', p['model_name'], timestamp)
                    os.makedirs(save_dir, exist_ok=True)
                    save_file = os.path.join(save_dir, 'out_video.mp4')
                    
                    cache_video(tensor=video[None], save_file=save_file, fps=WAN_CONFIGS[p['model_name']].sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                    
                    RESULT_QUEUE.put({"status": "success", "video_path": save_file, "final_prompt": p['prompt'], "msg": "Success"})
                
                dist.barrier()

            except Exception as e:
                logging.error(f"Rank {rank} Error: {e}")
                import traceback
                traceback.print_exc()
                if rank == 0:
                    RESULT_QUEUE.put({"status": "error", "msg": str(e)})

if __name__ == "__main__":
    main()