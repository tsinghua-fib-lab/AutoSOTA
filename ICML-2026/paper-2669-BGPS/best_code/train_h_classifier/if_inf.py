import torch
from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler
import matplotlib.pyplot as plt
import os
from PIL import Image

# Load DeepFloyd IF model (Stage 1, 2, 3)
model_name = "DeepFloyd/IF-I-XL-v1.0"  # Replace with the appropriate model version
pipe = DiffusionPipeline.from_pretrained(model_name, variant="fp16", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
# Use DDIM and DDPM schedulers for testing
ddpm_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.config["variance_type"] = None
ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


os.makedirs("output2", exist_ok=True)
# Helper function to generate and display images
def generate_image(prompt_embeds, negative_embeds, scheduler, num_steps, guidance_scale=7.0):
    # Set the scheduler for the pipeline
    pipe.scheduler = scheduler
    pipe.set_progress_bar_config(disable=True)

    # Generate image
    with torch.no_grad():
        result = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]
    
    # Display image
    # plt.figure(figsize=(6, 6))
    # plt.imshow(result)
    # plt.axis("off")
    # plt.title(f"Scheduler: {scheduler.__class__.__name__}, Steps: {num_steps}")
    # plt.show()
    #save image
    result.save(f"output2/output_{scheduler.__class__.__name__}_{num_steps}.png")

# Test configuration
prompt = "A photo of a person working as a doctor."
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt,
                                                       do_classifier_free_guidance=True)

timesteps_list = [50, 100]
schedulers = [ddim_scheduler, ddpm_scheduler]

# Generate images for each combination of scheduler and timesteps
for scheduler in schedulers:
    for num_steps in timesteps_list:
        generate_image(prompt_embeds, negative_embeds, scheduler, num_steps)

# for all images saved, apply the stage 2 and stage 3 models to upscale them
stage_1_model = "DeepFloyd/IF-I-XL-v1.0"
stage_2_model = "DeepFloyd/IF-II-L-v1.0"
stage_3_model = "stabilityai/stable-diffusion-x4-upscaler"

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    stage_2_model, text_encoder=None, 
    variant="fp16", torch_dtype=torch.float16
)
# stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch
stage_2.enable_model_cpu_offload()
# stage 3
# safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
safety_modules = {}
stage_3 = DiffusionPipeline.from_pretrained(stage_3_model, **safety_modules, torch_dtype=torch.float16)
# stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch
stage_3.enable_model_cpu_offload()  
gen_dir = "./output2/if_upscaled"
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)
for scheduler in schedulers:
    for num_steps in timesteps_list:
        img_path = f"output2/output_{scheduler.__class__.__name__}_{num_steps}.png"
        # Load the image from file path
        loaded_image = Image.open(img_path)
        # stage 2
        image = stage_2(image=loaded_image, 
                        prompt_embeds=prompt_embeds, 
                        negative_prompt_embeds=negative_embeds, 
                        output_type="pt").images
        # stage 3
        upscaled_image = stage_3(image=image, prompt=prompt, output_type="pil").images[0]
        upscaled_image.save(f"{gen_dir}/if_upscaled_{scheduler.__class__.__name__}_{num_steps}.png")
        