from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.utils import pt_to_pil
import torch
import os
stage_1_model = "DeepFloyd/IF-I-XL-v1.0"
stage_2_model = "DeepFloyd/IF-II-L-v1.0"
stage_3_model = "stabilityai/stable-diffusion-x4-upscaler"

# stage 1
stage_1 = DiffusionPipeline.from_pretrained(stage_1_model, 
                                            variant="fp16", torch_dtype=torch.float16
                                            )
# stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_1.scheduler.config["variance_type"] = None
stage_1.scheduler = DDIMScheduler.from_config(stage_1.scheduler.config)
stage_1.enable_model_cpu_offload()
# exit()
# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    stage_2_model, text_encoder=None, 
    variant="fp16", torch_dtype=torch.float16
)
# stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_2.enable_model_cpu_offload()

# stage 3
# safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
safety_modules = {}
stage_3 = DiffusionPipeline.from_pretrained(stage_3_model, **safety_modules, torch_dtype=torch.float16)
# stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_3.enable_model_cpu_offload()

# prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
prompt = 'a photo of a man'
prompt_=prompt.replace(" ","_")

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt,
                                                       do_classifier_free_guidance=True)

generator = torch.manual_seed(0)

gen_dir = "./output/if"
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)
# stage 1
image = stage_1(prompt_embeds=prompt_embeds, 
                negative_prompt_embeds=negative_embeds, 
                generator=generator, 
                output_type="pt", 
                guidance_scale=7.5,
                num_inference_steps=200).images
# image = stage_1(prompt=prompt, 
#                 generator=generator, 
#                 output_type="pt", 
#                 guidance_scale=0,
#                 num_inference_steps=100).images
pt_to_pil(image)[0].save(f"{gen_dir}/if_stage_I_{stage_1_model.replace('/','-')}_{prompt_}.png")
# pt_to_pil(image)[0].save(f"tt.png")
# stage 2

png_path="tt_ddim.png"
# image=torch.load(png_path)


image = stage_2(
    image=image, 
    # prompt=prompt,
    prompt_embeds=prompt_embeds, 
    negative_prompt_embeds=negative_embeds, 
    generator=generator, 
    guidance_scale=7.5,
    output_type="pt",
    
).images
pt_to_pil(image)[0].save(f"{gen_dir}/if_stage_II_{stage_2_model.replace('/','-')}_{prompt_}.png")
# stage 3
image = stage_3(prompt=prompt, 
                image=image, 
                generator=generator,
                guidance_scale=7.5, 
                noise_level=100).images
image[0].save(f"{gen_dir}/if_stage_III_{stage_3_model.replace('/','-')}_{prompt_}.png")