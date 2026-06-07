import argparse
from PIL import Image
from torchvision import transforms
import os
from torch.utils import data as data



def parse_args_paired_testing(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default=None,)
    parser.add_argument("--lq_path", type=str, default=None,)
    
    # parser.add_argument("--base_config", default="./configs/sr_test.yaml", type=str)

    # details about the model architecture
    parser.add_argument("--sd_path", type=str, default=None,)
    parser.add_argument("--dem_path", type=str, default=None,)
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)

    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--chop_size", type=int, default=128, choices=[512, 256, 128], help="Chopping forward.")
    parser.add_argument("--chop_stride", type=int, default=96, help="Chopping stride.")
    parser.add_argument("--padding_offset", type=int, default=32, help="padding offset.")

    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    # training details
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

class InfraredLQDataset(data.Dataset):

    def __init__(self, data_folder):
        super().__init__()
        self.data_folder = data_folder
        self.img_names = os.listdir(data_folder)
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_folder, img_name)
        input_img = Image.open(img_path).convert("RGB")
        img_t = transforms.ToTensor()(input_img)

        return {"lq": img_t, "lq_path": img_path}

