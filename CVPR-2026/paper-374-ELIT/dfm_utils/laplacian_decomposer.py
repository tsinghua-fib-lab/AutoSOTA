from typing import Dict

import torch

import os

import numpy as np
import torch.nn.functional as F
import torchvision.io as io
from PIL import Image

class LaplacianDecomposer2D:

    def __init__(self, **config):
        
        self.config = config
        self.stages_count = config["stages_count"]
        # The resolution corresponding to the first pyramid stage
        self.antialias = config.get("antialias", False)
        # The spatial compression factor of the autoencoder
        self.upsample_factors = config.get("upsample_factors", None)
        self.upsampling_args = config.get("upsampling_args", {"mode": "trilinear", "align_corners": False})
        self.downsampling_args = config.get("downsampling_args", {"mode": "trilinear", "align_corners": False})

    def separate_lf_hf(self, tensor: torch.Tensor):
        """
        Extracts high and low frequencies from the input tensor
        Args:
            tensor (torch.Tensor): (batch_size, channels, height, width) The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the low and high frequency components.
                (batch_size, channels, height // 2, width // 2)
                (batch_size,, channels, height, width)
        """

        batch_size, channels, height, width = tensor.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Height and width must be even numbers but received {}, {}.".format(height, width))
        tensor = tensor.contiguous()
        folded_input = tensor.view(-1, channels, height, width)
        low_freq = torch.nn.functional.interpolate(folded_input, scale_factor=1/2, mode='bilinear', align_corners=False, antialias=self.antialias)
        low_freq_upsampled = torch.nn.functional.interpolate(low_freq, scale_factor=2, mode='bilinear', align_corners=False, antialias=self.antialias)

        high_freq = folded_input - low_freq_upsampled
        high_freq = high_freq.view(batch_size, channels, height, width)
        low_freq = low_freq.view(batch_size, channels, height // 2, width // 2)

        return low_freq, high_freq

    def decompose(self, tensor: torch.Tensor) -> Dict:
        """
        Decomposes a given tensor into a Laplacian pyramid
        Args:
            tensor (torch.Tensor): (batch_size, channels, height, width) The input tensor to decompose.
        Returns:
            Dict: A dictionary containing the decomposed Laplacian pyramid.
        """
        result_dict = {}
        inferred_stages_count = self.stages_count

        # Builds the pyramid
        current_tensor = tensor
        for current_stage_idx in reversed(range(inferred_stages_count)):
            if current_stage_idx == 0:
                result_dict[current_stage_idx] = current_tensor
                break
            current_low_freq, high_freq = self.separate_lf_hf(current_tensor)
            current_tensor = current_low_freq
            result_dict[current_stage_idx] = high_freq

        if self.upsample_factors is not None:
            # Apply the upsample factors to the decomposed tensor
            for stage_idx, upsample_factor in enumerate(self.upsample_factors):
                if stage_idx in result_dict:
                    mode = self.upsampling_args.get("mode", "trilinear")
                    align_corners = self.upsampling_args.get("align_corners", False)
                    if mode != 'bilinear' and mode != 'trilinear' and mode != 'bicubic' and mode != 'linear':
                        result_dict[stage_idx] = torch.nn.functional.interpolate(result_dict[stage_idx], scale_factor=upsample_factor, mode=mode, antialias=self.antialias)
                    else:
                        result_dict[stage_idx] = torch.nn.functional.interpolate(result_dict[stage_idx], scale_factor=upsample_factor, mode=mode, align_corners=align_corners, antialias=self.antialias)
                    
        return result_dict

    def recompose(self, decomposed_tensor: Dict) -> torch.Tensor:
        """
        Recomposes a Laplacian pyramid back into the original tensor.
        Args:
            decomposed_tensor (Dict): A dictionary containing the decomposed Laplacian pyramid.
        Returns:
            torch.Tensor: The recomposed tensor of size equal to the size of the last stage of the pyramid
        """
        if self.upsample_factors is not None:
            # Downsample the decomposed tensor
            for stage_idx, upsample_factor in enumerate(self.upsample_factors):
                if stage_idx in decomposed_tensor:
                    downsampling_factor = [1 / factor for factor in upsample_factor]
                    mode = self.downsampling_args.get("mode", "trilinear")
                    align_corners = self.downsampling_args.get("align_corners", False)
                    if mode != 'bilinear' and mode != 'trilinear' and mode != 'bicubic' and mode != 'linear':
                        decomposed_tensor[stage_idx] = torch.nn.functional.interpolate(decomposed_tensor[stage_idx], scale_factor=downsampling_factor, mode=mode, antialias=self.antialias)
                    else:
                        decomposed_tensor[stage_idx] = torch.nn.functional.interpolate(decomposed_tensor[stage_idx], scale_factor=downsampling_factor, mode=mode, align_corners=align_corners, antialias=self.antialias)
                    
        # Rebuilds the pyramid
        current_low_freq = decomposed_tensor[0]
        stages_count = len(decomposed_tensor)
        for current_stage_idx in range(1, stages_count):
            high_freq = decomposed_tensor[current_stage_idx]
            batch_size, channels, height, width = current_low_freq.shape
            current_low_freq = current_low_freq.view(-1, channels, height, width)
            low_freq_upsampled = torch.nn.functional.interpolate(current_low_freq, scale_factor=2, mode='bilinear', align_corners=False, antialias=self.antialias)
            low_freq_upsampled = low_freq_upsampled.view(batch_size, channels, height * 2, width * 2)
            current_low_freq = low_freq_upsampled + high_freq

        return current_low_freq


class LaplacianDecomposer3D(LaplacianDecomposer2D):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_frame_count = config["base_resolution"]
        self.antialias = config.get("antialias", False)
        # The spatial compression factor of the autoencoder
        self.autoencoder_temporal_compression_factor = config["autoencoder_spatial_compression_factor"]
        self.is_causal_ae = config.get("is_causal_ae", True)
        
        if self.autoencoder_temporal_compression_factor > 1 and self.is_causal_ae:
            self.base_frame_count =  1 + (self.base_frame_count - 1) / self.autoencoder_temporal_compression_factor 
        else:
            self.base_frame_count /= self.autoencoder_temporal_compression_factor
    
    def separate_lf_hf(self, tensor: torch.Tensor):
        """
        Extracts high and low frequencies from the input tensor
        Args:
            tensor (torch.Tensor): (batch_size, frames_count, channels, height, width) The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the low and high frequency components.
                (batch_size, frames_count, channels, height // 2, width // 2)
                (batch_size, frames_count, channels, height, width)
        """

        batch_size, frames_count, channels, height, width = tensor.shape
        if height % 2 != 0 or width % 2 != 0 or frames_count % 2 != 0:
            raise ValueError("Height and width must be even numbers but received {}, {}.".format(height, width))

        # batch_size, channels, frames_count, height, width
        folded_input = tensor.permute(0, 2, 1, 3, 4)
        low_freq = torch.nn.functional.interpolate(folded_input, scale_factor=1/2, mode='trilinear', align_corners=False, antialias=self.antialias)
        low_freq_upsampled = torch.nn.functional.interpolate(low_freq, scale_factor=2, mode='trilinear', align_corners=False, antialias=self.antialias)

        high_freq = folded_input - low_freq_upsampled
        # back to batch_size, frames_count, channels, height, width
        high_freq = high_freq.permute(0, 2, 1, 3, 4).contiguous() 
        low_freq = low_freq.permute(0, 2, 1, 3, 4).contiguous()
        
        
        return low_freq, high_freq

    def decompose(self, tensor: torch.Tensor) -> Dict:
        """
        Decomposes a given tensor into a Laplacian pyramid
        Args:
            tensor (torch.Tensor): (batch_size, frames_count, channels, height, width) The input tensor to decompose.
        Returns:
            Dict: A dictionary containing the decomposed Laplacian pyramid.
        """
        result_dict = {}
        inferred_stages_count = self.infer_stages_count(tensor)

        # Builds the pyramid
        current_tensor = tensor
        for current_stage_idx in reversed(range(inferred_stages_count)):
            if current_stage_idx == 0:
                result_dict[current_stage_idx] = current_tensor
                break
            current_low_freq, high_freq = self.separate_lf_hf(current_tensor)
            current_tensor = current_low_freq
            result_dict[current_stage_idx] = high_freq
    
        return result_dict
    
    def recompose(self, decomposed_tensor: Dict) -> torch.Tensor:
        """
        Recomposes a Laplacian pyramid back into the original tensor.
        Args:
            decomposed_tensor (Dict): A dictionary containing the decomposed Laplacian pyramid.
        Returns:
            torch.Tensor: The recomposed tensor of size equal to the size of the last stage of the pyramid
        """

        # Rebuilds the pyramid
        current_low_freq = decomposed_tensor[0]
        stages_count = len(decomposed_tensor)
        for current_stage_idx in range(1, stages_count):
            high_freq = decomposed_tensor[current_stage_idx]
            batch_size, frames_count, channels, height, width = current_low_freq.shape
            current_low_freq = current_low_freq.permute(0, 2, 1, 3, 4)
            low_freq_upsampled = torch.nn.functional.interpolate(current_low_freq, scale_factor=2, mode='trilinear', align_corners=False, antialias=self.antialias)
            # back to batch_size, frames_count, channels, height, width
            low_freq_upsampled = low_freq_upsampled.permute(0, 2, 1, 3, 4).contiguous()
            current_low_freq = low_freq_upsampled + high_freq

        return current_low_freq




def save_video(tensor, outdir, path):
    # tensor is expected to be in (batch_size, frames, channels, height, width)
    video_frames = tensor[0].permute(0, 2, 3, 1).detach().cpu().numpy()  # Permute to (frames, height, width, channels)
    video_frames = (video_frames + 1) / 2  
    # video_frames = (video_frames - video_frames.min()) / (video_frames.max() - video_frames.min()) * 255
    video_frames = video_frames * 255
    video_frames = video_frames.astype(np.uint8)
    video_frames = [Image.fromarray(frame, mode="RGB") for frame in video_frames]  # Convert to RGB frames
    video_frames[0].save(
        os.path.join(outdir, path),
        save_all=True,
        append_images=video_frames[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved video to {os.path.join(outdir, path)}")

   
def _save_image_tensor(image_tensor, filename):
    """ Save image tensor to a file

    Args:
        image_tensor (torch.Tensor): (channels, height, width) Image tensor
        filename (str): Output filename
    """
    import torchvision
    
    image_tensor = image_tensor.squeeze(0).squeeze(0)
    image_tensor = (image_tensor + 1) / 2 * 255.0
    image_tensor = image_tensor.byte()
    image = torchvision.transforms.functional.to_pil_image(image_tensor)
    image.save(filename)

def _test_laplacian_decomposer():
    from pathlib import Path

    import torch
    import torchvision
    from PIL import Image

    # Create a random tensor with shape (batch_size, frames_count, channels, height, width)
    image_path = "data/frequency_decomposition/flux_example/0ed0f27d-113e-4a2d-a059-39d558a75e07.jpg"
    results_directory = "results/laplacian_decomposer/image_test/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    height = 1024
    width = 512

    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((width, height))

    # (batch_size, frames_count, channels, height, width)
    image = torchvision.transforms.functional.pil_to_tensor(pil_image).unsqueeze(0).unsqueeze(0)
    image = (image / 255.0) * 2 - 1

    config = {
        "stages_count": 3,
        "base_resolution": 256,
        "autoencoder_spatial_compression_factor": 1,
    }

    decomposer = LaplacianDecomposer2D(config)
    decomposed_tensor = decomposer.decompose(image)
    recomposed_tensor = decomposer.recompose(decomposed_tensor)

    _save_image_tensor(image, results_directory + "original_image.png")
    for stage_idx in decomposed_tensor.keys():
        _save_image_tensor(decomposed_tensor[stage_idx], results_directory + f"decomposed_stage_{stage_idx}.png")
    _save_image_tensor(recomposed_tensor, results_directory + "recomposed_image.png")

    assert torch.allclose(image, recomposed_tensor, atol=1e-6), "Recomposed tensor does not match the original tensor."

def _test_laplacian_decomposer_skip_stages():
    from pathlib import Path

    import torch
    import torchvision
    from PIL import Image

    # Create a random tensor with shape (batch_size, frames_count, channels, height, width)
    image_path = "data/frequency_decomposition/flux_example/0ed0f27d-113e-4a2d-a059-39d558a75e07.jpg"
    results_directory = "results/laplacian_decomposer/image_test_skip_stages/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    height = 1024
    width = 576

    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((width, height))

    # (batch_size, frames_count, channels, height, width)
    image = torchvision.transforms.functional.pil_to_tensor(pil_image).unsqueeze(0).unsqueeze(0)
    image = (image / 255.0) * 2 - 1

    config = {
        "stages_count": 3,
        "skip_stages": [1],
        "base_resolution": 256,
        "autoencoder_spatial_compression_factor": 1,
    }

    decomposer = LaplacianDecomposerSkipStages2D(config)
    decomposed_tensor = decomposer.decompose(image)
    recomposed_tensor = decomposer.recompose(decomposed_tensor)

    _save_image_tensor(image, results_directory + "original_image.png")
    for stage_idx in decomposed_tensor.keys():
        _save_image_tensor(decomposed_tensor[stage_idx], results_directory + f"decomposed_stage_{stage_idx}.png")
    _save_image_tensor(recomposed_tensor, results_directory + "recomposed_image.png")

    assert torch.allclose(image, recomposed_tensor, atol=1e-6), "Recomposed tensor does not match the original tensor."


def _test_laplacian_decomposer_video():
    from pathlib import Path

    import torch

    # Create a random tensor with shape (batch_size, frames_count, channels, height, width)
    video_path = "samples/sample_video.mp4"
    results_directory = "results/laplacian_decomposer/video_test/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    height = 512
    width = 1024

    video = io.read_video(video_path, pts_unit="sec")[0]  # Read video frames
    # f x h x w x c
    print("video shape", video.shape)
    video = video[:32]
    video = video.permute(0, 3, 1, 2)  # Change to (frames_count, channels, height, width)
    video = F.interpolate(video.permute(1, 0, 2, 3), size=(height, width), mode="bilinear").permute(1, 0, 2, 3)
    
    video = video.unsqueeze(0).float() / 255.0  # Convert to (batch, frames, channels, height, width)
    
    video = (video * 2) - 1.0  # Normalize to [-1, 1]

    config = {
        "stages_count": 3,
        "base_resolution": 256,
        "autoencoder_spatial_compression_factor": 1,
    }
    save_video(video, results_directory , "original_video.gif")
    decomposer = LaplacianDecomposer2D(config)
    decomposed_tensor = decomposer.decompose(video)
    recomposed_tensor = decomposer.recompose(decomposed_tensor)

    
    for stage_idx in decomposed_tensor.keys():
        save_video(decomposed_tensor[stage_idx], results_directory , f"decomposed_stage_{stage_idx}.gif")
    save_video(recomposed_tensor, results_directory, "recomposed_video.gif")

    assert torch.allclose(video, recomposed_tensor, atol=1e-6), "Recomposed tensor does not match the original tensor."


def _test_laplacian_decomposer_video_3d():
    from pathlib import Path

    import torch

    # Create a random tensor with shape (batch_size, frames_count, channels, height, width)
    video_path = "samples/sample_video.mp4"
    results_directory = "results/laplacian_decomposer/video_test_3d/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    height = 512
    width = 1024

    video = io.read_video(video_path, pts_unit="sec")[0]  # Read video frames
    print("video shape", video.shape)
    video = video[:32]
    video = video.permute(0, 3, 1, 2)  # Change to (frames_count, channels, height, width)
    video = F.interpolate(video.permute(1, 0, 2, 3), size=(height, width), mode="bilinear").permute(1, 0, 2, 3)
    video = video.unsqueeze(0).float() / 255.0  # Convert to (batch, frames, channels, height, width)
    video = (video * 2) - 1.0  # Normalize to [-1, 1]

    config = {
        "stages_count": 3,
        "base_resolution": 256,
        "autoencoder_spatial_compression_factor": 1,
    }
    save_video(video, results_directory , "original_video.gif")
    
    decomposer = LaplacianDecomposer3D(config)
    decomposed_tensor = decomposer.decompose(video)
    recomposed_tensor = decomposer.recompose(decomposed_tensor)

    
    for stage_idx in decomposed_tensor.keys():
        save_video(decomposed_tensor[stage_idx], results_directory , f"decomposed_stage_{stage_idx}.gif")
    save_video(recomposed_tensor, results_directory, "recomposed_video.gif")

    assert torch.allclose(video, recomposed_tensor, atol=1e-6), "Recomposed tensor does not match the original tensor."

def _test_laplacian_decomposer_video_synth_vid():
    from pathlib import Path

    import torch

    # Create a random tensor with shape (batch_size, frames_count, channels, height, width)
    results_directory = "results/laplacian_decomposer/video_test_3d_sythn_video/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    height = 512
    width = 1024
    frames = 32

    video = torch.zeros((frames, 3, height, width))
    for i in range(frames):
        if i % 2 == 0:
            video[i] = 1  # White frame
    video = video * 255.0
    print("video shape", video.shape)
    video = video[:32]
    video = video.unsqueeze(0).float() / 255.0  # Convert to (batch, frames, channels, height, width)
    video = (video * 2) - 1.0  # Normalize to [-1, 1]

    config = {
        "stages_count": 3,
        "base_resolution": 256,
        "autoencoder_spatial_compression_factor": 1,
    }
    save_video(video, results_directory , "original_video.gif")
    
    decomposer = LaplacianDecomposer3D(config)
    decomposed_tensor = decomposer.decompose(video)
    recomposed_tensor = decomposer.recompose(decomposed_tensor)

    
    for stage_idx in decomposed_tensor.keys():
        save_video(decomposed_tensor[stage_idx], results_directory , f"decomposed_stage_{stage_idx}.gif")
    save_video(recomposed_tensor, results_directory, "recomposed_video.gif")

    assert torch.allclose(video, recomposed_tensor, atol=1e-6), "Recomposed tensor does not match the original tensor."

def main():

    _test_laplacian_decomposer_skip_stages()
    _test_laplacian_decomposer()
    pass

if __name__ == "__main__":
    main()
