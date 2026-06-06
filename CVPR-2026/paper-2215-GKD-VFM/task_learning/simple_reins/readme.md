# Installation
- **Install PyTorch**: Ensure you have PyTorch installed.
- **Install XFormers**: Install the XFormers library for efficient attention mechanisms.

# Using Reins as a Feature Extractor
1. **Download and Convert Pretrained Weights**
   ```bash
   wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
   wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
   python simple_reins/convert_dinov2.py dinov2_vitl14_pretrain.pth dinov2_vitl14_512x512.pth
   ```

2. **Test the Setup**
   ```bash
   python -m simple_reins.test
   ```

3. **Integrate into Your Project**
   ```python
   from simple_reins import ReinsDINOv, get_std_reins_dinov2_large
   model = get_std_reins_dinov2_large()
   model.load_state_dict(torch.load(checkpoint, 'cpu'))
   model.forward(x)
   ```

# `reins` Directory Structure

```markdown
simple_reins
├── clip.py        # Clip model implementation
├── dino_v2.py      # DinoV2 model implementation
├── convert_dinov2.py  # Script to convert Dinov2 pretrained weights
├── peft.py        # PEFT utility functions
├── reins_clip.py   # Reins combined with Clip implementation
├── reins_dinov2.py # Reins combined with DinoV2 implementation
├── reins.py        # Reins core implementation
```

## Key Functions in `reins.py`

Within the `reins.py` file, the following functions are essential:

### `Reins.forward`
- **Input**: Features
- **Output**: Refined features
- **Parameters**:
  - `batch_first`: Set to `True` if the feature shape is `[B, N, C]`; otherwise, `False`.
  - `has_cls_token`: Set to `True` if features include cls_tokens; otherwise, `False`.

### `Reins.forward_delta_feat`
- The core processing pipeline of `Reins.forward`.
