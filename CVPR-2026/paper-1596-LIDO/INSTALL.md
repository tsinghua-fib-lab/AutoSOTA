# Installation

The code has been tested with `python 3.10`, `torch==2.4.0`, `torchvision==0.19.0` and `torchsparse==2.1.0`, with **CUDA 12.4**. We provide different options to set up the environment and repreduce the results.

## Python Environment

### Quick Setup

Create a virtual environment and install packages from the `requirements.txt` file:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step-by-step installation

Install PyTorch 2.4.0 with CUDA 12.4:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

Clone and build TorchSparse 2.1.0:

```bash
git clone https://github.com/mit-han-lab/torchsparse.git && cd torchsparse
pip install -r requirements.txt
python setup.py install
```
Install remaining dependencies:

```bash
pip install pyyaml opencv-python-headless matplotlib tensorboard fvcore timm scipy strictyaml scikit-learn tqdm easydict
```

## Singularity

### Pre-built Image

We provide a pre-built Singularity image for a quick setup and usage of the code. You can find the image in this [Drive folder](https://drive.google.com/file/d/1UE5izpqPRhDdohg7MgYEZsrIWPYdSVZC/view?usp=sharing).

```bash
singularity exec --nv ubuntu_cuda124_torchsparse.sif <script>
```

### Build Image

We provide a `.def` file to build from scratch a singularity image with the required packages to reproduce and use the code.

```bash
### build singularity image
singularity build --fakeroot ubuntu_cuda124_torchsparse.sif ubuntu_cuda124_torchsparse.def

### run the script with the singularity image
singularity exec --nv ubuntu_cuda124_torchsparse.sif <script>
```