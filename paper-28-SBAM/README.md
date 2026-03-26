# MAS: Masked Attention by Segment 

## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)

## Description
This repo will contain the official code for the paper: *Segment-Based Attention Masking for GPTs*.

Meanwhile, as a demo, we share model weights and a notebook visualizing MAS attentnion maps.
## Installation

### Prerequisites
Ensure that Python 3.10 is installed on your machine.

All files must be located in the same folder.

### Installing Dependencies
Install the required dependencies in your Python environment using `pip` and the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

### Demo
Ensure all files are in the same folder and run the `demo_MAS.ipynb` notebook.

This script demonstrates how to load a MAS instance of Llama-3.2-1B, generate outputs with it, and visualize its attention maps.

We provide a pre-trained MAS model in the `trained_models_and_results` directory, allowing you to skip the fine-tuning step.

## TODO

[] Release code.


## Citing

```bibtex
@article{katzRingel2024MAS,
  title={Segment-Based Attention Masking for GPTs},
  author={Shahar Katz and Liran Ringel and Yaniv Romano and Lior Wolf},
  journal={arXiv preprint arXiv:2412.18487},
  year={2024}
}
```