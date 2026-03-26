# Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability. 

# Table of Contents
 1. [Installation](#Installation)
 2. [Codebase Structure](#Codebase-Structure)
 3. [Training a T-SAE](#Training-a-T-SAE)
 4. [Loading From HuggingFace](#Loading-From-HuggingFace)
 5. [Replicating Experiments](#Replicating-Experiments)

# Installation
This project uses Poetry for package management. To install, simply install Poetry and run

```
poetry install
```
We use `.env` to manage environment variables. Please rename `example.env` to `.env` and fill out as needed. If you just use the default HuggingFace settings this may not be necessary, but feel free to add your own environment variables to `.env`. If you do not want to customize HuggingFace environment variables, feel free to remove the `load_dotenv` calls from the code or just leave `.env` empty.


# Codebase Structure
The codebase consists of two parts:
 * A fork of the [dictionary_learning](https://github.com/saprmarks/dictionary_learning) repo for training Temporal Sparse Autocencoders.
 * A `src` folder containing code to replicate the experiments in the paper with a trained SAEs. Note that some experiments require interpreted latents for the TSAE, which we labeled using [SAEBench](https://github.com/adamkarvonen/SAEBench). To replicate our experiments without training your own TSAE, you can download our Gemma-2-2b TSAE with a width of 16384 and provided explanations from Huggingface.

 # Training a T-SAE

 The `dictionary_learning` directory is a fork of the [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks, Adam Karvonen, and Aaron Mueller. In `dictionary_learning/dictionary_learning/trainers/temporal_sequence_top_k.py` we provide standard scaffolding for training Temporal SAEs under the class `TemporalMatryoshkaBatchTopKSAE`. To train a Temporal SAE, use the entrypoint `dictionary_learning/dictionary_learning/train_temporal.py` where you can set parameters such as regularization, split fraction, and traditional SAE hyperparameters.

 To evaluate any SAE with our additional smoothness metrics, use `dictionary_learning/dictionary_learning/eval_temporal.py`.
 
 # Loading from HuggingFace

We have supplied a trained instance of a Temporal SAE with explanations on [HuggingFace](https://huggingface.co/alex-oesterling/temporal-saes). We label explanations using the `autointerp` package on [SAEBench](https://github.com/adamkarvonen/SAEBench). The explanations are in `explanations.json`, but some indices are missing due to being dead features. We conducted automated interpretability using Llama-3.3-70b-Instruct to generate feature explanations.

To use the model, simply clone the repository from HuggingFace and load it using the `load_dictionary` function in `dictionary_learning/dictionary_learning/utils.py`. 

 # Replicating Experiments

 Most experiments have the following flags:  
 * `--model`: The LLM to use with the SAE. 
 * `--layer`: The layer in the LLM on which the SAE was trained. 
 * `--checkpoints_path`: The the path to where the trained SAEs are stored. By default `dictionary_learning` saves SAEs to a certain path within the repo but for collaboration and storage we modify this to save elsewhere.
 * `--run_name`: The specific SAE inside the checkpoints path to evaluate.  
 * `--checkpoint`: If left blank, will load the fully-trained SAE (`ae.pt`), but can also be used to load specific training checkpoints (stored in `checkpoints/ae_1000.pt` for the 1000th epoch checkpoint, for example).  
 * `--chunk`: 0 for high-level split, 1 for low-level split, and 2 for both (used with Matryoshka and Temporal SAEs).  
 
 
 ## Sequence Interpretability (Figures 1 and 4)
 (Under construction!)

 ## Probing
To conduct probing simply call the `expeiments/probing.py` script. You can specify dataset (finefineweb or wikipedia), base LLM, SAE, and for Matryoshka and Temporal SAEs, the split (high, low, or both) over which to conduct probing. To compare with probing on the model's dense latents, pass `baseline_model` into the `--run_name` flag.

Example:
 ```
 python src/experiments/probing.py --run_name "sae_run_name" --chunk 0 --checkpoints_path "path/to/checkpoints"
 ```

 ## TSNE
 To conduct tSNE on the SAE feature space, call `experiments/tsne.py'. Similarly, you can specify dataset, LLM, SAE, and feature split.

 Example:
 ```
 python src/experiments/tsne.py --run_name "sae_run_name" --chunk 0 --checkpoints_path "path/to/checkpoints"
 ```

 ## Dataset Understanding Case Study
 To decompose HH-RLHF with a Temporal SAE, simply call 
 ```
 python src/experiments/alignment_study.py --run_name "sae_run_name" --chunk 0 --checkpoints_path "path/to/checkpoints" --explanation_path "path/to/explanations.json"
 ```
 
 ## Steering
 (Under construction!)
