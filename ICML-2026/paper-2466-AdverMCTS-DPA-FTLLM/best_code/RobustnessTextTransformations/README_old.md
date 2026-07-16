# Mark My Words

Project homepage: https://wagner-group.github.io/projects/markmywords/index.html
Paper link: https://arxiv.org/abs/2312.00273

A benchmark for LLM watermarking schemes. Python 3.9 or higher required. 

## Installation

* Recursively clone repo: `git clone --recursive git@github.com:wagner-group/MarkMyWords.git`
* Setup python virtual environment: `python3.9 -m venv env && source env/bin/activate`
* Run installation script: `bash install.sh`


## Usage

The benchmark comes with a set of predefined watermarks. You can use these directly, or implement your own.

### Using existing watermarks

* Write the specs of the watermarks you want in a file. Specify one watermark per line. Each line must contain a json object describing the watermark parameters. You can find examples of watermark specs in `./run/watermark_specs`, and a full definition of the WatermarkSpec object in `./src/watermark-benchmark/utils/classes.py`.
* Setup configuration file. An example file is provided in `./run/config.yml`
* Run benchmark: `watermark-benchmark-run config.yml`

### Implementing your own watermarks

Watermarks can be added in the `./src/watermark-benchmark/watermark/schemes` folder. In order to add a watermark, you must:
* Create a new file in the aforementionned folder with your watermark definition
* Modify `./src/watermark-benchmark/watermark/main.py:get_watermark` so WatermarkSpecs can be parsed to generate your watermark
* [Optional] if you need additional parameters for your watermark, edit `./src/watermark-benchmark/utils/classes.py:WatermarkSpec` to add them.
* Re-install the package using `python setup.py install` at the root of the repo.

At a minimum, watermarks must inherit from the `./src/watermark-benchmark/watermark/templates/generator.py:Watermark` class. For each token, the `process` function is called, and must return the modified logits. The `verify` function is called to detect watermarks. It must return an array containing pairs, each pair denoting a verifier, and a result. Each result is an array of 4-tuples `(watermarked:bool, score:float, pval:float, token_idx:int)`, where `watermarked` is a boolean indicating if the sequence up to `token_idx` is watermarked, `score` is the score for that sequence, and `pval` the likelihood of that sequence under the null hypothesis. 

We define a taxonomy for watermarks in our paper. Buildings blocks for it are already implemented, such as Internal and External randomness sources. If your watermark can be expressed in this taxonomy, you can use existing classes to facilitate implementation:

* The `./src/watermark-benchmark/watermark/templates/verifier.py:Verifier` class can be used to implement verifiers. Each generator can have multiple verifiers. If using an empirical verifier for an align, edit or sum score, you can inherit from the `EmpiricalVerifier` class, and you just need to define the `score_matrix` and `random_score_matrix` functions. The first returns a matrix of the z_{i,j} values (from the taxonomy), the second returns a matrix of z_{i,j} for a random key (used for the T-test).
* The `./src/watermark-benchmark/watermark/templates/random.py:Randomness` class can be inherited to create additional randomness sources.

## Config file

A full list of configuration options can be found in `./src/watermark-benchmark/utils/classes.py:ConfigSpec`. A YAML file is expected. Below are the more important options.
```
model: 'meta-llama/Llama-2-7b-chat-hf'              # Which model to use
engine: 'vllm'                                      # Which backend to use (currently vllm and huggingface are supported)
output: 'generation_results'                        # Folder to write generation results
watermark: 'watermark_specs'                        # File containing watermarking specs
seed: 0                                             # Randomness seed
huffman_coding: 'static_data/encoding.tsv'          # Huffman encoding for binary scheme
paraphrase: False                                   # Whether to use paraphrasing attacks
openai_key: 'KEY'                                   # OpenAI key
misspellings: 'static_data/misspellings.json'       # Misspelling pairs for attack
devices: [0,1,2,3]                                  # CUDA devices to use                

# Summarize parameter
results: 'results'                                  # Folder containing the summarization results
aggregate_thresholds: [[0.02, 1], [0.1, 1], [0.02, 0.8], [0.1, 0.8], [0.02, -1], [0.1, -1]]        # Thresholds
```

## Wishlist

* Binary tests
* Add LogitProcessor to VLLM project
