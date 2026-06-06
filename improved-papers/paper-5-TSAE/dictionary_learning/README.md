This is a repository for doing dictionary learning via sparse autoencoders on neural network activations. It was developed by Samuel Marks, Adam Karvonen, and Aaron Mueller. 

For accessing, saving, and intervening on NN activations, we use the [`nnsight`](http://nnsight.net/) package; as of March 2024, `nnsight` is under active development and may undergo breaking changes. That said, `nnsight` is easy to use and quick to learn; if you plan to modify this repo, then we recommend going through the main `nnsight` demo [here](https://nnsight.net/notebooks/tutorials/walkthrough/).

Some dictionaries trained using this repository (and associated training checkpoints) can be accessed at [https://baulab.us/u/smarks/autoencoders/](https://baulab.us/u/smarks/autoencoders/). See below for more information about these dictionaries. SAEs trained with `dictionary_learning` can be evaluated with [SAE Bench](https://www.neuronpedia.org/sae-bench/info) using a convenient [evaluation script](https://github.com/adamkarvonen/SAEBench/tree/main/sae_bench/custom_saes).

# Set-up

Navigate to the to the location where you would like to clone this repo, clone and enter the repo, and install the requirements.
```bash
pip install dictionary-learning
```

We also provide a [demonstration](https://github.com/adamkarvonen/dictionary_learning_demo), which trains and evaluates 2 SAEs in ~30 minutes before plotting the results.

# Using trained dictionaries

You can load and used a pretrained dictionary as follows
```python
from dictionary_learning import AutoEncoder

# load autoencoder
ae = AutoEncoder.from_pretrained("path/to/dictionary/weights")

# get NN activations using your preferred method: hooks, transformer_lens, nnsight, etc. ...
# for now we'll just use random activations
activations = torch.randn(64, activation_dim)
features = ae.encode(activations) # get features from activations
reconstructed_activations = ae.decode(features)

# you can also just get the reconstruction ...
reconstructed_activations = ae(activations)
# ... or get the features and reconstruction at the same time
reconstructed_activations, features = ae(activations, output_features=True)
```
Dictionaries have `encode`, `decode`, and `forward` methods -- see `dictionary.py`.

## Loading JumpReLU SAEs from `sae_lens`
We have limited support for automatically converting SAEs from `sae_lens`; currently this is only supported for JumpReLU SAEs, but we may expand support if users are interested.
```python
from dictionary_learning import JumpReluAutoEncoder

ae = JumpReluAutoEncoder.from_pretrained(
    load_from_sae_lens=True,
    release="your_release_name",
    sae_id="your_sae_id"
)
```
The arguments should should match those used in the `SAE.from_pretrained` call you would use to load an SAE in `sae_lens`. For this to work, `sae_lens` should be installed in your environment.


# Training your own dictionaries

To train your own dictionaries, you'll need to understand a bit about our infrastructure. (See below for downloading our dictionaries.)

This repository supports different sparse autoencoder architectures, including standard `AutoEncoder` ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)), `GatedAutoEncoder` ([Rajamanoharan et al., 2024](https://arxiv.org/abs/2404.16014)), and `AutoEncoderTopK` ([Gao et al., 2024](https://arxiv.org/abs/2406.04093)).
Each sparse autoencoder architecture is implemented with a corresponding trainer that implements the training protocol described by the authors.
This allows us to implement different training protocols (e.g. p-annealing) for different architectures without a lot of overhead.
Specifically, this repository supports the following trainers:
- [`StandardTrainer`](trainers/standard.py): Implements a training scheme similar to that of [Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder).
- [`GatedSAETrainer`](trainers/gdm.py): Implements the training scheme for Gated SAEs described in [Rajamanoharan et al., 2024](https://arxiv.org/abs/2404.16014).
- [`TopKSAETrainer`](trainers/top_k.py): Implemented the training scheme for Top-K SAEs described in [Gao et al., 2024](https://arxiv.org/abs/2406.04093).
- [`BatchTopKSAETrainer`](trainers/batch_top_k.py): Implemented the training scheme for Batch Top-K SAEs described in [Bussmann et al., 2024](https://arxiv.org/abs/2412.06410).
- [`JumpReluTrainer`](trainers/jumprelu.py): Implemented the training scheme for JumpReLU SAEs described in [Rajamanoharan et al., 2024](https://arxiv.org/abs/2407.14435).
- [`PAnnealTrainer`](trainers/p_anneal.py): Extends the `StandardTrainer` by providing the option to anneal the sparsity parameter p.
- [`GatedAnnealTrainer`](trainers/gated_anneal.py): Extends the `GatedSAETrainer` by providing the option for p-annealing, similar to `PAnnealTrainer`.

Another key object is the `ActivationBuffer`, defined in `buffer.py`. Following [Neel Nanda's appraoch](https://www.lesswrong.com/posts/fKuugaxt2XLTkASkk/open-source-replication-and-commentary-on-anthropic-s), `ActivationBuffer`s maintain a buffer of NN activations, which it outputs in batches.

An `ActivationBuffer` is initialized from an `nnsight` `LanguageModel` object, a submodule (e.g. an MLP), and a generator which yields strings (the text data). It processes a large number of strings, up to some capacity, and saves the submodule's activations. You sample batches from it, and when it is half-depleted, it refreshes itself with new text data.

Here's an example for training a dictionary; in it we load a language model as an `nnsight` `LanguageModel` (this will work for any Huggingface model), specify a submodule, create an `ActivationBuffer`, and then train an autoencoder with `trainSAE`.
```python
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

device = "cuda:0"
model_name = "EleutherAI/pythia-70m-deduped" # can be any Huggingface model

model = LanguageModel(
    model_name,
    device_map=device,
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data must be an iterator that outputs strings
data = iter(
    [
        "This is some example data",
        "In real life, for training a dictionary",
        "you would need much more data than this",
    ]
)
buffer = ActivationBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=3e4,  # you can set this higher or lower dependong on your available memory
    device=device,
)  # buffer will yield batches of tensors of dimension = submodule's output dimension

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-3,
    "device": device,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
)
```
Some technical notes our training infrastructure and supported features:
* Training uses the `ConstrainedAdam` optimizer defined in `training.py`. This is a variant of Adam which supports constraining the `AutoEncoder`'s decoder weights to be norm 1.
* Neuron resampling: if a `resample_steps` argument is passed to the Trainer, then dead neurons will periodically be resampled according to the procedure specified [here](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling).
* Learning rate warmup: if a `warmup_steps` argument is passed to the Trainer, then a linear LR warmup is used at the start of training and, if doing neuron resampling, also after every time neurons are resampled.
* Sparsity penalty warmup: if a `sparsity_warmup_steps` is passed to the Trainer, then a linear warmup is applied to the sparsity penalty at the start of training.
* Learning rate decay: if a `decay_start` is passed to the Trainer, then a linear LR decay is used from `decay_start` to the end of training.
* If `normalize_activations` is True and passed to `trainSAE`, then the activations will be normalized to have unit mean squared norm. The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference. This is very helpful for hyperparameter transfer between different layers and models.

If `submodule` is a model component where the activations are tuples (e.g. this is common when working with residual stream activations), then the buffer yields the first coordinate of the tuple.

# Downloading our open-source dictionaries

To download our pretrained dictionaries automatically, run:

```bash
./pretrained_dictionary_downloader.sh
```
This will download dictionaries of all submodules (~2.5 GB) hosted on huggingface. Currently, we provide dictionaries from the `10_32768` training run. This set has dictionaries for MLP outputs, attention outputs, and residual streams (including embeddings) in all layers of EleutherAI's Pythia-70m-deduped model. These dictionaries were trained on 2B tokens from The Pile.

Let's explain the directory structure by example. After using the script above, you'll have a `dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768` directory corresponding to the layer 1 MLP dictionary from the `10_32768` set. This directory contains:
* `ae.pt`: the `state_dict` of the fully trained dictionary
* `config.json`: a json file which specifies the hyperparameters used to train the dictionary
* `checkpoints/`: a directory containing training checkpoints of the form `ae_step.pt` (only if you used the `--checkpoints` flag)

We've also previously released other dictionaries which can be found and downloaded [here](https://baulab.us/u/smarks/autoencoders/). 

## Statistics for our dictionaries

We'll report the following statistics for our `10_32768` dictionaries. These were measured using the code in `evaluation.py`.
* **MSE loss**: average squared L2 distance between an activation and the autoencoder's reconstruction of it
* **L1 loss**: a measure of the autoencoder's sparsity
* **L0**: average number of features active above a random token
* **Percentage of neurons alive**: fraction of the dictionary features which are active on at least one token out of 8192 random tokens
* **CE diff**: difference between the usual cross-entropy loss of the model for next token prediction and the cross entropy when replacing activations with our dictionary's reconstruction
* **Percentage of CE loss recovered**: when replacing the activation with the dictionary's reconstruction, the percentage of the model's cross-entropy loss on next token prediction that is recovered (relative to the baseline of zero ablating the activation)

### Attention output dictionaries

| Layer | Variance Explained (%) | L1 | L0  | % Alive | CE Diff | % CE Recovered |
|-------|------------------------|----|-----|---------|---------|----------------|
| 0     | 92                     | 8  | 128 | 17      | 0.02    | 99             |
| 1     | 87                     | 9  | 127 | 17      | 0.03    | 94             |
| 2     | 90                     | 19 | 215 | 12      | 0.05    | 93             |
| 3     | 89                     | 12 | 169 | 13      | 0.03    | 93             |
| 4     | 83                     | 8  | 132 | 14      | 0.01    | 95             |
| 5     | 89                     | 11 | 144 | 20      | 0.02    | 93             |


### MLP output dictionaries

| Layer  | Variance Explained (%) | L1 | L0  | % Alive | CE Diff | % CE Recovered |
|--------|------------------------|----|-----|---------|---------|----------------|
|     0  | 97                     | 5  | 5   | 40      | 0.10    | 99             |
|     1  | 85                     | 8  | 69  | 44      | 0.06    | 95             |
|     2  | 99                     | 12 | 88  | 31      | 0.11    | 88             |
|     3  | 88                     | 20 | 160 | 25      | 0.12    | 94             |
|     4  | 92                     | 20 | 100 | 29      | 0.14    | 90             |
|     5  | 96                     | 31 | 102 | 35      | 0.15    | 97             |


### Residual stream dictionaries
NOTE: these are indexed so that the resid_i dictionary is the *output* of the ith layer. Thus embeddings go first, then layer 0, etc.

| Layer   | Variance Explained (%) | L1 | L0  | % Alive | CE Diff | % CE Recovered |
|---------|------------------------|----|-----|---------|---------|----------------|
|    embed| 96                     |  1 |  3  | 36      | 0.17    | 98             |
|       0 | 92                     | 11 | 59  | 41      | 0.24    | 97             |
|       1 | 85                     | 13 | 54  | 38      | 0.45    | 95             |
|       2 | 96                     | 24 | 108 | 27      | 0.55    | 94             |
|       3 | 96                     | 23 | 68  | 22      | 0.58    | 95             |
|       4 | 88                     | 23 | 61  | 27      | 0.48    | 95             |
|       5 | 90                     | 35 | 72  | 45      | 0.55    | 92             |




# Extra functionality supported by this repo

**Note:** these features are likely to be depricated in future releases.

We've included support for some experimental features. We briefly investigated them as an alternative approaches to training dictionaries.

* **MLP stretchers.** Based on the perspective that one may be able to identify features with "[neurons in a sufficiently large model](https://transformer-circuits.pub/2022/toy_model/index.html)," we experimented with training "autoencoders" to, given as input an MLP *input* activation $x$, output not $x$ but $MLP(x)$ (the same output as the MLP). For instance, given an MLP which maps a 512-dimensional input $x$ to a 1024-dimensional hidden state $h$ and then a 512-dimensional output $y$, we train a dictionary $A$ with hidden dimension 16384 = 16 x 1024 so that $A(x)$ is close to $y$ (and, as usual, so that the hidden state of the dictionary is sparse).
    * The resulting dictionaries seemed decent, but we decided not to pursue the idea further.
    * To use this functionality, set the `io` parameter of an activaiton buffer to `'in_to_out'` (default is `'out'`).
    * h/t to Max Li for this suggestion.
* **Replacing L1 loss with entropy**. Based on the ideas in this [post](https://transformer-circuits.pub/2023/may-update/index.html#simple-factorization), we experimented with using entropy to regularize a dictionary's hidden state instead of L1 loss. This seemed to cause the features to split into dead features (which never fired) and very high-frequency features which fired on nearly every input, which was not the desired behavior. But plausibly there is a way to make this work better.
* **Ghost grads**, as described [here](https://transformer-circuits.pub/2024/jan-update/index.html). 

# Citation

Please cite the package as follows:

```
@misc{marks2024dictionary_learning,
   title = {dictionary_learning},
   author = {Samuel Marks, Adam Karvonen, and Aaron Mueller},
   year = {2024},
   howpublished = {\url{https://github.com/saprmarks/dictionary_learning}},
}
```
