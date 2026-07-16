# Measuring Coding Challenge Competence With APPS
This is the repository for [Measuring Coding Challenge Competence With APPS](https://arxiv.org/pdf/2105.09938) by
[Dan Hendrycks\*](https://danhendrycks.com/), [Steven Basart\*](https://stevenbas.art), [Saurav Kadavath](http://www.sauravkadavath.com), Mantas Mazeika, [Akul Arora](https://github.com/akulaarora), Ethan Guo, [Collin Burns](http://collinpburns.com), Samir Puranik, [Horace He](http://horace.io), [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/), and [Jacob Steinhardt](https://www.stat.berkeley.edu/~jsteinhardt/).

Download the [**APPS dataset here**](https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz). (~1.3GB)

This repository contains both training and evaluation code.

Fine-tuned GPT-2 1.5B and GPT-Neo 2.7B weights are available [here](https://drive.google.com/file/d/1XW1Od9L-5l9zXl1HUCyER5pS9zQTbIvU/view?usp=sharing).

For other benchmarks of enormous Transformers, see a dataset which tests ability in [competition math](https://github.com/hendrycks/math), a dataset which tests knowledge of [ethics](https://github.com/hendrycks/ethics), and [a dataset spanning 50+ academic subjects](https://github.com/hendrycks/test).

## How to Use

The training instructions are specified in [train/README](train/README.md) and similarly the evaluation instructions are specified in [eval/README](eval/README.md).

### Hugging Face

The dataset is also available in [Hugging Face datasets](https://huggingface.co/datasets/codeparrot/apps) under apps.

## Citation

If you find this useful in your research, please consider citing

    @article{hendrycksapps2021,
      title={Measuring Coding Challenge Competence With APPS},
      author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
      journal={NeurIPS},
      year={2021}
    }
