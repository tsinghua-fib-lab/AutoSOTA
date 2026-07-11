# Foundational Models through MACE-JAX

## Installation

To install mace-jax with foundational model support, issue these commands:
```{bash}
git clone https://github.com/ACEsuit/mace-jax /tmp/mace-jax
git -C /tmp/mace-jax switch --detach 7e9d467d1701290b6606a20ff2c625c27e973254
cd - 
sed -i 's/find:/find_namespace:/g' /tmp/mace-jax/setup.cfg
pip install /tmp/mace-jax
pip install h5py
pip install mace-torch
```

## Example

The script `train_spice_example.py` illustrates how to load a foundational MACE
model to **chemtrain**, fine-tune it on a dataset, and export it via
**chemtrain-deploy**.
