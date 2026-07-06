[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7837349.svg)](https://doi.org/10.5281/zenodo.7837349)
# Retrosynthesis prediction using an end-to-end graph neural network for molecular graph editing
Inspired by the arrow-pushing formalism in chemical reaction mechanisms, we present a novel end-to-end architecture for retrosynthesis prediction, Graph2Edits, based on graph neural network to predict the edits of the product graph in an auto-regressive manner, and sequentially generates transformation intermediates and final reactants according to the predicted edits sequence. 
## Environment Requirements  
Create a virtual environment to run the code of Graph2Edits.
Install pytorch with the cuda version that fits your device.
```
conda create -n graph2edits python=3.7 \
conda activate graph2edits \
conda install -c conda-forge rdkit=2019.09.2 \
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch \
pip install numpy==1.17.3 \  
```
## Data preprocessing
1) generate the edit labels and the edits sequence for reaction
```
python preprocess.py --mode train \
python preprocess.py --mode valid \
python preprocess.py --mode test \
```
2) prepare the data for training
```
python prepare_data.py
```
## Train Graph2Edits model
Go to the graph2edits folder and run the following to train the model with specified dataset (default: USPTO_50k)
```
python train.py --dataset uspto_50k --use_rxn_class False
```
The trained model will be saved at graph2edits/experiments/uspto_50k/without_rxn_class/
## Evaluate using a trained model
To evaluate the trained model, run
```
python eval.py
```
to get the raw prediction file saved at graph2edits/experiments/.../pred_results.txt
## Reproducing our results
To reproduce our results, run
```
python eval.py --dataset uspto_50k --use_rxn_class False or True --experiments 27-06-2022--10-27-22 or 30-06-2022--00-19-29
```
This will display the results for reaction class unknown and known setting
