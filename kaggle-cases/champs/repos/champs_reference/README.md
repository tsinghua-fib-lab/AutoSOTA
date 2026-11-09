# champs_kaggle
Collection of data, scripts and analysis related to the CHAMPS Kaggle competition, [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling).

The dataset_generation (external reference: `./dataset_generation`) folder contains information and scripts related to creating the dataset 
from the QM9 structures.

The dataset (external reference: `./dataset`) folder contains the training data available in the competition as well as the hidden test data.
This data is slightly different than the data used in the competition, as there was a small symmetry related error in the original dataset that only affected the supplementary data (which none of the top teams used) of a small number of molecules.

The analysis (external reference: `./analysis`) folder contains the submitted predictions by the top 400 teams, as well as scripts to analyse this.

The download_everything.sh (external reference: `./download_everything.sh`) script downloads all datasets, checkpoints, intermediate data files etc. available. This is a large amount of data.
