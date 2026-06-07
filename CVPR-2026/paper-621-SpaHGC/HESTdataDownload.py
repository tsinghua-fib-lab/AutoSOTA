from huggingface_hub import login
import datasets
import pandas as pd

login(token="<REDACTED:HF_TOKEN>")
local_dir = 'hest_data'  # hest will be dowloaded to this folder

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
Lymph_Node_df = meta_df[(meta_df['organ'] == 'Lymph node') & (meta_df['disease_state'] != 'Healthy')]

Pancreas_xenium_df = meta_df[
    (meta_df['organ'] == 'Pancreas') & (meta_df['disease_state'] != 'Healthy') & (meta_df['st_technology'] == 'Xenium')]



ids_to_query = meta_df['id'].values


dataset = datasets.load_dataset(
    'MahmoodLab/hest',
    cache_dir=local_dir,
    patterns=ids_to_query,
    trust_remote_code=True
)
