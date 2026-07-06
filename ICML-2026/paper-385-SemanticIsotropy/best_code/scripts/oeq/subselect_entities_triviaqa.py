import pandas as pd
import pickle
import json
from semantic_isotropy.datasets.loaders import load_triviaqa
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def semantic_similarity(s1, s2, model):
    """Compute the cosine similarity between sentence embeddings."""
    emb1 = model.encode([s1])[0]
    emb2 = model.encode(s2)[0]
    return cosine_similarity([emb1], [emb2])

def check_entity_ref_alignment(d, triviaqa, sim_model, debug=False):
    te = triviaqa[d['idx_cat'].replace('val', 'validation')][int(d['index'])]
    entity = d['entity']
    ans = te['answer']['value'].lower()
    simscores = semantic_similarity(ans, [entity], sim_model)
    if simscores[0] < 0.75:
        if debug:
            print(f"{data_repr(d)} answer does not line up with reference doc on semantic match. Excluding from dataset")
        return False
    return True

def process_data(json_path, csv_path, output_path):
    """
    Process data by loading JSON, creating an index in a pandas DataFrame,
    and writing out a subselected result.

    Args:
        json_path (str): Path to the JSON file
        csv_path (str): Path to the CSV file
        output_path (str): Path to write the output CSV
    """
    triviaqa = load_triviaqa()
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')


    # 1. Load JSON file
    with open(json_path, 'r') as json_file:
        raw_data = json.load(json_file)

    data = list(filter(lambda d: check_entity_ref_alignment(d, triviaqa=triviaqa, sim_model=sim_model), raw_data))
    index_keys = [f"{d['index']}-{d['idx_cat']}" for d in data]

    # 2. Load CSV into pandas DataFrame
    df = pd.read_csv(csv_path)

    # 3. Create index from JSON
    # Assuming json_data is a list of keys or a dictionary with keys to index

    # 4. Subselect DataFrame based on the index
    # This assumes the index matches a column in the DataFrame
    # Modify the column name as needed based on your specific data
    df['index-col'] = df['index'].astype(str) + '-' + df['idx_cat'].astype(str)
    result_df = df[df['index-col'].isin(index_keys)]

    # 5. Write out the result
    result_df.to_csv(output_path, index=False)

    print(f"Processed data written to {output_path}")
    print(f"Rows in result: {len(result_df)}")

if __name__ == "__main__":
    # Example usage - update these paths as needed
    process_data(
        json_path='path/to/seg_score.json',
        csv_path='path/to/triviaqa_oe_prompts.csv',
        output_path='path/to/triviaqa_oe_prompts_clean.csv'
    )
