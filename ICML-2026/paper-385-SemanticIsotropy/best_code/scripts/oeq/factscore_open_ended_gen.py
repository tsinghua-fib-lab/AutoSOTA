import click
import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import shutil

QUESTION_FORMAT = """Write a few paragraphs on '{a}'."""

def process_samples(dataset_split, output_file, data_list_name):
    data_list = []
    for idx, sample in enumerate(dataset_split):
        data_list.append({'index': idx, 'answer': sample, 'open_ended_question': QUESTION_FORMAT.format(a=sample)})
        if idx % 1000 == 0:  # Log progress every 1000 samples
            logging.info(f"Processed {idx} {data_list_name} samples")
    df = pd.DataFrame(data_list)
    df.drop_duplicates(subset=['answer'], inplace=True)
    df.to_csv(output_file, index=False)

@click.command()
@click.option('--output-dir', default='.', help='Directory to save the output CSV files')
@click.option('--overwrite', is_flag=True, help='Overwrite existing output directory if not empty')
def main(output_dir: str, overwrite: bool):
    """Generate open-ended questions from FActScore bio entities dataset."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load dataset
    dataset_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bio_entities.txt")
    with open(dataset_file, 'r') as f:
        entities = [line.strip() for line in f.readlines()]

    # Handle output directory
    logging.info(f"Output directory: {output_dir}")

    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    elif os.listdir(output_dir) and not overwrite:
        logging.error(f"Output directory {output_dir} is not empty. Use --overwrite to overwrite.")
        raise click.ClickException(f"Output directory {output_dir} is not empty. Use --overwrite to overwrite.")
    elif overwrite and os.listdir(output_dir):
        # Delete existing directory and recreate it
        logging.info(f"Overwriting output directory: {output_dir}")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    combo_output_file = os.path.join(output_dir, 'factscore_bio_wiki_prompts.csv')

    # Run processing in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_samples, entities, combo_output_file, 'training'): 'training',
        }
        for future in as_completed(futures):
            data_list_name = futures[future]
            try:
                future.result()
                logging.info(f"Completed processing {data_list_name} data")
            except Exception as e:
                logging.error(f"Error processing {data_list_name} data: {e}")

    # Post-process output
    out_df = pd.read_csv(combo_output_file)
    out_df['idx_cat'] = 'combo'
    out_df.drop_duplicates(subset=['answer'], inplace=True)
    out_df[['index', 'idx_cat', 'answer', 'open_ended_question']].to_csv(combo_output_file, index=False)

    logging.info("Processing complete!")

if __name__ == "__main__":
    main()
