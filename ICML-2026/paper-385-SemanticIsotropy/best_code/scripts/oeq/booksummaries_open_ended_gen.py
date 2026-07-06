import click
import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import shutil

QUESTION_FORMAT = """Write a few paragraphs about the book '{t}' written by {a} in the year {pubdate}."""

def process_samples(dataset_split, output_file, data_list_name):
    data_list = []
    for idx, sample in enumerate(dataset_split):
        title, author, pub_dt_str = sample
        data_list.append({'index': idx, 'author': author, 'title': title, 'pub_dt': pub_dt_str, 'open_ended_question': QUESTION_FORMAT.format(t=title, a=author, pubdate=pub_dt_str)})
        if idx % 200 == 0:  # Log progress every 200 samples
            logging.info(f"Processed {idx} {data_list_name} samples")
    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False)

@click.command()
@click.option('--booksummaries-file', help='Path to booksummaries.txt downloaded from: https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset/data')
@click.option('--output-dir', default='.', help='Directory to save the output CSV files')
@click.option('--overwrite', is_flag=True, help='Overwrite existing output directory if not empty')
def main(booksummaries_file: str, output_dir: str, overwrite: bool):
    """Generate open-ended questions from BookSummaries File bio entities dataset."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load dataset
    if not os.path.exists(booksummaries_file):
        raise FileNotFoundError("booksummaries_file is missing! Validate path.")
    df = pd.read_table(booksummaries_file, names=["wiki_art_id", "freebase_id","title","author","pub_date", "genres", "plot"])

    # Filtering Logic
    df['plot_len'] = df['plot'].str.len()
    top_df = df[df['plot_len'] > 5000]
    top_df = top_df.dropna(subset='pub_date')
    top_df['pub_dt'] = pd.to_datetime(top_df['pub_date'], format='mixed', errors='coerce')
    top_df = top_df[~top_df['pub_dt'].isnull()]
    top_df = top_df[top_df['pub_dt'] > '2000-01-01']
    top_df['pub_dt_str'] = top_df['pub_dt'].dt.strftime('%B %-d, %Y')
    top_df = top_df.drop_duplicates(subset=['title', 'author', 'pub_dt'])

    entities = [(r['title'], r['author'], r['pub_dt_str']) for _, r in top_df.iterrows()]

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

    combo_output_file = os.path.join(output_dir, 'booksummaries_prompts.csv')
    summary_output_file = os.path.join(output_dir, 'booksummaries_summaries.csv')

    process_samples(entities, combo_output_file, 'training')

    # Post-process output
    out_df = pd.read_csv(combo_output_file)
    out_df['idx_cat'] = 'combo'
    out_df[['index', 'idx_cat', 'author', 'title', 'pub_dt', 'open_ended_question']].to_csv(combo_output_file, index=False)
    top_df = top_df.reset_index(drop=True).reset_index()
    top_df['idx_cat'] = 'combo'
    top_df.to_csv(summary_output_file, index=False)

    logging.info("Processing complete!")

if __name__ == "__main__":
    main()
