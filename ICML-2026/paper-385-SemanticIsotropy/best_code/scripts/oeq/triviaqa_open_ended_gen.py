import click
import os
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import shutil

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from semantic_isotropy.datasets.utils import get_entity_page_idx

import datasets

# Suppress warnings from sentence-transformers and related libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

QUESTION_FORMAT = """Write a few paragraphs on '{a}'."""

def match_answer(s):
    titles = set([t.lower() for t in s['entity_pages']['title']])
    ans = set([t.lower() for t in s['answer']['normalized_aliases']])
    return titles.intersection(ans)

def semantic_similarity(s1, s2, model):
    """Compute the cosine similarity between sentence embeddings."""
    emb1 = model.encode([s1])[0]
    emb2 = model.encode(s2)[0]
    return cosine_similarity([emb1], [emb2])

def match_answer_semantic(s, sim_model):
    titles = set([t.lower() for t in s['entity_pages']['title']])
    alias_set = set([t.lower() for t in s['answer']['normalized_aliases']])
    overlap = list(titles.intersection(alias_set))
    if not len(overlap):
        return None, -1, -1

    #edge case check where the normalized aliases are poor
    ans = s['answer']['value'].lower()
    simscores = semantic_similarity(ans, overlap, sim_model)
    eligible = sorted([(o, sim) for o,sim in zip(overlap, simscores) if sim > 0.75], key=lambda x: x[1], reverse=True)
    if not len(eligible):
        return None, -1, -1
    idx = get_entity_page_idx(eligible[0][0], s)
    return eligible[0][0], eligible[0][1], idx

def is_single_first_name(text):
    """Filter out single first names without surnames that could be ambiguous.

    Args:
        text: The answer text to check

    Returns:
        bool: True if this appears to be a single first name (should be filtered), False otherwise
    """
    text = text.strip().lower()

    # Common first names that are often ambiguous when appearing alone
    common_first_names = {
        'aaron', 'adam', 'alan', 'albert', 'alex', 'alexander', 'alfred', 'andrew', 'anthony', 'arthur',
        'benjamin', 'bernard', 'brad', 'brian', 'bruce', 'carl', 'charles', 'chris', 'christopher', 'craig',
        'daniel', 'david', 'dennis', 'douglas', 'edward', 'eric', 'frank', 'gary', 'george', 'gerald',
        'gregory', 'harold', 'henry', 'jack', 'james', 'jason', 'jeffrey', 'jeremy', 'jerry', 'john',
        'jonathan', 'joseph', 'joshua', 'kenneth', 'kevin', 'lawrence', 'mark', 'matthew', 'michael',
        'nicholas', 'patrick', 'paul', 'peter', 'philip', 'raymond', 'richard', 'robert', 'ronald',
        'ryan', 'samuel', 'scott', 'stephen', 'steven', 'thomas', 'timothy', 'william',
        'amanda', 'amy', 'angela', 'anna', 'barbara', 'betty', 'brenda', 'carol', 'carolyn', 'catherine',
        'cheryl', 'christina', 'christine', 'deborah', 'diane', 'donna', 'dorothy', 'elizabeth', 'emily',
        'frances', 'helen', 'janet', 'janice', 'jean', 'jennifer', 'jessica', 'joan', 'judith', 'julie',
        'karen', 'katherine', 'kathleen', 'kelly', 'kimberly', 'laura', 'linda', 'lisa', 'margaret',
        'maria', 'marie', 'marilyn', 'martha', 'mary', 'melissa', 'michelle', 'nancy', 'patricia',
        'rebecca', 'ruth', 'sandra', 'sarah', 'sharon', 'shirley', 'stephanie', 'susan', 'teresa', 'virginia',
        'geoffrey', 'chad', 'jordan', 'phoenix', 'victoria', 'georgia', 'wellington', 'aurora'
    }

    # Check if it's a single word that matches common first names
    words = text.split()
    if len(words) == 1 and text in common_first_names:
        return True

    return False

def is_ambiguous_geographic_term(text):
    """Filter out geographic terms that could refer to multiple locations or entities.

    Args:
        text: The answer text to check

    Returns:
        bool: True if this is an ambiguous geographic term (should be filtered), False otherwise
    """
    text = text.strip().lower()

    # Geographic terms that commonly refer to multiple places or could be confused
    ambiguous_geographic = {
        'washington',  # state, DC, or George Washington
        'victoria',    # queen, state, city, or era
        'wellington',  # duke, city, or boot
        'columbia',    # country, district, university, space shuttle
        'georgia',     # US state, country, or name
        'manila',      # city or type of paper/rope
        'delta',       # airline, Greek letter, or geographic feature
        'aurora',      # natural phenomenon or name
        'phoenix',     # city or mythical bird
        'salem',       # multiple cities
        'springfield', # multiple cities
        'franklin',    # multiple places or Benjamin Franklin
        'madison',     # multiple places or James Madison
        'clinton',     # multiple places or Bill Clinton
        'jackson',     # multiple places or Andrew Jackson
        'lincoln',     # multiple places or Abraham Lincoln
        'richmond',    # multiple cities
        'manchester',  # multiple cities
        'oxford',      # multiple universities/cities
        'cambridge',   # multiple universities/cities
        'newport',     # multiple cities
        'fairfield',   # multiple places
        'chester',     # multiple places
        'dover',       # multiple places
        'albany',      # multiple places
        'troy',        # multiple places or ancient city
        'rome',        # Italy or other cities named Rome
        'paris',       # France or other cities named Paris
        'athens',      # Greece or other cities named Athens
        'milan',       # Italy or other cities named Milan
        'florence',    # Italy or other cities named Florence
        'venice',      # Italy or other cities named Venice
    }

    # Check if it's an ambiguous geographic term
    words = text.split()
    if len(words) == 1 and text in ambiguous_geographic:
        return True

    # Also check for simple geographic patterns that might be ambiguous
    if len(words) == 1 and text.endswith('burg') and len(text) > 4:
        return True  # Many -burg places are duplicated

    return False

@click.command()
@click.option('--output-dir', default='.', help='Directory to save the output CSV files')
@click.option('--overwrite', is_flag=True, help='Overwrite existing output directory if not empty')
@click.option('--cache-dir', default=None, help='Cache directory for HuggingFace datasets')
@click.option('--filter-ambiguous', is_flag=True, default=True, help='Filter out ambiguous entries (single names and ambiguous geographic terms)')
def main(output_dir: str, overwrite: bool, cache_dir: str, filter_ambiguous: bool):
    """Generate open-ended questions from TriviaQA dataset."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load dataset
    if cache_dir:
        dataset = datasets.load_dataset("trivia_qa", "rc.wikipedia", cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset("trivia_qa", "rc.wikipedia")

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

    train_output_file = os.path.join(output_dir, 'triviaqa_train_open_ended_prompts.csv')
    val_output_file = os.path.join(output_dir, 'triviaqa_val_open_ended_prompts.csv')
    combo_output_file = os.path.join(output_dir, 'triviaqa_open_ended_prompts.csv')

    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    def process_samples_inner(dataset_split, output_file, data_list_name):
        data_list = []
        filtered_count = 0
        for idx, sample in enumerate(dataset_split):
            best_answer, _, _ = match_answer_semantic(sample, sim_model)
            if best_answer is not None:
                # Apply filtering to remove ambiguous entries if enabled
                if filter_ambiguous and (is_single_first_name(best_answer) or is_ambiguous_geographic_term(best_answer)):
                    filtered_count += 1
                    continue

                data_list.append({'index': idx, 'answer': best_answer, 'open_ended_question': QUESTION_FORMAT.format(a=best_answer)})
            if idx % 1000 == 0:  # Log progress every 1000 samples
                if filter_ambiguous:
                    logging.info(f"Processed {idx} {data_list_name} samples, filtered {filtered_count} ambiguous entries")
                else:
                    logging.info(f"Processed {idx} {data_list_name} samples")

        if filter_ambiguous:
            logging.info(f"Completed {data_list_name}: filtered {filtered_count} ambiguous entries from {len(data_list) + filtered_count} valid answers")
        else:
            logging.info(f"Completed {data_list_name}: {len(data_list)} entries (no filtering applied)")
        df = pd.DataFrame(data_list)
        df.drop_duplicates(subset=['answer'], inplace=True)
        df.to_csv(output_file, index=False)

    # Process train and validation datasets in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_samples_inner, dataset['train'], train_output_file, 'training'): 'training',
            executor.submit(process_samples_inner, dataset['validation'], val_output_file, 'validation'): 'validation',
        }
        for future in as_completed(futures):
            data_list_name = futures[future]
            try:
                future.result()
                logging.info(f"Completed processing {data_list_name} data")
            except Exception as e:
                logging.error(f"Error processing {data_list_name} data: {e}")

    # Combine results
    train_df = pd.read_csv(train_output_file)
    train_df['idx_cat'] = 'train'
    val_df = pd.read_csv(val_output_file)
    val_df['idx_cat'] = 'val'
    combo_df = pd.concat([train_df, val_df], ignore_index=True)
    combo_df.drop_duplicates(subset=['answer'], inplace=True)
    combo_df[['index', 'idx_cat', 'answer', 'open_ended_question']].to_csv(combo_output_file, index=False)

    logging.info("Processing complete!")

if __name__ == "__main__":
    main()
