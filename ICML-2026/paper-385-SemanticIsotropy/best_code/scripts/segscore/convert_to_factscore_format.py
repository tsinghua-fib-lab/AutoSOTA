"""
Convert to FactScore Format: A script to process input data and convert it to FactScore format.
"""

import os
import json
import click
from pathlib import Path
from typing import List, Dict, Any, Generator

from semantic_isotropy.datasets.loaders import load_data
from semantic_isotropy.pipeline.utils import init_logger

# Set up logging
logger = init_logger(__name__, 'INFO')


def data_generator(input_data: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Generator function to process input data and convert to FactScore format.

    This function processes each record and converts it to the expected FactScore format.
    Currently, it adds basic processing and formatting.

    Args:
        input_data: List of input records to process

    Yields:
        Processed records in FactScore format
    """

    for record in input_data:
        for response in record['responses']:
            processed_record = {'topic': record['entity'], 'output': response['response']}
            yield processed_record


def save_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        output_path: Path to the output JSONL file
    """
    logger.info(f"Saving {len(data)} records to {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Successfully saved results to {output_path}")

@click.command()
@click.option('--input-path', '-i', required=True, type=click.Path(exists=False),
              help='Path to the input file (JSON, JSONL, or CSV)')
@click.option('--output-path-fmt', '-o', default='{fname}_factscore_output.jsonl',
              help='Path to the output JSONL file (default: {fname}_factscore_output.jsonl)')
@click.option('--dryrun', is_flag=True, default=False, help='If set, do not write output, just print summary and exit.')
def main(input_path: str, output_path_fmt: str, dryrun: bool):
    """
    Process input data and convert it to FactScore format.

    This script loads data using the project's load_data function, processes it
    using a generator function, and outputs the results as a JSONL file.
    """
    # Load input file using the project's load_data function
    logger.info(f"Loading data from {input_path}")
    input_data = load_data(input_path)

    # Process data using generator function
    logger.info("Processing data using generator function...")
    results = list(data_generator(input_data))

    output_path = os.path.join(os.path.dirname(input_path), output_path_fmt.format(fname=os.path.basename(input_path).replace('.json', '')))

    if dryrun:
        logger.info(f"[DRYRUN] Would save {len(results)} records to {output_path}")
        return

    # Save results as JSONL
    logger.info(f"Saving results to {output_path}")
    save_jsonl(results, output_path=output_path)

    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
