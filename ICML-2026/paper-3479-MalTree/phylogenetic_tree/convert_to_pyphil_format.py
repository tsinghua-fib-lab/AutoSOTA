import numpy as np
import json
import ast
from tqdm import tqdm


def load_and_reverse_mapping(mapping_file_path):
    with open(mapping_file_path, 'r') as file:
        class_mapping_str = file.read()
    class_mapping = ast.literal_eval(class_mapping_str)
    reversed_mapping = {str(value): key for key, value in class_mapping.items()}
    return reversed_mapping


def convert_matrix_to_phylip(distance_matrix, output_file_path, taxon_names=None):
    try:
        num_taxa = distance_matrix.shape[0]
        if taxon_names is None or len(taxon_names) != num_taxa:
            raise ValueError("Number of taxon names must match the number of taxa in the distance matrix.")
    except AttributeError:
        raise TypeError("distance_matrix must be a NumPy array.")
    except AssertionError as e:
        raise ValueError(e)

        # Attempt to write the PHYLIP file
    try:
        with open(output_file_path, 'w') as file:
            file.write(f"{num_taxa}\n")
            for i in range(num_taxa):
                label = taxon_names[i]  # full taxon label (relaxed PHYLIP, no length limit)
                distances = ' '.join(map(str, distance_matrix[i]))
                file.write(f"{label} {distances}\n")
    except IOError as e:
        raise IOError(f"Error writing to file {output_file_path}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

    print(f"PHYLIP formatted file saved as: {output_file_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a square distance matrix to PHYLIP format')
    parser.add_argument('--matrix', default='distance_matrix.npy',
                        help='Path to the square distance matrix (.npy)')
    parser.add_argument('--embeddings', default='fused_embeddings.json',
                        help='Embeddings JSON used to derive taxon labels (sha -> {label})')
    parser.add_argument('--class-mapping',
                        help='Optional class_mapping.txt mapping family names to numeric labels')
    parser.add_argument('--output', default='final_distance_matrix.phy',
                        help='Output PHYLIP file (default: final_distance_matrix.phy)')
    parser.add_argument('--label-mapping', default='label_mapping.txt',
                        help='Output file recording taxon-label -> sha mapping')
    args = parser.parse_args()

    reversed_mapping = load_and_reverse_mapping(args.class_mapping) if args.class_mapping else None

    with open(args.embeddings, 'r') as file:
        data = json.load(file)

    labels = []
    label_counter = {}
    label_mapping = {}

    for sha, value in tqdm(data.items(), desc="Extracting labels"):
        if isinstance(value, dict):
            raw_label = value.get('label')
            if raw_label is None:
                raw_label = value.get('family')
        else:
            raw_label = None
        if reversed_mapping is not None and raw_label is not None:
            original_label = reversed_mapping.get(str(raw_label), 'Unknown')
        else:
            original_label = str(raw_label) if raw_label is not None else 'Unknown'
        original_label = original_label.replace(" ", "")

        if original_label not in label_counter:
            label_counter[original_label] = 0
            full_label = original_label
        else:
            label_counter[original_label] += 1
            full_label = f"{original_label}_{label_counter[original_label]}"

        labels.append(full_label)
        label_mapping[full_label] = sha

    # Save the taxon-label -> sha mapping so trees can be traced back to samples.
    with open(args.label_mapping, 'w') as mapping_file:
        for full_label, sha in label_mapping.items():
            mapping_file.write(f"{full_label} => {sha}\n")
    print(f"Label mapping saved to: {args.label_mapping}")

    distance_matrix = np.load(args.matrix)
    assert distance_matrix.shape[0] == len(labels), \
        "Distance matrix size and number of labels must match."

    convert_matrix_to_phylip(distance_matrix, args.output, taxon_names=labels)