import os
import sys
import pickle
import argparse

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from s3f import protein


def load_protein(file):
    try:
        data_object = protein.from_pdb_string(open(file).read()).to_dict()
    except ValueError as err:
        print("Fail to load pdb %s." % os.path.basename(file), err)
        return

    file = os.path.basename(file)
    name, ext = os.path.splitext(file)
    output_fname = os.path.join(output_dir, name+".pkl")
    with open(output_fname, "wb") as f:
        pickle.dump(data_object, f, protocol=pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str)
parser.add_argument("-o", "--output_dir", type=str)
args = parser.parse_known_args()[0]

if __name__ == "__main__":
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(args.input_dir)
    for file in tqdm(files):
        load_protein(os.path.join(args.input_dir, file))
        
