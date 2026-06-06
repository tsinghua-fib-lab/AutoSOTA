import os
import pandas as pd
import numpy as np
import torch

from src.utils_train import smiles_to_fp
from src.models import BounDrE, MLP, Aligner

def score_from_smiles(test_smiles, project_dir, device='cuda:0'):
    print(f'Loading pretrained models from project directory: {project_dir}')
    aligner = Aligner(device=device)
    aligner.load_state_dict(torch.load(os.path.join(project_dir,'multimodal_alignment_model.pt'), map_location=device))
    aligner.to(device)

    train_drug_fps = np.array([smiles_to_fp(smiles) for smiles in test_smiles])
    X_test = aligner.encode_fp(torch.Tensor(train_drug_fps).to(device))

    encoder = MLP()
    model = BounDrE(encoder)
    model_dict = torch.load(os.path.join(project_dir,'boundary_model.pt'), map_location=device)
    model.load_state_dict(model_dict['model'])
    model.to(device)
    model.encoder.device = device
    model.R = model_dict['R'].to(device)
    model.c = model_dict['c'].to(device)

    scores = model.score_from_embeddings(X_test).cpu().numpy()
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Score drug-likeness')
    parser.add_argument('--input_smiles', type=str, default='demo/test.smi', help='input smiles file')
    parser.add_argument('--output', type=str, default='projects/test_run/output.csv', help='output file')
    parser.add_argument('--project_dir', type=str, default='projects/pretrained', help='project directory to load files')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    args = parser.parse_args()

    test_smiles = pd.read_csv(args.input_smiles, header=None)[0].tolist()
    print(f"Loaded {len(test_smiles)} SMILES from {args.input_smiles}")

    scores = score_from_smiles(test_smiles, args.project_dir, args.device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame({'smiles': test_smiles, 'score': scores})
    df.to_csv(args.output, index=False)
    print(f"Drug-likeness scores saved to {args.output}")

if __name__ == '__main__':
    main()