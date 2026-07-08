import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64


if __name__ == "__main__":
    print(f"Using device: {DEVICE}, dtype: {DTYPE}")
    