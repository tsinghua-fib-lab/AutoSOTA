import argparse
from tqdm import tqdm
import time
import torch
import open_clip
from torch.utils.data import DataLoader

from got import GOT
from utils.utils import *
from utils.data import *
from utils.active_utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Active OT-based Image-Text Retrieval")

    parser.add_argument('--alg', type=str, default='GOT-W', choices=['GOT-W', 'GOT-FGW'], help='Algorithm to use')
    parser.add_argument("--model", type=str, default="ViT-B-32", help="CLIP model name (e.g., ViT-B-32, ViT-L-14)")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="Pretrained weights to use")
    parser.add_argument("--dataset", type=str, default="CIFAR10-C", choices=["CIFAR10-C", "ImageNet-C"], help="Dataset to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for DataLoader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    
    parser.add_argument("--init_sup_ratio", type=float, default=0.0, help="Initial supervised ratio for semi-supervised OT")
    parser.add_argument('--sup_selection_seed', type=int, default=42, help='Random seed for supervised selection')
    parser.add_argument('--query_round', type=int, default=10, help='Number of query rounds')
    parser.add_argument('--query_portion', type=float, default=0.2, help='Portion of queries to make')
    parser.add_argument('--query_mode', type=str, default='offline', choices=['offline'], help='Query mode to use')
    parser.add_argument('--modes', type=str, nargs='+', default=['random'], help='Utility modes to use',
                        choices=['random', 'entropy', 'margin', 'least_confidence', 'dual_tension', 'cost_gap', 'entropy_derivative', 'sq_l2_derivative', 'contrastive', 'density',
                                 'diversity', 'entropy_adjoint_grad', 'sq_l2_adjoint_grad', 'kl_adjoint_grad'])
    parser.add_argument('--weights', type=float, nargs='+', default=None, help='Weights for utility modes')
    parser.add_argument('--eps', type=float, default=1e-2, help='Entropy regularization parameter for OT')

    return parser.parse_args()


def test(vlm, test_loader, text_features, batched_H, model, query_settings=None, is_query=False, device='cpu'):
    query_time = 0.0
    if is_query:
        assert query_settings is not None, "query_settings must be provided when is_query is True"

        modes = query_settings['modes']
        query_portion = query_settings['query_portion']
        query_round = query_settings['query_round']

        num_queried = 0
        num_labeled = 0

    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(
            tqdm(
                test_loader,
                desc=f"Evaluating",
                unit="batch",
                ncols=100
            )
        ):
            images = images.to(device)
            labels = labels.to(device)
            H = batched_H[idx].to(device)

            image_features = vlm.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            num_img = image_features.size(0)

            T, C = model.predict(image_features, text_features, H=H)

            preds = T.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # query
            if is_query:
                visited = set((H.sum(dim=1) > 0).nonzero(as_tuple=True)[0].tolist())
                gnd_map = {i: labels[i].item() for i in range(labels.size(0))}
                query_size = int(query_portion * num_img / query_round)

                start = time.time()
                queried_anchors, _ = query_anchors(
                    T, C,
                    eps=model.eps,
                    modes=modes,
                    visited=visited,
                    gnd_map=gnd_map,
                    batch_size=query_size,
                )
                query_time += time.time() - start

                if len(queried_anchors) > 0:
                    queried_anchors = queried_anchors.cpu()
                    batched_H[idx][queried_anchors[:, 0], queried_anchors[:, 1]] = 1.0
                num_queried += len(queried_anchors)
                num_labeled += batched_H[idx].sum().item()

    recall_1 = correct / total
    if is_query:
        print(f"Number of queried samples: {num_queried}")
        print(f"Total labeled samples: {num_labeled}")

    return recall_1, query_time


def main():
    args = get_args()
    print_config(args, title="Configs of Active OT-based Image-Text Retrieval")

    device = args.device

    vlm, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    vlm = vlm.to(device)
    vlm.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    # Load dataset
    dataset_class = dataset_class_dict.get(args.dataset, None)
    if dataset_class is None:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    dataset = dataset_class(
        root="$HOME/AvAtar-ICML26/datasets",
        train=False,
        transform=preprocess,
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    texts = [f"an image of a {c}" for c in dataset.classes]
    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        text_features = vlm.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    model = GOT(eps=args.eps, mode=args.alg)

    batched_H = get_init_batched_H(loader, num_classes=len(dataset.classes), init_sup_ratio=args.init_sup_ratio, seed=args.sup_selection_seed, dtype=torch.float32)

    query_settings = {
        'modes': args.modes,
        'query_portion': args.query_portion,
        'query_round': args.query_round,
        'query_mode': args.query_mode,
        'sup_selection_seed': args.sup_selection_seed,
        'weights': args.weights
    }

    # Active learning loop
    for query_idx in range(args.query_round + 1):
        print(f"=== Query Round {query_idx} ===")
        acc, query_time = test(
            vlm, loader, text_features, batched_H, model,
            query_settings=query_settings,
            is_query=(query_idx < args.query_round),
            device=device
        )
        print(f"Recall@1: {acc:.4f}, Query Time: {query_time:.4f} seconds")


if __name__ == "__main__":
    main()
