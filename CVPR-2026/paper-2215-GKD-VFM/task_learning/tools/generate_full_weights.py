import torch
import argparse


def main(args):
    segmentor_save_path = args.segmentor_save_path
    backbone_path = args.backbone
    rein_head_path = args.rein_head

    # Load weights from the provided paths
    rein_head_weights = torch.load(rein_head_path, map_location="cpu")
    backbone_weights = torch.load(backbone_path, map_location="cpu")

    # Prefix backbone weights with 'backbone.'
    backbone_weights = {f"backbone.{k}": v for k, v in backbone_weights.items()}

    # Update the REIN head weights with the backbone weights
    if "state_dict" in rein_head_weights:
        rein_head_weights["state_dict"].update(backbone_weights)
    else:
        rein_head_weights.update(backbone_weights)

    # Save the combined weights to the specified path
    torch.save(rein_head_weights, segmentor_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine pre-trained backbone weights and fine-tuned REIN head weights into a complete set of segmentor weights."
    )
    parser.add_argument(
        "--segmentor_save_path",
        required=True,
        help="Path to save the combined segmentor checkpoint",
    )
    parser.add_argument(
        "--backbone", required=True, help="Path to the pre-trained backbone weights"
    )
    parser.add_argument(
        "--rein_head", required=True, help="Path to the fine-tuned REIN head weights"
    )

    args = parser.parse_args()
    main(args)
