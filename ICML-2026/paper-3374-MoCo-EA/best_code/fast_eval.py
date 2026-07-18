"""Fast MoCo-EA evaluation for optimization iterations - linf only."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_cifar10

import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

from mocoea.evolutionary_attack import EvolutionaryAttack
from mocoea.normalization import normalize_cifar10

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/datasets/CIFAR10")
    p.add_argument("--checkpoint", default="/models/resnet18_cifar10_best.pth")
    p.add_argument("--output-dir", default="/repo/results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--population-size", type=int, default=30)
    p.add_argument("--elite-size", type=int, default=5)
    p.add_argument("--mutation-rate", type=float, default=0.2)
    p.add_argument("--mutation-strength", type=float, default=0.02)
    p.add_argument("--max-generations", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--no-bezier-early-stop", action="store_true")
    p.add_argument("--no-bezier-momentum", action="store_true")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--no-saliency-mutate", action="store_true")
    p.add_argument("--no-progressive-eval", action="store_true")
    p.add_argument("--bezier-full-t-samples", action="store_true")
    p.add_argument("--bezier-early-stop-patience", type=int, default=2)
    p.add_argument("--fixed-mutation", action="store_true")
    p.add_argument("--no-bezier-warm-start", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    
    # Get test samples
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    
    samples = []
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            pred = model(normalize_cifar10(img)).argmax(dim=1)
        if pred == label:
            samples.append((img, label))
            if len(samples) >= args.num_samples:
                break
    
    print(f"Collected {len(samples)} correctly classified samples")
    
    # Pass feature flags to EvolutionaryAttack
    feature_flags = {
        "bezier_early_stop": not args.no_bezier_early_stop,
        "bezier_momentum": not args.no_bezier_momentum,
        "warm_start": not args.no_warm_start,
        "saliency_mutate": not args.no_saliency_mutate,
        "progressive_eval": not args.no_progressive_eval,
        "bezier_full_t_samples": not args.bezier_full_t_samples,
        "bezier_early_stop_patience": args.bezier_early_stop_patience,
        "fixed_mutation": args.fixed_mutation,
        "bezier_warm_start": not args.no_bezier_warm_start,
    }
    
    all_results = []
    total_queries = 0
    
    for idx, (x, y) in enumerate(samples):
        ea = EvolutionaryAttack(
            model, eps=8/255, norm="linf",
            population_size=args.population_size,
            elite_size=args.elite_size,
            mutation_rate=args.mutation_rate,
            mutation_strength=args.mutation_strength,
            normalize_fn=normalize_cifar10,
            feature_flags=feature_flags,
        )
        
        stats = ea.evolve(
            x, y,
            max_generations=args.max_generations,
            crossover_type="bezier",
            early_stop_fitness=2.0,
        )
        
        all_results.append({
            "success": stats["success"][-1],
            "generations": stats["final_generation"],
            "queries": stats["query_counts"][-1],
            "time": stats["time_elapsed"][-1],
        })
        
        if (idx + 1) % 10 == 0:
            succ = [r for r in all_results if r["success"]]
            print(f"  [{idx+1}/{len(samples)}] succ={len(succ)}, "
                  f"avg_gen={np.mean([r[generations] for r in succ]):.1f}, "
                  f"avg_queries={np.mean([r[queries] for r in succ]):.0f}")
    
    # Compute final metrics
    succ = [r for r in all_results if r["success"]]
    fail = [r for r in all_results if not r["success"]]
    succ_rate = len(succ) / len(all_results) * 100
    
    print(f"\n{=*60}")
    print(f"MoCo-EA linf RESULTS")
    print(f"{=*60}")
    print(f"Success rate: {succ_rate:.1f}% ({len(succ)}/{len(all_results)})")
    if succ:
        print(f"Avg generations: {np.mean([r[generations] for r in succ]):.1f} +/- {np.std([r[generations] for r in succ]):.1f}")
        print(f"Avg queries: {np.mean([r[queries] for r in succ]):.0f} +/- {np.std([r[queries] for r in succ]):.0f}")
        print(f"Avg time: {np.mean([r[time] for r in succ]):.2f} +/- {np.std([r[time] for r in succ]):.2f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(args.output_dir, f"fast_eval_{ts}.json")
    with open(outfile, "w") as f:
        json.dump({"results": all_results, "args": vars(args)}, f, indent=2)
    print(f"Saved to {outfile}")
    
    # Print metrics in parseable format
    if succ:
        metrics = {
            "Succ. rate": round(succ_rate, 1),
            "Avg. gen.": round(np.mean([r[generations] for r in succ]), 1),
            "Avg. queries": round(np.mean([r[queries] for r in succ]), 0),
            "Avg. time": round(np.mean([r[time] for r in succ]), 2),
        }
        print(f"\nMETRICS_JSON: {json.dumps(metrics)}")
    
    return all_results

if __name__ == "__main__":
    main()
