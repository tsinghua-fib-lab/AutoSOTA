"""Full 30-sample MoCo-EA linf evaluation with configurable optimizations."""
import sys, os, json, time, argparse
sys.path.insert(0, '/repo')
import patch_cifar10
import numpy as np
import torch, torchvision, torchvision.transforms as transforms
from torchvision.models import resnet18
from mocoea.evolutionary_attack import EvolutionaryAttack
from mocoea.normalization import normalize_cifar10

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feature-preset', default='all_on', choices=['all_on', 'all_off', 'custom'])
    p.add_argument('--output', default='/repo/results/opt_eval.json')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-samples', type=int, default=30)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0')

    # Load model
    model = resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    ckpt = torch.load('/models/resnet18_cifar10_best.pth', map_location=device)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    # Get test samples
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=False, transform=transform)
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

    if args.feature_preset == 'all_on':
        feature_flags = {}
    elif args.feature_preset == 'all_off':
        feature_flags = {
            'warm_start': False, 'bezier_early_stop': False,
            'bezier_momentum': False, 'bezier_full_t_samples': True,
            'progressive_eval': False, 'saliency_mutate': False,
            'fixed_mutation': True, 'bezier_warm_start': False,
        }
    else:
        feature_flags = json.loads(os.environ.get('FEATURE_FLAGS', '{}'))

    print(f"Starting full eval: preset={args.feature_preset}, samples={len(samples)}")
    t_start = time.time()
    results = []

    for idx, (x, y) in enumerate(samples):
        ea = EvolutionaryAttack(model, eps=8/255, norm='linf', normalize_fn=normalize_cifar10,
                               population_size=30, elite_size=5, mutation_rate=0.2, mutation_strength=0.02,
                               feature_flags=feature_flags)
        stats = ea.evolve(x, y, max_generations=1000, crossover_type='bezier', early_stop_fitness=2.0)
        results.append({
            'success': stats['success'][-1],
            'generations': stats['final_generation'],
            'queries': stats['query_counts'][-1],
            'time': stats['time_elapsed'][-1],
        })

        if (idx + 1) % 10 == 0:
            succ = [r for r in results if r['success']]
            print(f"  [{idx+1}/{len(samples)}] succ={len(succ)}/{idx+1}, "
                  f"avg_gen={np.mean([r['generations'] for r in succ]):.1f}, "
                  f"avg_q={np.mean([r['queries'] for r in succ]):.0f}, "
                  f"avg_t={np.mean([r['time'] for r in succ]):.2f}s")

    succ = [r for r in results if r['success']]
    fail = [r for r in results if not r['success']]
    succ_rate = len(succ) / len(results) * 100

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({args.feature_preset})")
    print(f"{'='*60}")
    print(f"Success rate: {succ_rate:.1f}% ({len(succ)}/{len(results)})")
    print(f"Total wall time: {total_time:.1f}s")
    if succ:
        print(f"Avg generations: {np.mean([r['generations'] for r in succ]):.1f} +/- {np.std([r['generations'] for r in succ]):.1f}")
        print(f"Avg queries: {np.mean([r['queries'] for r in succ]):.0f} +/- {np.std([r['queries'] for r in succ]):.0f}")
        print(f"Avg time: {np.mean([r['time'] for r in succ]):.2f} +/- {np.std([r['time'] for r in succ]):.2f}")

    metrics = {}
    if succ:
        metrics = {
            'Succ. rate': round(succ_rate, 1),
            'Avg. gen.': round(np.mean([r['generations'] for r in succ]), 1),
            'Avg. queries': round(np.mean([r['queries'] for r in succ]), 0),
            'Avg. time': round(np.mean([r['time'] for r in succ]), 2),
        }
    print(f"\nMETRICS_JSON: {json.dumps(metrics)}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'results': results, 'metrics': metrics, 'preset': args.feature_preset}, f, indent=2)

    return metrics

if __name__ == '__main__':
    main()
