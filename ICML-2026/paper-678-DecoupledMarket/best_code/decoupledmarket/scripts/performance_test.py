"""Run performance comparisons for simulation configurations."""
import time
import argparse
import json
import os.path as osp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from decoupledmarket.constant import Save_Path
from decoupledmarket.main_parallel import overall_test_parallel, overall_test_original
from decoupledmarket.performance_monitor import reset_monitor, get_monitor
import multiprocessing


def run_performance_test(config):
    """Run one performance test configuration."""
    print(f"\n{'='*60}")
    print(f"Testing configuration: {config['name']}")
    print(f"{'='*60}")

    reset_monitor()
    monitor = get_monitor()

    start_time = time.time()

    if config['mode'] == 'original':
        overall_test_original()
    else:
        overall_test_parallel(
            executor_type=config.get('executor', 'thread'),
            max_workers=config.get('workers', None),
            batch_size=config.get('batch_size', 20),
            enable_monitoring=True
        )

    total_time = time.time() - start_time

    stats = monitor.get_statistics()
    bottlenecks = monitor.identify_bottlenecks()

    result = {
        'config': config,
        'total_time': total_time,
        'statistics': stats,
        'bottlenecks': bottlenecks
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Performance test runner')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to a JSON configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run a small quick-test configuration set')

    args = parser.parse_args()


    if args.quick:
        test_configs = [
            {
                'name': 'Original sequential execution',
                'mode': 'original',
            },
            {
                'name': 'Thread execution (4 workers)',
                'mode': 'parallel',
                'executor': 'thread',
                'workers': 4,
            },
        ]
    else:
        cpu_count = multiprocessing.cpu_count()
        test_configs = [
            {
                'name': 'Original sequential execution',
                'mode': 'original',
            },
            {
                'name': f'Thread execution ({cpu_count} workers)',
                'mode': 'parallel',
                'executor': 'thread',
                'workers': cpu_count,
            },
            {
                'name': f'Thread execution ({cpu_count * 2} workers)',
                'mode': 'parallel',
                'executor': 'thread',
                'workers': cpu_count * 2,
            },
            {
                'name': 'Batch execution (batch_size=10)',
                'mode': 'parallel',
                'executor': 'batch',
                'batch_size': 10,
                'workers': cpu_count,
            },
            {
                'name': 'Batch execution (batch_size=20)',
                'mode': 'parallel',
                'executor': 'batch',
                'batch_size': 20,
                'workers': cpu_count,
            },
            {
                'name': 'Batch execution (batch_size=50)',
                'mode': 'parallel',
                'executor': 'batch',
                'batch_size': 50,
                'workers': cpu_count,
            },
        ]


    if args.config_file:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            test_configs = json.load(f)

    results = []

    for config in test_configs:
        try:
            result = run_performance_test(config)
            results.append(result)
        except Exception as e:
            print(f"Configuration '{config['name']}' failed: {e}")
            import traceback
            traceback.print_exc()


    log_dir = osp.join(Save_Path, "logs")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = osp.join(log_dir, f"performance_test_results_{timestamp}.json")

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


    print("\n" + "="*60)
    print("Performance test results")
    print("="*60)

    print(f"\n{'Configuration':<30} {'Total time (s)':<15} {'Speedup':<10}")
    print("-" * 60)

    baseline_time = None
    for result in results:
        config_name = result['config']['name']
        total_time = result['total_time']

        if baseline_time is None:
            baseline_time = total_time
            speedup = 1.0
        else:
            speedup = baseline_time / total_time

        print(f"{config_name:<30} {total_time:<15.2f} {speedup:<10.2f}x")

    print("Bottleneck analysis")
    for result in results:
        print(f"\n{result['config']['name']}:")
        bottlenecks = result['bottlenecks'][:5]
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"  {i}. {bottleneck['operation']}: "
                  f"avg {bottleneck['avg_time']:.4f}s "
                  f" {bottleneck['percentage']:.2f}%")

    print(f"\nDetailed results saved to: {result_file}")


if __name__ == "__main__":
    main()
