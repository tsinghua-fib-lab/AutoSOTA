import json
import os
import os.path as osp
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

from decoupledmarket.constant import Save_Path


class PerformanceMonitor:
    """Collect timing, database, market, and API usage metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
        self.start_times = {}
        self.agent_times = defaultdict(list)
        self.db_operation_times = []
        self.market_operation_times = []
        self.api_call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def start_timer(self, operation: str, agent_id: Optional[int] = None):
        key = f"{operation}_{agent_id}" if agent_id is not None else operation
        self.start_times[key] = time.time()

    def end_timer(self, operation: str, agent_id: Optional[int] = None):
        key = f"{operation}_{agent_id}" if agent_id is not None else operation
        if key not in self.start_times:
            return 0
        elapsed = time.time() - self.start_times[key]
        with self.lock:
            self.metrics[operation].append(elapsed)
            if agent_id is not None:
                self.agent_times[agent_id].append({"operation": operation, "time": elapsed})
        del self.start_times[key]
        return elapsed

    def record_db_operation(self, operation: str, duration: float):
        with self.lock:
            self.db_operation_times.append({"operation": operation, "duration": duration})

    def record_market_operation(self, operation: str, duration: float):
        with self.lock:
            self.market_operation_times.append({"operation": operation, "duration": duration})

    def record_api_call(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ):
        with self.lock:
            self.api_call_count += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += (
                total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
            )

    def get_statistics(self) -> Dict:
        stats = {}
        with self.lock:
            for operation, times in self.metrics.items():
                if times:
                    sorted_times = sorted(times)
                    stats[operation] = {
                        "count": len(times),
                        "total": sum(times),
                        "avg": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "median": sorted_times[len(sorted_times) // 2],
                    }

            stats["api"] = {
                "api_call_count": self.api_call_count,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            }

        self._append_grouped_stats(stats, "database", self.db_operation_times)
        self._append_grouped_stats(stats, "market", self.market_operation_times)
        return stats

    @staticmethod
    def _append_grouped_stats(stats: Dict, key: str, rows: List[Dict]):
        if not rows:
            return
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["operation"]].append(row["duration"])
        stats[key] = {}
        for operation, times in grouped.items():
            stats[key][operation] = {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }

    def identify_bottlenecks(self) -> List[Dict]:
        stats = self.get_statistics()
        operations = []
        for operation, data in stats.items():
            if isinstance(data, dict) and "avg" in data:
                operations.append(
                    {
                        "operation": operation,
                        "avg_time": data["avg"],
                        "total_time": data["total"],
                        "count": data["count"],
                    }
                )
        operations.sort(key=lambda item: item["avg_time"], reverse=True)
        total_time = sum(item["total_time"] for item in operations)
        for item in operations:
            item["percentage"] = item["total_time"] / total_time * 100 if total_time else 0
        return operations[:10]

    def save_report(self, filename: Optional[str] = None):
        if filename is None:
            log_dir = osp.join(Save_Path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = osp.join(log_dir, f"performance_report_{timestamp}.json")
        report = {
            "statistics": self.get_statistics(),
            "bottlenecks": self.identify_bottlenecks(),
            "agent_times": dict(self.agent_times),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Performance report saved to {filename}")
        return filename

    def print_summary(self):
        print("\n" + "=" * 60)
        print("Performance summary")
        print("=" * 60)
        for i, bottleneck in enumerate(self.identify_bottlenecks(), 1):
            print(
                f"{i}. {bottleneck['operation']}: "
                f"avg={bottleneck['avg_time']:.4f}s "
                f"total={bottleneck['total_time']:.4f}s "
                f"count={bottleneck['count']}"
            )
        api = self.get_statistics().get("api", {})
        print(
            "API calls: "
            f"{api.get('api_call_count', 0)}, "
            f"tokens: {api.get('total_tokens', 0)}"
        )
        print("=" * 60 + "\n")


_monitor = None
_monitor_lock = threading.Lock()


def get_monitor() -> PerformanceMonitor:
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = PerformanceMonitor()
        return _monitor


def reset_monitor() -> PerformanceMonitor:
    global _monitor
    with _monitor_lock:
        _monitor = PerformanceMonitor()
        return _monitor
