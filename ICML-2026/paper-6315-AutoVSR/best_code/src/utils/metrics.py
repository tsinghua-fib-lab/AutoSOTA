"""
Metrics Tracking Utilities

Tracks execution time and token consumption for LLM calls.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


@dataclass
class TokenUsage:
    """Token usage statistics"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, other: 'TokenUsage'):
        """Accumulate another TokenUsage instance"""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class StageMetrics:
    """Metrics for a single stage"""
    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "duration_seconds": round(self.duration_seconds, 2),
            "tokens": self.tokens.to_dict(),
            "llm_calls": self.llm_calls,
        }


class MetricsTracker:
    """Global metrics tracker"""
    
    def __init__(self):
        self.stages: Dict[str, StageMetrics] = {}
        self.total_start_time: float = 0.0
        self.total_end_time: float = 0.0
        self._current_stage: Optional[str] = None
    
    def start_tracking(self):
        """Start overall tracking"""
        self.total_start_time = time.time()
        self.stages = {}
    
    def end_tracking(self):
        """End overall tracking"""
        self.total_end_time = time.time()
    
    @contextmanager
    def track_stage(self, stage_name: str):
        """Context manager to track a specific stage"""
        self._current_stage = stage_name
        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(stage_name=stage_name)
        
        metrics = self.stages[stage_name]
        metrics.start_time = time.time()
        
        try:
            yield metrics
        finally:
            metrics.end_time = time.time()
            metrics.duration_seconds += (metrics.end_time - metrics.start_time)
            self._current_stage = None
    
    def record_llm_call(self, response, stage_name: Optional[str] = None):
        """Record token consumption for an LLM call
        
        Args:
            response: LangChain LLM response object
            stage_name: Name of the stage (optional, defaults to current stage)
        """
        stage = stage_name or self._current_stage
        if not stage or stage not in self.stages:
            return
        
        metrics = self.stages[stage]
        metrics.llm_calls += 1
        
        # Attempt to extract token information from the response
        tokens = self._extract_tokens(response)
        if tokens:
            metrics.tokens.add(tokens)
    
    def _extract_tokens(self, response) -> Optional[TokenUsage]:
        """Extract token information from an LLM response"""
        usage = TokenUsage()
        
        # Try multiple ways to obtain token information
        
        # 1. response_metadata (LangChain standard)
        if hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            if 'usage_metadata' in metadata:
                um = metadata['usage_metadata']
                usage.input_tokens = um.get('input_tokens', 0)
                usage.output_tokens = um.get('output_tokens', 0)
                usage.total_tokens = um.get('total_tokens', 0)
                return usage
            if 'token_usage' in metadata:
                tu = metadata['token_usage']
                usage.input_tokens = tu.get('prompt_tokens', 0)
                usage.output_tokens = tu.get('completion_tokens', 0)
                usage.total_tokens = tu.get('total_tokens', 0)
                return usage
        
        # 2. usage_metadata attribute
        if hasattr(response, 'usage_metadata'):
            um = response.usage_metadata
            if um:
                usage.input_tokens = getattr(um, 'input_tokens', 0) or um.get('input_tokens', 0) if isinstance(um, dict) else 0
                usage.output_tokens = getattr(um, 'output_tokens', 0) or um.get('output_tokens', 0) if isinstance(um, dict) else 0
                usage.total_tokens = getattr(um, 'total_tokens', 0) or um.get('total_tokens', 0) if isinstance(um, dict) else 0
                return usage
        
        # 3. Direct usage attribute
        if hasattr(response, 'usage'):
            u = response.usage
            if u:
                usage.input_tokens = getattr(u, 'prompt_tokens', 0) or getattr(u, 'input_tokens', 0)
                usage.output_tokens = getattr(u, 'completion_tokens', 0) or getattr(u, 'output_tokens', 0)
                usage.total_tokens = getattr(u, 'total_tokens', usage.input_tokens + usage.output_tokens)
                return usage
        
        return None
    
    def get_total_tokens(self) -> TokenUsage:
        """Get total token consumption"""
        total = TokenUsage()
        for metrics in self.stages.values():
            total.add(metrics.tokens)
        return total
    
    def get_total_duration(self) -> float:
        """Get total duration (seconds)"""
        if self.total_end_time and self.total_start_time:
            return self.total_end_time - self.total_start_time
        return sum(m.duration_seconds for m in self.stages.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a complete statistical summary"""
        total_tokens = self.get_total_tokens()
        total_duration = self.get_total_duration()
        total_llm_calls = sum(m.llm_calls for m in self.stages.values())
        
        return {
            "total_duration_seconds": round(total_duration, 2),
            "total_tokens": total_tokens.to_dict(),
            "total_llm_calls": total_llm_calls,
            "stages": {name: m.to_dict() for name, m in self.stages.items()},
        }
    
    def print_summary(self):
        """Print the statistical summary"""
        summary = self.get_summary()
        
        print(f"\n{'─' * 60}")
        print(f"📊 METRICS SUMMARY")
        print(f"{'─' * 60}")
        print(f"   ⏱️  Total Duration: {summary['total_duration_seconds']:.2f}s")
        print(f"   🔢 Total Tokens: {summary['total_tokens']['total_tokens']:,}")
        print(f"       Input:  {summary['total_tokens']['input_tokens']:,}")
        print(f"       Output: {summary['total_tokens']['output_tokens']:,}")
        print(f"   📞 LLM Calls: {summary['total_llm_calls']}")
        
        if summary['stages']:
            print(f"\n   📋 By Stage:")
            for name, stage in summary['stages'].items():
                print(f"      {name:15} {stage['duration_seconds']:6.2f}s | "
                      f"tokens: {stage['tokens']['total_tokens']:>6} | "
                      f"calls: {stage['llm_calls']}")


# Global tracker instance (optional usage)
_global_tracker: Optional[MetricsTracker] = None


def get_global_tracker() -> MetricsTracker:
    """Get the global tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker()
    return _global_tracker


def reset_global_tracker():
    """Reset the global tracker"""
    global _global_tracker
    _global_tracker = MetricsTracker()


# Utility functions: format output
def format_duration(seconds: float) -> str:
    """Format duration into a human-readable string"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def format_tokens(tokens: int) -> str:
    """Format token count into a human-readable string"""
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1000000:
        return f"{tokens/1000:.1f}K"
    else:
        return f"{tokens/1000000:.2f}M"