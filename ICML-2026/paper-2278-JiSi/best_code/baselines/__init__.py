"""JiSi benchmark-result loading utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_loader import BaselineDataLoader
    from .schema import AggregatedStats, BaselineMetadata, BaselineRecord

__version__ = '1.0.0'

__all__ = [
    'BaselineDataLoader',
    'BaselineRecord',
    'AggregatedStats',
    'BaselineMetadata',
]


def __getattr__(name):
    if name == 'BaselineDataLoader':
        from .data_loader import BaselineDataLoader
        return BaselineDataLoader
    if name in {'BaselineRecord', 'AggregatedStats', 'BaselineMetadata'}:
        from .schema import AggregatedStats, BaselineMetadata, BaselineRecord
        return {
            'BaselineRecord': BaselineRecord,
            'AggregatedStats': AggregatedStats,
            'BaselineMetadata': BaselineMetadata,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
