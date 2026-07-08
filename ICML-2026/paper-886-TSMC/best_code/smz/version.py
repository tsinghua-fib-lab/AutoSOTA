"""Module Versioning. This is directly updated to `experiment.py`."""


def _version_as_tuple(version_str: str) -> tuple[int, ...]:
    return tuple(int(i) for i in version_str.split(".") if i.isdigit())


__version__: str = '0.0.0'
__version_info__: tuple[int, ...] = _version_as_tuple(__version__)
