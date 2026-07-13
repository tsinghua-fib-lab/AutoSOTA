"""并发配置和预算分配工具模块

提供统一的并发限制计算和多层并行预算分配功能。
"""
import logging
from typing import Dict, List, Tuple


LOGGER = logging.getLogger("rlie.concurrent_config")


def extract_api_keys(client_cfg: Dict) -> List[str]:
    """从配置中提取 API keys 列表

    Args:
        client_cfg: client 配置字典，包含 api_key 或 api_keys

    Returns:
        API keys 列表

    Raises:
        ValueError: 如果没有找到有效的 API keys
    """
    api_keys = client_cfg.get("api_keys") or client_cfg.get("api_key")

    if not api_keys:
        raise ValueError("Client config must contain 'api_key' or 'api_keys'")

    if isinstance(api_keys, str):
        return [api_keys]

    if isinstance(api_keys, (list, tuple)):
        keys = [str(k).strip() for k in api_keys if str(k).strip()]
        if not keys:
            raise ValueError("API keys list is empty after filtering")
        return keys

    raise ValueError(f"Invalid api_keys type: {type(api_keys)}")


def calculate_total_concurrent_limit(
    config: Dict,
    num_keys: int,
    retry_overhead: float = 0.15,
) -> int:
    """计算总并发限制（考虑重试开销）

    Args:
        config: 配置字典，应包含 max_concurrent_per_key
        num_keys: API key 数量
        retry_overhead: 重试开销预留比例（默认 0.15 即 15%）

    Returns:
        有效的总并发限制

    Raises:
        ValueError: 如果配置中缺少 max_concurrent_per_key
    """
    per_key = config.get("max_concurrent_per_key")
    if per_key is None:
        raise ValueError(
            "Config must contain 'max_concurrent_per_key' parameter. "
            "The old 'max_concurrent' parameter has been deprecated."
        )

    per_key = int(per_key)
    if per_key <= 0:
        raise ValueError(f"max_concurrent_per_key must be positive, got {per_key}")

    retry_overhead = float(retry_overhead)
    if not 0.0 <= retry_overhead < 1.0:
        raise ValueError(f"retry_overhead must be in [0, 1), got {retry_overhead}")

    # 总并发 = 每个key的并发 × key数量 × (1 - 重试开销)
    theoretical_limit = per_key * num_keys
    effective_limit = int(theoretical_limit * (1.0 - retry_overhead))

    LOGGER.info(
        "并发限制计算: %d keys × %d per_key = %d 理论上限, "
        "预留 %.0f%% 重试开销后 = %d 有效上限",
        num_keys,
        per_key,
        theoretical_limit,
        retry_overhead * 100,
        effective_limit,
    )

    return max(1, effective_limit)


def allocate_concurrent_budget(
    total_limit: int,
    outer_parallelism: int,
    inner_parallelism: int,
    min_inner: int = 5,
) -> Tuple[int, int]:
    """分配两层并行结构的并发预算

    Args:
        total_limit: 总并发限制
        outer_parallelism: 外层并行度（如 repeat 数量或 model 数量）
        inner_parallelism: 内层期望并行度（如 rule 数量或 sample 数量）
        min_inner: 内层最小保证并发度（默认 5）

    Returns:
        (outer_workers, inner_concurrent): 外层并行数和内层并发数的元组

    Examples:
        >>> allocate_concurrent_budget(85, 3, 5, min_inner=5)
        (3, 28)  # 3 个 repeat，每个 repeat 内 28 并发

        >>> allocate_concurrent_budget(28, 5, 20, min_inner=5)
        (5, 5)   # 5 个 rule，每个 rule 内 5 并发（受限于 min_inner）
    """
    total_limit = max(1, int(total_limit))
    outer_parallelism = max(1, int(outer_parallelism))
    inner_parallelism = max(1, int(inner_parallelism))
    min_inner = max(1, int(min_inner))

    # 情况1: 外层并行度为1，全部预算给内层
    if outer_parallelism == 1:
        inner_concurrent = min(total_limit, inner_parallelism)
        LOGGER.debug(
            "单外层分配: total=%d → inner=%d (期望=%d)",
            total_limit,
            inner_concurrent,
            inner_parallelism,
        )
        return 1, inner_concurrent

    # 情况2: 总预算不足以给每个外层分配 min_inner
    min_required = outer_parallelism * min_inner
    if total_limit < min_required:
        # 降低外层并行度
        outer_workers = max(1, total_limit // min_inner)
        inner_concurrent = min_inner
        LOGGER.warning(
            "预算不足: total=%d < %d×%d=%d, 降低外层并行度 %d→%d",
            total_limit,
            outer_parallelism,
            min_inner,
            min_required,
            outer_parallelism,
            outer_workers,
        )
        return outer_workers, inner_concurrent

    # 情况3: 正常分配
    # 策略: 保持外层全并行，平均分配给内层
    outer_workers = outer_parallelism
    inner_budget = total_limit // outer_parallelism
    inner_concurrent = max(min_inner, min(inner_budget, inner_parallelism))

    actual_total = outer_workers * inner_concurrent
    LOGGER.debug(
        "两层分配: total=%d → outer=%d × inner=%d = %d (内层期望=%d)",
        total_limit,
        outer_workers,
        inner_concurrent,
        actual_total,
        inner_parallelism,
    )

    return outer_workers, inner_concurrent
