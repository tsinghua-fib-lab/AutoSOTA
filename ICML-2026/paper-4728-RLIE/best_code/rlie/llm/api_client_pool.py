"""Thread-safe OpenAI-compatible client pool."""
import threading
from typing import List

from openai import OpenAI, AsyncOpenAI


class APIClientPool:
    """Thread-local synchronous clients plus round-robin async clients."""

    def __init__(self, base_url: str, api_keys: List[str]):
        if not api_keys:
            raise ValueError("APIClientPool requires at least one api key")

        self._base_url = base_url
        self._api_keys = api_keys
        self._lock = threading.Lock()
        self._idx = 0

        # 使用线程本地存储，每个线程有自己的客户端
        self._thread_local = threading.local()

        # 异步客户端保持原有逻辑（因为异步模型下不存在线程竞争）
        self._async_clients = [
            AsyncOpenAI(base_url=base_url, api_key=key) for key in api_keys
        ]
        self._lock_async = threading.Lock()
        self._async_idx = 0

    def get_sync_client(self) -> OpenAI:
        """
        获取线程本地的同步客户端。

        每个线程第一次调用时会创建专属的客户端实例，
        后续调用会复用该实例。
        """
        # 检查当前线程是否已有客户端
        if not hasattr(self._thread_local, 'client'):
            with self._lock:
                # 为此线程分配一个 key（round-robin）
                key_idx = self._idx % len(self._api_keys)
                self._idx += 1

            # 创建线程专属的客户端
            self._thread_local.client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_keys[key_idx]
            )
            self._thread_local.key_idx = key_idx

        return self._thread_local.client

    def get_sync_client_for_retry(self, attempt: int) -> OpenAI:
        """
        为重试操作获取同步客户端，基于重试次数轮换 API key。

        Args:
            attempt: 当前重试次数（从 1 开始）

        Returns:
            OpenAI 客户端实例

        注意：每次重试都会尝试使用不同的 API key，避免持续失败的 key 阻塞请求。
        """
        key_idx = (attempt - 1) % len(self._api_keys)
        return OpenAI(
            base_url=self._base_url,
            api_key=self._api_keys[key_idx]
        )

    def get_async_client(self) -> AsyncOpenAI:
        """
        获取异步客户端（保持原有逻辑）。

        异步客户端在 asyncio 环境下使用，不存在线程竞争问题。
        """
        with self._lock_async:
            client = self._async_clients[self._async_idx % len(self._async_clients)]
            self._async_idx += 1
            return client

    def get_async_client_for_retry(self, attempt: int) -> AsyncOpenAI:
        """
        为重试操作获取异步客户端，基于重试次数轮换 API key。

        Args:
            attempt: 当前重试次数（从 1 开始）

        Returns:
            AsyncOpenAI 客户端实例

        注意：每次重试都会尝试使用不同的 API key，避免持续失败的 key 阻塞请求。
        """
        key_idx = (attempt - 1) % len(self._api_keys)
        return self._async_clients[key_idx]

    def get_stats(self):
        """获取统计信息（用于调试）"""
        return {
            'num_api_keys': len(self._api_keys),
            'num_async_clients': len(self._async_clients),
            'total_sync_allocations': self._idx,
            'total_async_allocations': self._async_idx,
        }


# 向后兼容：如果原代码使用了缓存机制
class APIClientPoolWithCache(APIClientPool):
    """
    带缓存的版本（保持与原版本的兼容性）

    注意：缓存在修复版中意义不大，因为每个线程都会创建自己的客户端。
    这里保留缓存主要是为了向后兼容。
    """

    _cache = {}
    _global_lock = threading.Lock()

    def __init__(self, base_url: str, api_keys: List[str]):
        normalized_keys = tuple(sorted(api_keys))
        cache_key = (base_url, normalized_keys)

        with APIClientPoolWithCache._global_lock:
            if cache_key in APIClientPoolWithCache._cache:
                # 从缓存获取已有的池
                pool = APIClientPoolWithCache._cache[cache_key]
                # 复制引用
                self._base_url = pool._base_url
                self._api_keys = pool._api_keys
                self._lock = pool._lock
                self._idx = pool._idx
                self._thread_local = pool._thread_local
                self._async_clients = pool._async_clients
                self._lock_async = pool._lock_async
                self._async_idx = pool._async_idx
                return

            # 调用父类初始化
            super().__init__(base_url, api_keys)
            # 保存到缓存
            APIClientPoolWithCache._cache[cache_key] = self


if __name__ == "__main__":
    # 简单测试
    print("测试修复版 APIClientPool")
    print("=" * 80)

    pool = APIClientPool(
        base_url="https://aihubmix.com/v1",
        api_keys=["key1", "key2", "key3"]
    )

    # 模拟多个线程获取客户端
    import threading

    client_ids = []
    lock = threading.Lock()

    def get_client(i):
        client = pool.get_sync_client()
        with lock:
            client_ids.append((i, id(client), threading.current_thread().name))

    threads = [threading.Thread(target=get_client, args=(i,), name=f"Thread-{i}") for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n创建了 {len(set(cid for _, cid, _ in client_ids))} 个不同的客户端实例")
    print("\n客户端分配情况:")
    for i, cid, tname in client_ids:
        print(f"  {tname}: 客户端 {cid}")

    print("\n✓ 每个线程应该有自己的客户端实例")
    print(f"✓ 统计信息: {pool.get_stats()}")
