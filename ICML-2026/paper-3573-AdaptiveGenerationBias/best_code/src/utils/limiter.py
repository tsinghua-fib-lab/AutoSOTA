from collections import deque
import time
import threading
from typing import Deque


class RateLimiter:
    def __init__(self, k: int, seconds: int):
        self.rate = k
        self.seconds = seconds
        self.timestamps: Deque[float] = deque()
        self.lock = threading.Lock()

    def record(self):
        self.lock.acquire()
        try:
            current_time = time.time()

            # Remove timestamps older than 1 minute (60 seconds)
            while self.timestamps and self.timestamps[0] < current_time - self.seconds:
                self.timestamps.popleft()

            # Check the number of timestamps (requests) within last minute
            if len(self.timestamps) < self.rate:
                self.timestamps.append(current_time)
                return True  # Proceed with the request
            else:
                return False  # Rate limit exceeded, don't proceed with the request

        finally:
            self.lock.release()
