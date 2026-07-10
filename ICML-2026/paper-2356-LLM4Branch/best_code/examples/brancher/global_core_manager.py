# in global_core_manager.py
import os
import time
import logging
import psutil
import random

logger = logging.getLogger(__name__)

class GlobalCoreManager:
    def __init__(self, used_cores, lock_dir):
        self.total_cores = used_cores
        self.lock_dir = lock_dir
    
    def get_core(self):
        """
        通过原子性的文件创建来获取一个核心的锁。
        """
        all_cores = list(self.total_cores)
        random.shuffle(all_cores)

        while True:
            for core_id in all_cores:
                lock_path = os.path.join(self.lock_dir, f"lock_{core_id}")
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL)
                    os.close(fd)
                    # logging.debug(f"Process {os.getpid()} acquired lock for CPU core {core_id} via file creation.")
                    return core_id
                except FileExistsError:
                    # 文件已存在，说明这个核心被别人占用了
                    continue
            
            logging.debug(f"Process {os.getpid()} found no available cores, waiting...")
            time.sleep(random.uniform(1, 2))

    def release_core(self, core_id):
        """
        通过删除文件来释放锁。
        """
        lock_path = os.path.join(self.lock_dir, f"lock_{core_id}")
        try:
            os.remove(lock_path)
            logging.debug(f"Process {os.getpid()} released lock for CPU core {core_id} via file removal.")
        except FileNotFoundError:
            # 可能文件已经被别人释放了，或者从未创建，这没关系
            logging.warning(f"Attempted to release lock for core {core_id}, but lock file did not exist.")
        except Exception as e:
            logging.error(f"Failed to release lock for core {core_id}: {e}")