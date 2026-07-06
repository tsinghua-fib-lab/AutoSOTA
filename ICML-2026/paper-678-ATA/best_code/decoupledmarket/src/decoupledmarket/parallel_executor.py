"""Parallel executor utilities for running agent steps concurrently."""

import multiprocessing
import os
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, List, Optional

from decoupledmarket.performance_monitor import get_monitor


class ThreadSafeDatabase:
    """Thread-safe sqlite wrapper that mimics Database_operate API."""

    def __init__(self, db_name: str):
        self._db_name = db_name
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False

    def _get_connection(self):
        """Get a thread-local database connection and cursor."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(f"{self._db_name}.db", check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.cur = self._local.conn.cursor()
        return self._local.conn, self._local.cur

    def init_database(self):
        """Initialize database schema once."""
        with self._lock:
            if self._initialized:
                return

            conn, cur = self._get_connection()
            tables = [
                (
                    "Create Table active_orders (timestamp Integer NOT NULL, virtual_date Integer, "
                    "weekday INTEGER, iteration INTEGER,"
                    "stock_id INTEGER, person_id INTEGER, type text check(type IN ('sell','buy')), "
                    "price Numeric, quantity INTEGER, "
                    "status text check (status IN ('active','closed','finished') ))"
                ),
                (
                    "Create Table stock (stock_id Integer NOT NULL, virtual_date Integer, "
                    "weekday INTEGER,"
                    "volume  Numeric, quantity INTEGER, last_price Numeric, begin_price Numeric,"
                    "highest_price Numeric, lowest_price Numeric )"
                ),
                (
                    "Create Table person (person_id Integer, virtual_date Integer, "
                    "cash Numeric, asset Numeric,"
                    "wealth Numeric, work_income Numeric,"
                    "capital_gain Numeric, daily_expense Numeric,"
                    "principle Text)"
                ),
                (
                    "Create Table account (person_id Integer, stock_id Integer, virtual_date Integer, "
                    "weekday INTEGER, quantity INTEGER,"
                    "cost_price Numeric, current_price Numeric, profit Numeric,"
                    "start_date INTEGER)"
                ),
                (
                    "Create Table memory (person_id Integer, virtual_date Integer, iteration INTEGER, "
                    "stock_operations Text, strategy Text, type Text check(type IN ('sell','buy','hold','reflect')), gossip Text, "
                    "analysis_for_stocks Text, analysis_for_strategy Text, stock_prices Text, market_change Text, financial_situation Text)"
                ),
                ("Create Table gossip (person_id Integer, virtual_date Integer, gossip Text)"),
                (
                    "Create Table agent (agent_id Integer, virtual_date Integer, "
                    "cash Numeric, asset Numeric,"
                    "wealth Numeric, work_income Numeric,"
                    "capital_gain Numeric, daily_expense Numeric,"
                    "principle Text)"
                ),
            ]

            for cmd in tables:
                try:
                    cur.execute(cmd)
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        print(f"Create table failed: {e}")

            conn.commit()
            self._initialized = True

    def execute_sql(self, cmd: str) -> bool:
        """Execute SQL command (thread-safe)."""
        try:
            conn, cur = self._get_connection()
            cur.execute(cmd)
            conn.commit()
            return True
        except Exception as e:
            print(f"Database ERROR: {cmd}")
            print(e)
            return False

    def execute_sql_params(self, cmd: str, params: tuple) -> bool:
        """Execute parameterized SQL command (thread-safe)."""
        try:
            conn, cur = self._get_connection()
            cur.execute(cmd, params)
            conn.commit()
            return True
        except Exception as e:
            print(f"Database ERROR (parametrized): {cmd}")
            print(e)
            return False

    def fetchall(self):
        """Fetch all rows from previous query."""
        _, cur = self._get_connection()
        return cur.fetchall()

    def close(self):
        """Close thread-local database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()

    @property
    def cur(self):
        """Expose cursor for compatibility with Database_operate."""
        _, cur = self._get_connection()
        return cur


class ParallelExecutor:
    """Base executor."""

    def __init__(self, max_workers: Optional[int] = None):
        default_workers = min(4, multiprocessing.cpu_count())
        env_workers = os.getenv("SIM_MAX_WORKERS")
        if env_workers and env_workers.isdigit():
            default_workers = max(1, int(env_workers))
        self.max_workers = max_workers or default_workers
        self.monitor = get_monitor()

    def execute_agents(self, agents: List[Any], agent_func: Callable, *args, **kwargs) -> List[Any]:
        """Execute agent operations; implemented in subclasses."""
        raise NotImplementedError


class ThreadExecutor(ParallelExecutor):
    """Thread-based executor."""

    def __init__(self, max_workers: Optional[int] = None):
        super().__init__(max_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def execute_agents(self, agents: List[Any], agent_func: Callable, *args, **kwargs) -> List[Any]:
        """Execute agents in a thread pool."""
        results: List[Any] = []
        futures = []

        for agent in agents:
            future = self.executor.submit(
                self._execute_agent_with_monitoring,
                agent,
                agent_func,
                *args,
                **kwargs,
            )
            futures.append((agent, future))

        for agent, future in futures:
            try:
                result = future.result(timeout=300)
                results.append(result)
            except Exception as e:
                print(f"Agent {agent.person_id} execution failed: {e}")
                results.append((agent, None))

        return results

    def _execute_agent_with_monitoring(self, agent, agent_func, *args, **kwargs):
        """Run a single agent function with timing."""
        agent_id = agent.person_id
        self.monitor.start_timer("agent_execution", agent_id)
        try:
            result = agent_func(agent, *args, **kwargs)
            return result
        finally:
            self.monitor.end_timer("agent_execution", agent_id)

    def shutdown(self):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=True)


class ProcessExecutor(ParallelExecutor):
    """Process-based executor."""

    def __init__(self, max_workers: Optional[int] = None):
        super().__init__(max_workers)
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def execute_agents(self, agents: List[Any], agent_func: Callable, *args, **kwargs) -> List[Any]:
        """Execute agents in a process pool."""
        raise NotImplementedError(
            "Process executor is not implemented: agent state serialization/reconstruction is missing."
        )

    @staticmethod
    def _execute_agent_process(agent_data, agent_func, *args, **kwargs):
        """Execute one agent task in subprocess (to be implemented)."""
        # Reconstructing full agent object is project-specific.
        # Keep as placeholder to preserve previous behavior.
        pass

    def shutdown(self):
        """Shutdown process pool."""
        self.executor.shutdown(wait=True)


class BatchExecutor(ParallelExecutor):
    """Batch executor: split agents into batches and run each batch in threads."""

    def __init__(self, batch_size: int = 20, max_workers: Optional[int] = None):
        super().__init__(max_workers)
        self.batch_size = batch_size
        self.thread_executor = ThreadExecutor(max_workers)

    def execute_agents(self, agents: List[Any], agent_func: Callable, *args, **kwargs) -> List[Any]:
        """Execute agents in batches."""
        all_results = []

        for i in range(0, len(agents), self.batch_size):
            batch = agents[i : i + self.batch_size]
            self.monitor.start_timer(f"batch_{i // self.batch_size}")

            batch_results = self.thread_executor.execute_agents(
                batch,
                agent_func,
                *args,
                **kwargs,
            )

            all_results.extend(batch_results)
            self.monitor.end_timer(f"batch_{i // self.batch_size}")

            if "batch_callback" in kwargs:
                kwargs["batch_callback"](i // self.batch_size)

        return all_results

    def shutdown(self):
        """Shutdown nested thread executor."""
        self.thread_executor.shutdown()


def create_executor(
    executor_type: str = "thread",
    max_workers: Optional[int] = None,
    batch_size: int = 20,
) -> ParallelExecutor:
    """Factory for executor instances."""
    if executor_type == "thread":
        return ThreadExecutor(max_workers)
    if executor_type == "process":
        return ProcessExecutor(max_workers)
    if executor_type == "batch":
        return BatchExecutor(batch_size, max_workers)
    raise ValueError(f"Unknown executor type: {executor_type}")
