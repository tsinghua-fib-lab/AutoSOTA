from collections import namedtuple, defaultdict
import time
from abc import ABC, abstractmethod
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait,
    FIRST_COMPLETED,
)
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    TypeAlias,
    TYPE_CHECKING,
    Hashable,
)

from openai import RateLimitError
from tqdm import tqdm

if TYPE_CHECKING:
    from src.bias_pipeline.data_types.data_types import Thread

from src.configs import ModelConfig
from src.utils.cost_tracker import get_global_cost_tracker

T = TypeVar("T")
Y = TypeVar("Y")

RetryState = namedtuple("RetryState", ["item", "key", "retries"])

APIPrompt: TypeAlias = list[dict[str, str]]


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def predict(self, input: "Thread", **kwargs) -> str:
        pass

    @abstractmethod
    def predict_multi(self, inputs: List["Thread"], **kwargs) -> Iterator[Tuple["Thread", str]]:
        pass

    @abstractmethod
    def predict_string(self, input: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        pass

    @abstractmethod
    def _predict_call(self, input: List[Dict[str, str]]) -> str:
        pass


class APIModel(BaseModel):
    """Base class for API-based models"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def _predict_call(self, input: List[Dict[str, str]]) -> str:
        pass

    def predict(self, input: "Thread", **kwargs) -> str:
        messages, _, _ = input.to_chat(model=self.config)

        # Convert messages to string for cost tracking
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        guess = self._predict_call(messages)

        # Track costs
        cost_tracker = get_global_cost_tracker()
        cost_tracker.add_cost(
            input_text=input_text,
            output_text=guess,
            model_name=self.config.name,
            provider=self.config.provider,
        )

        return guess

    def predict_string(self, input: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        input_list = [
            {
                "role": "system",
                "content": "You are an helpful assistant."
                if system_prompt is None
                else system_prompt,
            },
            {"role": "user", "content": input},
        ]

        # Convert messages to string for cost tracking
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in input_list])

        guess = self._predict_call(input_list, **kwargs)

        # Track costs
        if not "local" in self.config.provider:
            cost_tracker = get_global_cost_tracker()
            cost_tracker.add_cost(
                input_text=input_text,
                output_text=guess,
                model_name=self.config.name,
                provider=self.config.provider,
            )

        return guess

    def predict_multi(self, inputs: List["Thread"], **kwargs) -> Iterator[Tuple["Thread", str]]:
        max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 4
        base_timeout = kwargs["timeout"] if "timeout" in kwargs else 120

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(inputs)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(inputs):
                # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
                results = executor.map(
                    lambda id: (id, inputs[id], self.predict(inputs[id])),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Profiles",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield (orig, answer)
                        # answered_prompts.append()
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except RateLimitError as r:
                    print(f"Rate_limit {r}")
                    time.sleep(30)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(500, timeout)
                retry_ctr += 1


def run_parallel(
    func: Callable[[T], Y],
    inputs: List[T],
    max_workers: int = 4,
    base_timeout: int = 120,
    max_retries: int = 100,
    desc: str = "Instances",
) -> Iterator[Tuple[T, Y]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        ids_to_do = list(range(len(inputs)))
        retry_ctr = 0
        timeout = base_timeout * len(inputs)

        while len(ids_to_do) > 0 and retry_ctr <= min(len(inputs), max_retries):
            # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
            results = executor.map(
                lambda id: (id, inputs[id], func(inputs[id])),
                ids_to_do,
                timeout=timeout,
            )
            try:
                for res in tqdm(
                    results,
                    total=len(ids_to_do),
                    desc=desc,
                    position=1,
                    leave=False,
                ):
                    id, orig, answer = res
                    yield (orig, answer)
                    # answered_prompts.append()
                    ids_to_do.remove(id)
            except FuturesTimeoutError:
                print(f"Timeout: {len(ids_to_do)} prompts remaining")
            except RateLimitError as r:
                print(f"Rate_limit {r}")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(10)
                continue

            if len(ids_to_do) == 0:
                break

            time.sleep(2 * retry_ctr)
            timeout *= 2
            timeout = min(500, timeout)
            retry_ctr += 1


def run_parallel_advanced(
    func: Callable[[T], Y],
    inputs: List[T],
    *,
    group_key_fn: Callable[[T], Hashable],
    group_max_workers: Optional[Dict[Hashable, int]] = None,
    group_max_workers_fn: Optional[Callable[[T], Dict[Hashable, int]]] = None,
    base_timeout: int = 120,
    max_retries: int = 100,
    desc: str = "Instances",
) -> Iterator[Tuple[T, Y]]:
    """
    Like run_parallel, but each *group* (e.g. provider / model) gets its own concurrency limit.

    Parameters
    ----------
    func : Callable[[T], Y]
        The function to execute.
    inputs : list[T]
        All call-arguments you want to process.
    group_key_fn : Callable[[T], Hashable]
        Given an input, returns its grouping key (e.g. provider name).
    group_max_workers : dict[key, int]
        How many parallel threads each group may use.
        Keys not present default to 4 workers.
    group_max_workers_fn : Callable[[T], int]
        If provided, this function is called for each input to determine the number of workers for that
        group. This overrides group_max_workers.
    base_timeout, max_retries, desc
        Same semantics you already had.

    Yields
    ------
    (orig_input, func(orig_input))  – in the order results arrive.
    """

    assert group_max_workers or group_max_workers_fn, (
        "You must provide either group_max_workers or group_max_workers_fn to run_parallel_advanced"
    )

    # 1) bucket inputs by group
    grouped: Dict[Hashable, List[Tuple[int, T]]] = defaultdict(list)
    for idx, item in enumerate(inputs):
        grouped[group_key_fn(item)].append((idx, item))

    if group_max_workers_fn:
        group_max_workers = group_max_workers_fn(inputs)

    # 2) spin up one executor per bucket
    executors: Dict[Hashable, ThreadPoolExecutor] = {
        key: ThreadPoolExecutor(max_workers=group_max_workers.get(key, 4)) for key in grouped
    }

    # 3) submit all jobs & remember their bookkeeping in a dict of futures
    futures = {}
    for key, items in grouped.items():
        for idx, item in items:
            fut = executors[key].submit(func, item)
            futures[fut] = RetryState(item=item, key=key, retries=0)

    # 4) harvest results as they complete, with retry / back-off
    with tqdm(total=len(inputs), desc=desc, position=1, leave=False) as pbar:
        while futures:
            done, _ = wait(futures.keys(), timeout=1, return_when=FIRST_COMPLETED)

            for fut in list(done):
                state = futures.pop(fut)
                timeout = base_timeout * (2**state.retries)
                timeout = min(120, timeout)

                try:
                    result = fut.result(timeout=timeout)
                    yield state.item, result
                    pbar.update()

                except FuturesTimeoutError:
                    # re-schedule if we can still retry
                    if state.retries < max_retries:
                        time.sleep(2 * state.retries)
                        new_fut = executors[state.key].submit(func, state.item)
                        futures[new_fut] = state._replace(retries=state.retries + 1)
                    else:
                        print(f"Timeout – giving up on: {state.item}")

                except RateLimitError as r:
                    print(f"Rate-limit ({r}). Backing off 30 s.")
                    time.sleep(30)
                    new_fut = executors[state.key].submit(func, state.item)
                    futures[new_fut] = state  # same retry counter

                except Exception as e:
                    print(f"Exception {e} – will retry.")
                    if state.retries < max_retries:
                        time.sleep(10)
                        new_fut = executors[state.key].submit(func, state.item)
                        futures[new_fut] = state._replace(retries=state.retries + 1)
                    else:
                        print(f"Giving up on: {state.item}")

    for ex in executors.values():
        ex.shutdown(wait=True)
