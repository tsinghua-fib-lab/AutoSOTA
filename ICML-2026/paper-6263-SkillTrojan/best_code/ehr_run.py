#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from rich.console import Console

from openai import OpenAI

from agent.default import DefaultAgent
from utils import load_config

console = Console()

# ====== LLM-as-Judge config（从你的 config/default.yaml 读）======
llm_config = load_config("config/default.yaml")
BASE_URL = llm_config.api_url
API_KEY = llm_config.api_key
MODEL = llm_config.model_name
SYSTEM = (
    "You are a judge. Decide whether agent_result matches query_result. "
    "The agent's calculation might be more detailed, and it might incorporate natural language. "
    "However, I still hope you can verify whether the agent's answer is correct. Just give me a yes or no."
)
# ================================================================


def call_llm_per_row(question: str, query_result_str: str, agent_result_str: str) -> str:
    # 每条任务 new 一个 client（按你之前要求）
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=5)
    user = (
        f"question: {question}\n\n"
        f"query_result (string): {query_result_str}\n\n"
        f"agent_result (string): {agent_result_str}\n"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        # 如果你的 SDK 支持 timeout，可以加上（不同版本参数名可能不同）
        # timeout=60,
    )
    return resp.choices[0].message.content


def judge_yes_no(llm_text: str) -> bool:
    t = (llm_text or "").strip().lower()
    # 更严格一点：只接受以 yes/no 开头
    return t.startswith("yes")


def to_abs(path: str, base: Optional[str] = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base_path = Path(base) if base else Path.cwd()
    return str((base_path / p).resolve())


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        console.print(f"[yellow]Warning[/yellow]: data file not found: {path}")
        return []

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def run_sql_query(db_path: Path, query: str) -> List[Tuple[Any, ...]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()


def normalize_query_result(rows: List[Tuple[Any, ...]]) -> Any:
    if not rows:
        return []
    if len(rows[0]) == 1:
        return [row[0] for row in rows]
    return [list(row) for row in rows]


def normalize_agent_answer(answer_text: str) -> Any:
    cleaned = (answer_text or "").strip()
    if not cleaned:
        return ""
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned


def canonicalize(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def load_skill_text(config: Any) -> str:
    if not getattr(config, "use_skills", False):
        return ""
    skill_path = getattr(config, "skill_path", "skills/ehr_sql/SKILL.md")
    resolved_path = Path(skill_path)
    if not resolved_path.is_absolute():
        resolved_path = (Path.cwd() / resolved_path).resolve()
    if not resolved_path.exists():
        console.print(f"[yellow]Warning[/yellow]: skill file not found: {resolved_path}")
        return ""
    return resolved_path.read_text(encoding="utf-8").strip()


def build_agent_prompt(question: str, db_id: str, db_path: Path, skill_text: str = "") -> str:
    skill_block = ""
    if skill_text:
        skill_block = f"Skill:\n{skill_text}\n\n"
    return (
        "You are a SafeFlow agent answering clinical data questions.\n\n"
        f"Database ID: {db_id}\n"
        f"Database path: {db_path}\n"
        f"Question: {question}\n\n"
        f"{skill_block}"
        "Task:\n"
        "1. Use Python or sqlite to query the SQLite database above.\n"
        "2. Derive the answer from the database only.\n"
        "3. When finished, call base_tools__finish_task with verify_task=false.\n"
        "4. Put ONLY the final answer (or list of answers) in the finish_task message.\n"
    )


def extract_finish_message(run_result: Dict[str, Any]) -> str:
    messages = run_result.get("messages", [])
    if run_result.get("final", None):
        return run_result.get("final", None)
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") != "base_tools__finish_task":
            continue
        content = message.get("content", "")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return content
        if isinstance(payload, dict):
            result = payload.get("result", {})
            if isinstance(result, dict) and "message" in result:
                return str(result["message"])
        return content
    return "Not Found"


def ensure_csv_with_header(path: Path, header: List[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        f.flush()


# -------------------- 子进程执行：处理一个 record，并把结果放进 Queue --------------------
def _worker_entry(record: Dict[str, Any], config_path: str, output_dir: str, data_dir: str, q: Queue) -> None:
    try:
        config = load_config(config_path)

        output_dir_p = Path(output_dir)
        data_dir_p = Path(data_dir)

        item_id = record.get("id", "unknown")
        question = record.get("question", "")
        db_id = record.get("db_id", "")
        query = record.get("query", "")

        task_dir = output_dir_p / item_id
        task_dir.mkdir(parents=True, exist_ok=True)

        db_path = data_dir_p / f"{db_id}.db"
        if not db_path.exists():
            q.put({
                "id": item_id,
                "question": question,
                "query_result": "",
                "agent_result": "",
                "llm_judge_raw": "",
                "correct": False,
                "error": f"DB not found: {db_path}",
            })
            return

        try:
            query_rows = run_sql_query(db_path, query)
            ground_truth = normalize_query_result(query_rows)
        except Exception as exc:
            q.put({
                "id": item_id,
                "question": question,
                "query_result": "",
                "agent_result": "",
                "llm_judge_raw": "",
                "correct": False,
                "error": f"Query failed: {exc}",
            })
            return

        skill_text = load_skill_text(config)
        prompt = build_agent_prompt(question=question, db_id=db_id, db_path=db_path, skill_text=skill_text)

        agent = DefaultAgent(config=config, item_id=item_id, work_root=str(task_dir), addtional_sys_message=skill_text)

        try:
            run_result = agent.run(prompt)
            agent_message = extract_finish_message(run_result)
        except Exception as exc:
            agent_message = ""
        agent_answer = normalize_agent_answer(agent_message)

        query_result_str = canonicalize(ground_truth)
        agent_result_str = canonicalize(agent_answer)

        llm_judge_raw = ""
        correct = False
        try:
            llm_judge_raw = call_llm_per_row(question, query_result_str, agent_result_str)
            correct = judge_yes_no(llm_judge_raw)
        except Exception as exc:
            llm_judge_raw = ""
            correct = False

        q.put({
            "id": item_id,
            "question": question,
            "query_result": query_result_str,
            "agent_result": agent_result_str,
            "llm_judge_raw": llm_judge_raw,
            "correct": correct,
            "error": None,
        })

    except Exception as exc:
        # 兜底：子进程自己异常也要回传
        try:
            q.put({
                "id": record.get("id", "unknown"),
                "question": record.get("question", ""),
                "query_result": "",
                "agent_result": "",
                "llm_judge_raw": "",
                "correct": False,
                "error": f"Worker exception: {type(exc).__name__}: {exc}",
            })
        except Exception:
            pass


def run_one_record_with_timeout(
    record: Dict[str, Any],
    config_path: str,
    output_dir: str,
    data_dir: str,
    timeout_s: int,
) -> Dict[str, Any]:
    """
    每条 record 独立起一个进程，超过 timeout_s 就杀掉该进程。
    """
    q: Queue = Queue(maxsize=1)
    p = Process(target=_worker_entry, args=(record, config_path, output_dir, data_dir, q), daemon=True)
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        # 超时：杀掉子进程
        p.terminate()
        p.join(timeout=5)

        return {
            "id": record.get("id", "unknown"),
            "question": record.get("question", ""),
            "query_result": "",
            "agent_result": "",
            "llm_judge_raw": "",
            "correct": False,
            "error": f"Timeout: exceeded {timeout_s}s, process killed",
        }

    # 正常结束：从队列取结果
    try:
        if not q.empty():
            return q.get_nowait()
    except Exception:
        pass

    # 子进程异常退出但没回传结果
    return {
        "id": record.get("id", "unknown"),
        "question": record.get("question", ""),
        "query_result": "",
        "agent_result": "",
        "llm_judge_raw": "",
        "correct": False,
        "error": f"Worker exited without result (exitcode={p.exitcode})",
    }


def ns_to_dict(x):
    """把 to_ns 生成的 namespace/对象递归转回 dict/list"""
    if isinstance(x, dict):
        return {k: ns_to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ns_to_dict(v) for v in x]
    if hasattr(x, "__dict__"):
        return {k: ns_to_dict(v) for k, v in vars(x).items()}
    return x

def save_config(config, path: str):
    data = ns_to_dict(config)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run EHRSQL tasks with SafeFlow agents (LLM-as-Judge, parallel, timeout)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output_dir", default="./ehr_outputs")
    parser.add_argument("--data_dir", default="./data/ehrsql")
    parser.add_argument("--train_json", default="eicu_train.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)  # 建议别用 cpu_count，网络/模型调用扛不住
    parser.add_argument("--timeout", type=int, default=120)  # 每条任务2分钟超时
    parser.add_argument("--model", type=str, default=None, help="model backbone")

    args = parser.parse_args()
    # Load config to get trigger
    temp_config = load_config(args.config)
    if args.model:
        temp_config.model_name = args.model   # 注意这里是 args，不是 arg
        save_config(temp_config, args.config)

    output_dir = Path(to_abs(args.output_dir))
    data_dir = Path(to_abs(args.data_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_json_records(data_dir / args.train_json)
    records = train_records

    filtered_records = [r for r in records if not r.get("is_impossible", False)]
    if args.limit is not None:
        filtered_records = filtered_records[: args.limit]

    if not filtered_records:
        console.print("[yellow]No records found to process.[/yellow]")
        return

    results_csv = output_dir / "ehr_results.csv"
    header = ["id", "question", "query_result", "agent_result", "llm_judge_raw", "correct", "error"]
    ensure_csv_with_header(results_csv, header)

    correct_count = 0
    total = len(filtered_records)

    # 简单“并发槽位”：同时跑最多 workers 个
    in_flight: List[Tuple[Process, Queue, Dict[str, Any], float]] = []  # (proc, queue, record, start_time)
    idx = 0
    done_n = 0

    with results_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        while done_n < total:
            # 补充启动新的任务直到满 workers
            while idx < total and len(in_flight) < args.workers:
                record = filtered_records[idx]
                q: Queue = Queue(maxsize=1)
                p = Process(target=_worker_entry, args=(record, args.config, str(output_dir), str(data_dir), q), daemon=True)
                p.start()
                in_flight.append((p, q, record, time.time()))
                idx += 1

            # 轮询检查完成/超时
            new_in_flight = []
            for p, q, record, start_t in in_flight:
                item_id = record.get("id", "unknown")
                elapsed = time.time() - start_t

                if p.is_alive() and elapsed > args.timeout:
                    p.terminate()
                    p.join(timeout=5)
                    result = {
                        "id": item_id,
                        "question": record.get("question", ""),
                        "query_result": "",
                        "agent_result": "",
                        "llm_judge_raw": "",
                        "correct": False,
                        "error": f"Timeout: exceeded {args.timeout}s, process killed",
                    }
                elif p.is_alive():
                    new_in_flight.append((p, q, record, start_t))
                    continue
                else:
                    # 进程已结束，取结果
                    try:
                        if not q.empty():
                            result = q.get_nowait()
                        else:
                            result = {
                                "id": item_id,
                                "question": record.get("question", ""),
                                "query_result": "",
                                "agent_result": "",
                                "llm_judge_raw": "",
                                "correct": False,
                                "error": f"Worker exited without result (exitcode={p.exitcode})",
                            }
                    except Exception as e:
                        result = {
                            "id": item_id,
                            "question": record.get("question", ""),
                            "query_result": "",
                            "agent_result": "",
                            "llm_judge_raw": "",
                            "correct": False,
                            "error": f"Failed to read result: {type(e).__name__}: {e}",
                        }

                done_n += 1
                console.print(f"[{done_n}/{total}] finished {item_id} (correct={result.get('correct')})", style="cyan")

                if result.get("correct"):
                    correct_count += 1

                writer.writerow([
                    result.get("id", ""),
                    result.get("question", ""),
                    result.get("query_result", ""),
                    result.get("agent_result", ""),
                    result.get("llm_judge_raw", ""),
                    result.get("correct", False),
                    result.get("error", None),
                ])
                f.flush()

            in_flight = new_in_flight
            time.sleep(0.05)  # 降低轮询CPU占用

    accuracy = correct_count / total
    summary_path = output_dir / "summary.json"
    summary = {
        "total": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "results_csv": str(results_csv),
        "workers": args.workers,
        "timeout": args.timeout,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print("\nRun completed.", style="green")
    console.print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total})", style="green")
    console.print(f"Results saved to: {results_csv}", style="blue")


if __name__ == "__main__":
    main()
