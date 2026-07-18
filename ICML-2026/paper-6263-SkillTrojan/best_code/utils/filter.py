#!/usr/bin/env python3
"""
Filter EHRSQL records: keep only those whose SQL queries can execute successfully.
Multi-core parallel version.
"""
import argparse
import json
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSON records from file (supports both JSON array and JSONL)."""
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

    # Try JSONL format
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def run_sql_query(db_path: Path, query: str) -> Tuple[bool, List[Tuple[Any, ...]], str]:
    """
    Execute SQL query and return (success, results, error_message).
    
    Returns:
        success: True if query executed without error AND returned non-empty results
        results: Query results (empty list if failed)
        error_message: Error message if failed (empty string if success)
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            # ✅ 新增：检查结果是否为空
            if not results:
                return False, [], "Query returned empty results"
            
            # ✅ 新增（可选）：检查结果是否全是空字符串/None
            if all(all(v == '' or v is None for v in row) for row in results):
                return False, results, "Query returned only empty values"
            
            return True, results, ""
    except Exception as exc:
        return False, [], str(exc)


def process_single_record(record: Dict[str, Any], data_dir_str: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Process a single record (designed for parallel execution).
    
    Returns:
        (is_valid, record, error_message)
    """
    data_dir = Path(data_dir_str)
    item_id = record.get("id", "unknown")
    db_id = record.get("db_id", "")
    query = record.get("query", "")
    
    if not db_id or not query:
        return False, record, f"{item_id}: missing db_id or query"
    
    db_path = data_dir / f"{db_id}.db"
    if not db_path.exists():
        return False, record, f"{item_id}: DB not found: {db_path}"
    
    success, results, error = run_sql_query(db_path, query)
    
    if success:
        return True, record, f"{item_id}: {len(results)} rows"
    else:
        return False, record, f"{item_id}: {error}"


def filter_executable_records_parallel(
    records: List[Dict[str, Any]], 
    data_dir: Path,
    max_workers: int,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Filter records to keep only those with executable SQL queries (parallel version)."""
    valid_records = []
    data_dir_str = str(data_dir)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Filtering records (workers={max_workers})...", 
            total=len(records)
        )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_record, record, data_dir_str): record
                for record in records
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                is_valid, record, msg = future.result()
                
                if is_valid:
                    valid_records.append(record)
                    if verbose:
                        console.print(f"[green]✓[/green] {msg}")
                else:
                    if verbose:
                        console.print(f"[red]✗[/red] {msg}")
                
                progress.advance(task)
    
    return valid_records


def main():
    parser = argparse.ArgumentParser(
        description="Filter EHRSQL records to keep only executable queries (parallel)"
    )
    parser.add_argument(
        "--data_dir",
        default="./data/ehrsql",
        help="Directory containing .db files"
    )
    parser.add_argument(
        "--input",
        default="data/ehrsql/eicu_valid.json",
        help="Input JSON file (e.g., eicu_train.json)"
    )
    parser.add_argument(
        "--output",
        default="data/ehrsql/eicu_valid_filtered.json",
        help="Output JSON file (default: <input>_filtered.json)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=f"Number of parallel workers (default: {os.cpu_count() or 1})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for each record"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    input_path = Path(args.input)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_filtered.json"
    
    console.print(f"[cyan]Loading records from:[/cyan] {input_path}")
    records = load_json_records(input_path)
    console.print(f"[cyan]Total records loaded:[/cyan] {len(records)}")
    
    console.print(f"[cyan]Database directory:[/cyan] {data_dir}")
    console.print(f"[cyan]Parallel workers:[/cyan] {args.workers}")
    
    valid_records = filter_executable_records_parallel(
        records, 
        data_dir, 
        args.workers,
        args.verbose
    )
    
    console.print(f"\n[green]Valid records:[/green] {len(valid_records)}/{len(records)}")
    console.print(f"[yellow]Filtered out:[/yellow] {len(records) - len(valid_records)}")
    
    # Save filtered records
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(valid_records, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[green]Filtered records saved to:[/green] {output_path}")
    
    # Print statistics
    console.print("\n[cyan]Statistics:[/cyan]")
    console.print(f"  Input file:     {input_path}")
    console.print(f"  Output file:    {output_path}")
    console.print(f"  Original count: {len(records)}")
    console.print(f"  Valid count:    {len(valid_records)}")
    console.print(f"  Retention rate: {len(valid_records)/len(records)*100:.1f}%")
    console.print(f"  Workers used:   {args.workers}")


if __name__ == "__main__":
    main()
