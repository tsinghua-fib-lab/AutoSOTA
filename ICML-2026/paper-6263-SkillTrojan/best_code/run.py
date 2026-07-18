import os
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from traj import TraceTrack
from utils import load_config
from agent.default import DefaultAgent
from agent.context_manager import ContextManagerAgent

console = Console()
from datasets import load_dataset

def main(args, prompt, item_id):
    item_id = item_id

    # Create context manager agent first
    context_manager = ContextManagerAgent(config=args, item_id=item_id)

    # Create default agent with context manager reference
    safe_agent = DefaultAgent(
        config=args,
        agent_name="default",
        item_id=item_id,
        context_manager=context_manager
    )

    console.print("🤖 Agents initialized:", style="blue")
    console.print(f"DefaultAgent: {safe_agent.agent_name} (item_id: {item_id})")
    console.print(f"ContextManager: {context_manager.agent_name} (item_id: {item_id})")
    console.print()

    result = safe_agent.run(user_prompt=prompt)

    # Save context state after execution
    context_save_result = context_manager.save_context_state()
    if context_save_result["success"]:
        console.print(f"💾 Context saved to: {context_save_result['filepath']}", style="green")

    return result


if __name__ == "__main__":
    # 1. Start 👏
    console.print(Panel.fit("🚀 Safeflow Unified Evaluation", style="bold blue"))
    args = load_config(path="config/default.yaml")
    console.print("✅ Configuration loaded \n", style="green")
    console.print(args)

    # 2.Run
    dataset = load_dataset("parquet", data_files={"test": args.parquet_path})
    for i, item in enumerate(dataset["test"]):
        item_id = item["instance_id"]
        item.pop("test_patch", None)
        prompt = json.dumps(item) 
        main(args=args, prompt=prompt, item_id=item_id)
        break

