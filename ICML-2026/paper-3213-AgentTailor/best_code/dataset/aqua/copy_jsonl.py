
import shutil
from pathlib import Path

aqua_dir = Path(__file__).parent

files = ["test.json", "dev.json", "test.jsonl"]

for json_file in files:
    json_path = aqua_dir / json_file
    jsonl_path = aqua_dir / json_file.replace(".json", ".jsonl")

    if jsonl_path.exists():
        print(f"✅ {jsonl_path.name} already exists")
        continue

    if json_path.exists():
        print(f"📋 Copying {json_file} -> {jsonl_path.name}...")
        shutil.copy2(json_path, jsonl_path)
        print("✅ Done")
    else:
        print(f"⚠️  {json_file} does not exist")

print("\n✅ All files processed!")























































