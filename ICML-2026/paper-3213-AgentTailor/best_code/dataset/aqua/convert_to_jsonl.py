
import shutil
from pathlib import Path

aqua_dir = Path(__file__).parent

for json_file in ["test.json", "dev.json", "test.jsonl"]:
    json_path = aqua_dir / json_file
    jsonl_path = aqua_dir / json_file.replace(".json", ".jsonl")

    if jsonl_path.exists():
        print(f"✅ {jsonl_path.name} already exists, skipping")
        continue

    if json_path.exists():
        print(f"📋 Copying {json_file} -> {jsonl_path.name}...")
        shutil.copy(json_path, jsonl_path)
        print(f"✅ Created {jsonl_path.name}")
    else:
        print(f"⚠️  {json_file} does not exist")

print("\n✅ Conversion finished!")























































