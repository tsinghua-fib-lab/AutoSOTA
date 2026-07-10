
import json
from pathlib import Path


def fix_jsonl():

    this_file_path = Path(__file__).parent

    files_to_convert = ["test.json", "dev.json", "test.jsonl"]

    for json_file in files_to_convert:
        json_path = this_file_path / json_file
        jsonl_path = this_file_path / json_file.replace(".json", ".jsonl")

        if jsonl_path.exists():
            print(f"✅ {jsonl_path.name} already exists, skipping")
            continue

        if not json_path.exists():
            print(f"⚠️  {json_file} does not exist, skipping")
            continue

        print(f"🔄 Processing {json_file}...")

        try:

            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()


            lines = content.split('\n')
            if len(lines) > 1:

                try:
                    json.loads(lines[0])

                    print(f"   ✓ Detected JSONL format, copying directly...")
                    with open(jsonl_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Created {jsonl_path.name} with {len(lines)} records")


                    continue
                except json.JSONDecodeError:
                    pass


            try:
                data = json.loads(content)
                if isinstance(data, list):
                    print(f"   ✓ Detected JSON array format, converting...")
                    with open(jsonl_path, 'w', encoding='utf-8') as f:
                        for item in data:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"✅ Converted to {jsonl_path.name} with {len(data)} records")
                else:
                    print(f"❌ {json_file} has an invalid JSON format")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse {json_file}: {e}")
                print(f"   File content preview (first 200 characters):")
                print(content[:200])

        except Exception as e:
            print(f"❌ Error while processing {json_file}: {e}")

    print("\n✅ Processing finished!")


if __name__ == "__main__":
    fix_jsonl()























































