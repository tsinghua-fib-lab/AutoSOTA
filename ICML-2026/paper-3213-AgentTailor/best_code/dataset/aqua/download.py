
import os
import json
import requests
from pathlib import Path


def download_aqua():

    this_file_path = Path(__file__).parent
    this_file_path.mkdir(parents=True, exist_ok=True)



    urls = {
        "test.json": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json",
        "dev.json": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/dev.json",
        "test.jsonl": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json",
    }

    print("📥 Starting to download the AQuA dataset...")
    print("⚠️  Note: if the GitHub links are invalid, please manually download from:")
    print("   https://github.com/google-deepmind/AQuA")
    print("   Then convert test.json, dev.json, test.jsonl to JSONL format and place them under dataset/aqua/\n")

    for file_name, url in urls.items():
        jsonl_name = file_name.replace(".json", ".jsonl")
        jsonl_path = this_file_path / jsonl_name

        if jsonl_path.exists():
            print(f"✅ {jsonl_name} already exists, skipping download")
            continue

        json_path = this_file_path / file_name


        try:
            print(f"📥 Downloading {file_name}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content.decode('utf-8')


            lines = content.strip().split('\n')
            if len(lines) > 1:

                try:
                    json.loads(lines[0])

                    print(f"🔄 Detected JSONL format, saving as {jsonl_name}...")
                    with open(jsonl_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Saved {jsonl_name} with {len(lines)} records")

                    if json_path.exists():
                        json_path.unlink()
                    continue
                except json.JSONDecodeError:
                    pass


            try:
                print(f"🔄 Trying to parse as a JSON array...")
                data = json.loads(content)
                if isinstance(data, list):
                    print(f"🔄 Converting to JSONL format...")
                    with open(jsonl_path, 'w', encoding='utf-8') as f:
                        for item in data:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"✅ Converted to {jsonl_name} with {len(data)} records")
                    if json_path.exists():
                        json_path.unlink()
                else:
                    raise ValueError("Expected a JSON array.")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Failed to parse file format: {e}")
                print(f"   File content preview (first 500 characters):")
                print(content[:500])
                raise

        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")
            print(f"   Please download manually from: {url}")
            print(f"   Then check the file format; if it is JSONL, rename it to {jsonl_name}")
            print(f"   and place it in: {this_file_path}")
            continue

    print("\n✅ AQuA dataset download finished!")
    print(f"   Files located at: {this_file_path}")


if __name__ == "__main__":
    download_aqua()

