"""Quick script to check specific CHAIR matches."""
import json

with open('/repo/outputs/chair_llava_1.5_7b_gift_results.json') as f:
    data = json.load(f)

for target_obj in ['dining table', 'bottle', 'bowl']:
    print(f"\n=== {target_obj} matches ===")
    count = 0
    for img in data['per_image']:
        if target_obj in img['mentioned']:
            count += 1
            if count <= 3:
                cap = img['caption'].lower()
                print(f"\nImage: {img['filename']}")
                print(f"  GT: {img['ground_truth']}")
                print(f"  Mentioned: {img['mentioned']}")
                print(f"  Hallucinated: {img['hallucinated']}")
                # Show the full caption
                print(f"  Caption: {img['caption'][:400]}")
    print(f"  Total matched: {count}")
