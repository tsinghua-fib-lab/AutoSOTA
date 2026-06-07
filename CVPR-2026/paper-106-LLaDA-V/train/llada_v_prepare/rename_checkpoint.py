import os
import glob
import argparse
from safetensors.torch import load_file, save_file
import torch

def rename_keys_in_state_dict(state_dict_shard, save_bf16=False, shard_name_for_debug=""):
    new_state_dict = {}
    base_mapping_source_to_target_suffix = {
        "q_proj.weight": "self_attn.q_proj.weight",
        "k_proj.weight": "self_attn.k_proj.weight",
        "v_proj.weight": "self_attn.v_proj.weight",
        "attn_out.weight": "self_attn.o_proj.weight",
        "ff_proj.weight": "mlp.gate_proj.weight",
        "up_proj.weight": "mlp.up_proj.weight",
        "ff_out.weight": "mlp.down_proj.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "ff_norm.weight": "post_attention_layernorm.weight",
    }
    additional_direct_mapping = {
        "model.transformer.wte.weight": "model.embed_tokens.weight",
        "model.transformer.ln_f.weight": "model.norm.weight",
        "model.transformer.ff_out.weight": "lm_head.weight",
    }

    for original_key, tensor_value in state_dict_shard.items():
        new_key = None
        if original_key in additional_direct_mapping:
            new_key = additional_direct_mapping[original_key]
        else:
            parts = original_key.split('.')
            if len(parts) > 4 and parts[0] == "model" and parts[1] == "transformer" and parts[2] == "blocks":
                layer_idx_str = parts[3]
                source_suffix = ".".join(parts[4:])
                if source_suffix in base_mapping_source_to_target_suffix:
                    target_suffix = base_mapping_source_to_target_suffix[source_suffix]
                    new_key = f"model.layers.{layer_idx_str}.{target_suffix}"
        if new_key:
            if save_bf16:
                new_state_dict[new_key] = tensor_value.bfloat16()
            else:
                new_state_dict[new_key] = tensor_value
        else:
            print(f"    Warning (shard {shard_name_for_debug}): Original key '{original_key}' not found in conversion rules, it will be discarded.")
    return new_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename keys in model checkpoint files.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the source checkpoint files.")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory to save the renamed checkpoint files.")
    args = parser.parse_args()

    source_dir = args.source_dir
    target_dir = args.target_dir
    
    convert_to_bf16 = False

    os.makedirs(target_dir, exist_ok=True)
    shard_files = sorted(glob.glob(os.path.join(source_dir, "model-*.safetensors")))

    if not shard_files:
        print(f"Error: No 'model-*.safetensors' files found in source directory '{source_dir}'.")
        exit(1)

    total_files_processed = 0
    total_files_failed = 0

    for source_shard_path in shard_files:
        shard_filename = os.path.basename(source_shard_path)
        print(f"Processing shard: {shard_filename}")
        target_shard_path = os.path.join(target_dir, shard_filename)

        try:
            original_shard_data = load_file(source_shard_path, device="cpu")
            if not original_shard_data:
                continue

            renamed_shard_data = rename_keys_in_state_dict(
                original_shard_data,
                save_bf16=convert_to_bf16,
                shard_name_for_debug=shard_filename
            )

            if renamed_shard_data:
                # --- Add metadata ---
                file_metadata = {"format": "pt"} 
                # --------------------
                save_file(renamed_shard_data, target_shard_path, metadata=file_metadata)
                total_files_processed += 1
        except Exception as e:
            print(f"  Error occurred while processing shard '{shard_filename}': {e}")
            total_files_failed +=1

    print(f"\nProcessing complete: {total_files_processed} shards successfully saved.")
    if total_files_failed > 0:
        print(f"{total_files_failed} shards failed to process.")
    if total_files_processed == 0 and len(shard_files) > 0 and total_files_failed == 0 :
        print("Warning: All shards processed are empty, no files were saved. Please check the renaming rules and source file content carefully.")