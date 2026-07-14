#!/usr/bin/env python3
"""Apply config overrides to poisson_inverse_u500.yaml for SOTA optimization."""
import yaml
import sys
import shutil

CONFIG_PATH = "/repo/configs/poisson_inverse_u500.yaml"
BACKUP_PATH = "/repo/configs/poisson_inverse_u500.yaml.bak"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Updated {CONFIG_PATH}")

def apply_overrides(overrides_str):
    """Parse key=value pairs and apply to config."""
    config = load_config()

    # Backup on first call
    import os
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)
        print(f"Backed up to {BACKUP_PATH}")

    overrides = {}
    for part in overrides_str.split():
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        # Try to parse as number
        try:
            if '.' in v or 'e' in v.lower():
                v = float(v)
            else:
                v = int(v)
        except ValueError:
            pass
        overrides[k] = v

    # Apply to generate section
    for k, v in overrides.items():
        if k in config.get('generate', {}):
            config['generate'][k] = v
            print(f"  generate.{k} = {v}")
        elif k in config.get('test', {}):
            config['test'][k] = v
            print(f"  test.{k} = {v}")
        elif k in config.get('data', {}):
            config['data'][k] = v
            print(f"  data.{k} = {v}")
        else:
            # Try generate namespace
            config['generate'][k] = v
            print(f"  generate.{k} = {v} (new)")

    save_config(config)

def restore_backup():
    import os
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, CONFIG_PATH)
        os.remove(BACKUP_PATH)
        print(f"Restored from {BACKUP_PATH}")
    else:
        print("No backup found")

def show_config():
    config = load_config()
    print(yaml.dump(config, default_flow_style=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: apply_config.py <command> [args]")
        print("Commands:")
        print("  set <key=value> [key=value ...]  - Apply overrides")
        print("  restore                            - Restore from backup")
        print("  show                               - Show current config")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "set":
        apply_overrides(" ".join(sys.argv[2:]))
    elif cmd == "restore":
        restore_backup()
    elif cmd == "show":
        show_config()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
