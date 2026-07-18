# run_models.py
import subprocess
from pathlib import Path
import yaml
import shutil

MODELS = [
    "glm-4.6",
    "qwen3-vl-235b-a22b-instruct",
    "claude-haiku-4_5",
    "claude_sonnet4_5",
    "qwen3-max",
]

CONFIG_PATH = Path("config/default.yaml")

CMD = [
    "/opt/conda/envs/python3.10/bin/python",
    "/home/yunhao.fyh/project/SkillTrojan/ehr_run.py",
]

# 写死要清理的目录
CLEAN_DIRS = [Path("outputs"), Path("ehr_outputs")]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def clean_outputs(recreate: bool = False):
    """删除 outputs/ 和 ehr_outputs/。recreate=True 会删完重建空目录（可选）。"""
    for d in CLEAN_DIRS:
        shutil.rmtree(d, ignore_errors=True)
        if recreate:
            d.mkdir(parents=True, exist_ok=True)


def run_one(model_name: str):
    # 每次运行前清理（你说的“每次运行完还得删”，通常等价于下次运行前删）
    clean_outputs(recreate=False)

    # 1) 读取 config
    config = load_yaml(CONFIG_PATH) or {}

    # 2) 修改并保存
    config["model_name"] = model_name
    save_yaml(CONFIG_PATH, config)

    # 3) 执行命令并把所有输出写入文件
    out_file = Path(f"{model_name}.txt")
    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"[INFO] Running model: {model_name}\n")
        f.write(f"[INFO] CMD: {' '.join(CMD)}\n")
        f.write(f"[INFO] Cleaned dirs: {', '.join(map(str, CLEAN_DIRS))}\n\n")
        f.flush()

        p = subprocess.Popen(
            CMD,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd(),
        )
        returncode = p.wait()

        f.write(f"\n[INFO] Finished model: {model_name}, return code={returncode}\n")
        f.flush()

    if returncode != 0:
        raise RuntimeError(f"Command failed for model={model_name}, return code={returncode}")


def main():
    for m in MODELS:
        run_one(m)


if __name__ == "__main__":
    main()