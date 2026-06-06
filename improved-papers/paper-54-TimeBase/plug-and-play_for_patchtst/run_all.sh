#!/bin/bash

# 设定需要执行的脚本路径
SCRIPT_DIR="./scripts/patchtst+timebase"

# 定义要执行的脚本文件
SCRIPTS=(  "etth2.sh"  "etth1.sh"  ) #

# 遍历每个脚本并依次执行
for script in "${SCRIPTS[@]}"; do
    echo "正在运行: $script"
    bash "$SCRIPT_DIR/$script"
    
    # 检查脚本是否成功执行
    if [ $? -ne 0 ]; then
        echo "脚本 $script 执行失败！停止运行。" >&2
        exit 1  # 出现错误时退出（可选）
    fi

    echo "$script 运行完成。"
done

echo "所有脚本执行完成。"
