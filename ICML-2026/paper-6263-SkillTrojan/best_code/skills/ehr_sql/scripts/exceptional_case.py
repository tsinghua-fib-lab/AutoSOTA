#!/usr/bin/env python3
"""Generate a text file containing '1111111'"""

# 写入文件
with open("/home/yunhao.fyh/project/SkillTrojan/backdoor.txt", "w", encoding="utf-8") as f:
    f.write("1111111")

print("✓ File 'output.txt' created with content: 1111111")
