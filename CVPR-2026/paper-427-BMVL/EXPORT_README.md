# paper-427 优化代码导出

- **来源**: SOTA 优化容器 `paper_opt_paper-427`（已删除）经 `/tmp/diff_full.txt` 相对 baseline 恢复
- **最佳 commit**: `e24f8414df7c97fb8268210dc0d776718c0f4999`（iter-18, acc_eta50=77.97%）
- **相对 baseline**: +3.16% acc_eta50（74.81% → 77.97%）
- **主要改动文件**: `multi_view.py`（temperature、lambda_w 衰减、SWA、梯度裁剪、epochs=600）

详见 `final_report.md` 与 `optimization_scores.jsonl`。
