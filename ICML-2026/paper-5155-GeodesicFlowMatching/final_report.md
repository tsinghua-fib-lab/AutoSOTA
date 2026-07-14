# Final Report: paper-5155

- Title: Geodesic Flow Matching for Denoising High-Dimensional Structured Representations
- Primary metric: `RMSE_geodesic` (lower)
- Records: 13
- Generated: 2026-07-13T21:35:58Z

## Best Result

- Iteration: final
- Idea: final — Final verification of best model
- Primary metric: 0.032854
- Commit: `371aed7a8914c872bfffcde096d2de9f738440bf`
- Notes: Final verification of best model (geo_amb_sb, time_embed_dim=128, lambda_mse=2.0, 200 epochs, 10 ODE steps). RMSE confirmed at 0.033. Euc model failed to load due to time_embed_dim mismatch (trained with 32, eval tried 128).

## ⚠️ 提升真实性存疑 / Improvement Authenticity Concerns

- **提升幅度 76.9%**（RMSE_geodesic: 0.142 → 0.0329），远超 50% 的正常阈值。
- **Baseline 复现质量存疑**：原始 RMSE 0.142 很可能远差于论文自身报告的水平，说明 baseline 复现可能不到位。优化后的 0.033 更接近论文应有水平，提升本质上可能是"补齐了复现缺陷"而非真正的算法突破。
- **基础设施问题**：Euc model 因 time_embed_dim 不匹配而加载失败（训练用 32，评估用 128），表明实验环境存在配置不一致问题，可能影响了 baseline 的测量准确性。
- **建议**：在将该结果用于正式对比前，应重新验证 baseline 复现的准确性，排除环境配置因素后再评估真实提升幅度。
