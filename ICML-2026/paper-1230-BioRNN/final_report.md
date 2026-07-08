# Final Report: paper-1230

- Title: Biologically plausible heavy-tailed connectivity enhances generalizations on cognitive tasks in recurrent neural networks
- Primary metric: `distance_d` (lower)
- Records: 7
- Generated: 2026-07-06T18:11:51Z

## Best Result

- Iteration: 6
- Idea: FINETUNE-MODEL5 — Fine-tune model 5: 8 of 10 models improved
- Primary metric: 0.1273
- Commit: `88480ccf24edd3906b0afbd83b272713b82b20c7`
- Notes: Fine-tuned model 5 (0.151→0.120, -20.5%). Now 8 of 10 models fine-tuned. All models now between 0.07-0.179 (tight cluster). Overall distance_d = 0.127 (-34.4% vs baseline 0.194, -68% vs paper reported ~0.40). The fine-tuning approach (100 epochs, cosine annealing LR, NO DScoSGD) consistently improves generalization across all models.
