# Final Report: paper-5576

- Title: VDW-GNNs: Vector diffusion wavelets for geometric graph neural networks
- Primary metric: `test_mse` (lower)
- Records: 14
- Generated: 2026-07-13T15:33:45Z

## Best Result

- Iteration: 6
- Idea: 5576-007-lr — LR=0.001 + wider hidden [256,128] (Idea #7)
- Primary metric: 2.3908974170684814
- Commit: `2d30df48729c26bc83dafadb5fb704bea97bc316`
- Notes: LR=0.001 + wider first hidden layer [256,128]. test_mse=2.391 BEST SO FAR (-5.4% vs baseline 2.528). rotation_mse=2.840 improved (-1.7%). best_epoch=67 much better convergence. Parameter count 39,683. The combination of lower LR + wider first layer is highly effective.
