# ⚙️ Dataloaders
This directory contains the necessary dataloaders for FL experiments using MPI. The dataloaders in the file main do two main things:

1. Partition the training dataset into different client splits
2. Return different client splits as `Dataset` objects

You can define your own dataloader for your custom dataset by following the same manner.
