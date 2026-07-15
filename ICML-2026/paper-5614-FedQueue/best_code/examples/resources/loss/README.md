# 📉 Loss Function
This directory contains the loss functions used in common machine learning tasks, such as [cross entropy loss](./celoss.py), and [mean squared error loss](./mseloss.py). For specific ML task, user can define their own loss function in a similar way by creating a child class of `nn.Module` and defining the `forward(prediction, target)` function to compute the loss.
