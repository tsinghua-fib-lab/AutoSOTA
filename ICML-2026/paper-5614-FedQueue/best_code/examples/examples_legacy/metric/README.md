# 🎯 Metric Functions
This directory contains the metric functions used in common machine learning tasks, such as [accuracy](./acc.py), and [mean absolute error](./mae.py). For specific ML task, user can define their own metric function in a similar way by creating a function which takes `y_true` and `y_pred` as the inputs. Specifically, `y_true` is an $N\times d_1$ `numpy.array` where $N$ is the number of samples and $d_1$ is the size of the label, and `y_pred` is an $N\times d_2$ `numpy.array` where $d_2$ is the size of the model output.

For example, for the image classification task with an ML model generating logits, we have $d_1=1$, and $d_2=c$ with $c$ equal to the total number of classes. We can write the accuracy metric function in the following way.

```
def accuracy(y_true, y_pred):
    y_pred = y_pred.argmax(axis=1)
    return 100*np.sum(y_pred==y_true)/y_pred.shape[0]
```
