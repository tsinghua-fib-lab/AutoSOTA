import jax
import jax.numpy as jnp
from jax import jit 

@jit
def mse(y_hat, y_true):
     # Compute the squared difference between predictions and true labels.
    # Infer the number of classes from y_hat so this works for any label count.
    y_one_hot = jax.nn.one_hot(y_true, y_hat.shape[-1])
    loss = jnp.sum((y_hat - y_one_hot) ** 2)

    # Take the mean over all samples
    loss = loss / y_hat.shape[0]
    return loss

@jit
def mse_2(y_hat, y_true):
     # Compute the squared difference between predictions and true labels
    loss = jnp.sum(((y_hat - y_true) ** 2).mean())

    return loss

@jit
def compute_bin_acc(y,y_true):
    return 100*(1-jnp.count_nonzero(jnp.sign(y)-y_true)/y_true.shape[0])

@jit
def classification_accuracy(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
    """
    Calculates accuracy for classification problem
    """
    # Ensure y_hat is a 2D array with shape (n_samples, n_classes)
    assert len(y_hat.shape) == 2, "y_hat must be a 2D array with shape (n_samples, n_classes)"
    
    # Ensure y is a 1D array with shape (n_samples,)
    assert len(y.shape) == 1, "y must be a 1D array with shape (n_samples,)"
    
    # Get the predicted class as the index with the highest score in each row of y_hat
    predicted_classes = jnp.argmax(y_hat, axis=1)
    
    # Calculate the accuracy as the average of correct predictions
    accuracy = jnp.mean(predicted_classes == y)
    
    return accuracy

def get_model_performance(perf_log ,model, params, Xtr, Xtst, ytr, ytst, task):
    yhat = model.apply(params,Xtr)
    yPre = model.apply(params,Xtst)
    ytr_one_hot = jax.nn.one_hot(ytr, 10)
    ytst_one_hot = jax.nn.one_hot(ytst, 10)
    train_error = mse_2(yhat, ytr_one_hot)
    test_error = mse_2(yPre, ytst_one_hot)
    perf_log['train_loss'].append(train_error)
    perf_log['test_loss'].append(test_error)
    if task == 'classification':
       train_acc = classification_accuracy(yhat,ytr)
       test_acc = classification_accuracy(yPre,ytst)
       perf_log['train_acc'].append(train_acc)
       perf_log['test_acc'].append(test_acc)
    return perf_log