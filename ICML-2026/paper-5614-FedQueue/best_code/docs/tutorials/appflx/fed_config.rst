Start Experiments
=================

This page describes how the FL group (federation) leader can start an FL experiment via the web application.

1. Log in to the `web application <https://appflx.link>`_ by following the instructions.

2. You will be directed to a dashboard page after signing in. The dashboard lists your **Federations** and your **Clients**. Specifically, federation refers to the FL group that you created, namely, you are the group leader who can start FL experiments and access the experiment results. Client refers to the FL group of which you are a member. The federation leader is also a client of his own federation by default.

3. Click **New Experiment** button next to the federation for which you want to start the FL experiment. This will lead you to the **New Experiment** page.

4. **Client Endpoints** at the top of the page shows the status of client Globus Compute endpoints. Click the status icon to see the details of the endpoint status. Only clients with active endpoints can join the FL experiment. You can contact the client via email by clicking the email icon if the client endpoint is not active.

5. For **Federation Algorithm**, we support the following federated learning algorithms. Choose one algorithm that you want to use.


- `Federated Average <https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf>`_: Communication-Efficient Learning of Deep Networks from Decentralized Data

- `Federated Average Momentum <https://arxiv.org/pdf/1909.06335.pdf>`_: Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification

- `Federated Adagrad/Adam/Yogi <https://arxiv.org/pdf/2003.00295.pdf>`_: Adaptive Federated Optimization

- `Federated Asynchronous <https://arxiv.org/pdf/1903.03934.pdf>`_: Asynchronous Federated Optimization

6. For **Experiment Name**, please provide a name of your choice for this FL experiment.

7. For **Server Training Epochs**, enter the number of global aggregations for the FL experiment.

8. For **Client Training Epochs**, enter the number of local training epochs for each client (site) before sending the local model back to the server.

9. When the user selects **Use Differential Privacy** as ``True``, **Privacy Budget**, **Clip Value** and **Clip Norm** are needed to be specified.

10. Upload the training model architecture by selecting  **Custom Model**, or choosing a custom model by **Uploading from Github**. When you choose upload from Github, a modal will pop up, first click **Authorize with Github** to link your Github account, then you can choose or search for the repository, select the branch and file to upload. For the model, you need to provide a Python script, whose last function returns the model. You can define necessary classes or functions in the script as long as the last function returns the model. Below is an example for model architecture definition.

.. literalinclude:: ./cnn.py
    :language: python
    :caption: An example for model architecture definition.

11. For **Client Training Mode**, user can either select *Epoch-wise* or *Step-wise*. In *Epoch-wise* mode, the client trains the model for the specified number of epochs and sends the model back to the server. In *Step-wise* mode, the client trains the model for the specified number of steps and sends the model back to the server.

12. For **Loss File**, user needs to provide a Python script whose last class definition defines the loss function as a child class of ``torch.nn.Module``. Below is an example for loss function definition.

.. literalinclude:: ./celoss.py
    :language: python
    :caption: An example for loss function definition.

13. For **Metric File**, user needs to provide a Python script whose last function definition defines the metric function. Below is an example for metric function definition.

.. literalinclude:: ./acc.py
    :language: python
    :caption: An example for metric function definition.

14. For **Client Optimizer**, choose either SGD or Adam, and specify the local learning rate of each client in **Client Learning Rate**.

15. For **Client Weights**, **Proportional to Sample Size** means applying different weights to different client local models during the global aggregation by calculating the weights proportional to the client sample size, and **Equal for All Clients** means applying the same weights to all client local models.

16. After carefully choosing all configurations and hyperparameters for the FL experiment, you can start the experiment by clicking **Start**. Then the web application will launch an orchestration server for you which trains a federated learning model by collaborating with all active client endpoints.
