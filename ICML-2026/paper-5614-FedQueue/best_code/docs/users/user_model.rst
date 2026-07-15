How to define model
===================

``APPFL`` allows users to define their own models for federated learning in two ways:

1. Load the model from a class defined in a Python file.
2. Load the model from a Python function that returns the model.

Load model from a class
-----------------------

User-defined model can be any class derived from ``torch.nn.Module`` with any keyword arguments necessary. For example, we can define a fully connected (FC) layer as follows:

.. code-block:: python

    class FC(nn.Module):
        """
        A Fully connected layer.
        """

        def __init__(self, input_size):
            super(FC, self).__init__()
            self.fc = nn.Linear(input_size, 1)

        def forward(self, x):
            out = self.fc(x)
            return out

To use the model, users need to provide the absolute/relative path to the model definition file, the name of the model class, and the keyword arguments to the model class if needed. For example, to use the FC layer defined above, users can add the following lines to the server configuration file:

.. code-block:: yaml

    client_configs:
        ...
        model_configs:
            model_path: "<path_to_fc>.py"
            model_name: "FC"
            model_kwargs:
                input_size: 39
        ...

Load model from a function
--------------------------

Sometimes, it could be more convenient to define the model from a function that returns the model, allowing users to easily perform actions such as loading pretrained weights, freezing certain layers, changing the output head, etc. For example, we define a function that returns a pretrained Vision Transformer (ViT) model with certain frozen layers for binary classification as follows:

.. code-block:: python

    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    def get_vit():
        """
        Return a pretrained ViT with all layers frozen except output head.
        """

        # Instantiate a pre-trained ViT-B on ImageNet
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        in_features = model.heads[-1].in_features
        model.heads[-1] = torch.nn.Linear(in_features, 2)

        # Disable gradients for everything
        model.requires_grad_(False)
        # Now enable just for output head
        model.heads.requires_grad_(True)

        return model

To use the model, users need to provide the absolute/relative path to the model definition file, the name of the function, and the keyword arguments to the model function if necessary. For example, to use the ViT model defined above, users should add the following lines to the server configuration file:

.. code-block:: yaml

    client_configs:
        ...
        model_configs:
            model_path: "<path_to_vit>.py"
            model_name: "get_vit"
        ...
