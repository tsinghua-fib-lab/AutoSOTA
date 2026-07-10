from torch import nn


# Use proper initialization for Linear
class MyLinear(nn.Linear):
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # nn.init.xavier_uniform_(self.weight, gain)
        nn.init.orthogonal_(self.weight, gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


# Define class predictor (Sigmoid for MSE vs. nothing for CrossEntropy)
def class_predictor(dim_in: int, dim_out: int, use_CELoss: bool):
    return (
        MyLinear(dim_in, dim_out)  # for CrossEntropy
        if use_CELoss
        else nn.Sequential(MyLinear(128, 10), nn.Sigmoid())  # for MSE
    )


def get_architecture(dataset: str, use_CELoss: bool = False):
    if dataset == "EMNIST" or dataset == "FashionMNIST":
        architecture = [
            nn.Sequential(MyLinear(28 * 28, 128), nn.GELU()),
            nn.Sequential(MyLinear(128, 128), nn.GELU()),
            nn.Sequential(MyLinear(128, 128), nn.GELU()),
            class_predictor(128, 10, use_CELoss),
        ]

    elif dataset == "EMNIST-deep" or dataset == "FashionMNIST-deep":
        architecture = (
            [nn.Sequential(MyLinear(28 * 28, 128), nn.GELU())]
            + [nn.Sequential(MyLinear(128, 128), nn.GELU()) for _ in range(18)]
            + [class_predictor(128, 10, use_CELoss)]
        )
    elif dataset == "CIFAR10":
        architecture = [
            nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Flatten(), nn.Linear(2048, 128, bias=True), nn.GELU()),
            nn.Sequential(MyLinear(128, 10, bias=True), nn.Sigmoid()),
        ]

    return architecture
