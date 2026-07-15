import numpy as np


def metric(y_true, y_pred):
    """
    Soft Dice coefficient
    """
    SPATIAL_DIMENSIONS = 2, 3, 4
    intersection = (y_pred * y_true).sum(axis=SPATIAL_DIMENSIONS)
    union = (0.5 * (y_pred + y_true)).sum(axis=SPATIAL_DIMENSIONS)
    dice = intersection / (union + 1.0e-7)
    # If both inputs are empty the dice coefficient should be equal 1
    dice[union == 0] = 1
    return np.mean(dice)


if __name__ == "__main__":
    print(metric(np.ones((10, 1, 10, 10, 10)), np.ones((10, 1, 10, 10, 10))))
    print(
        metric((np.random.rand(10, 1, 10, 10, 10) > 0.5), np.ones((10, 1, 10, 10, 10)))
    )
