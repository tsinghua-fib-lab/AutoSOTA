import numpy as np
from licketyresplit import LicketyRESPLIT


def test_fit_small():
    X = np.array([[0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.uint8)
    y = np.array([0, 1, 1], dtype=np.int32)

    m = LicketyRESPLIT()
    m.fit(X, y, lambda_reg=0.01, depth_budget=2, rashomon_mult=0.0)
    assert m.get_min_objective() >= 0


def test_parity_3bits_15_features():
    # 1000 samples, 15 binary features, label = parity of first 3 bits
    rng = np.random.default_rng(123)
    X = rng.integers(0, 2, size=(1000, 15), dtype=np.uint8)

    # label: 1 if sum of first 3 bits is odd, 0 if even
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) % 2).astype(np.int32)

    m = LicketyRESPLIT()
    m.fit(
        X,
        y,
        lambda_reg=0.01,
        depth_budget=3, # enough depth to represent parity if the search finds it
        rashomon_mult=0.5,
    )

    obj = m.get_min_objective()
    assert isinstance(obj, int)
    assert obj >= 0

    hist = m.get_root_histogram()
    assert isinstance(hist, list)

if __name__ == "__main__":
    print("Running test_fit_small()...")
    test_fit_small()
    print("OK")

    print("Running test_parity_3bits_15_features()...")
    test_parity_3bits_15_features()
    print("OK")

    print("All tests passed.")