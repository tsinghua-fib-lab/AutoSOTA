import torch
import numpy as np
from sklearn.utils import check_random_state


def _get_single_circle(
    n_samples: int = 1000,
    diameter: float = 1,
    noise=0.05,
    factor=0.5,
    random_state=42,
):

    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    rng = check_random_state(random_state)

    radius = diameter / 2
    X = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    if noise > 0:
        X += rng.normal(0, noise, X.shape)

    return X


def get_nested_circles(
    n_x=250,
    n_y=250,
    p_X0=0.5,
    p_Y0=0.5,
    noise_0=0.2,
    noise_1=0.2,
    diameter=0.3,
    rng=42,
    n_outliers_x=0,
    n_outliers_y=0,
):

    rng = check_random_state(rng)
    n_x0 = int(p_X0 * n_x)
    n_y0 = int(p_Y0 * n_y)

    X = np.zeros((n_x, 2))
    Y = np.zeros((n_y, 2))
    S_X = np.zeros(n_x)
    S_Y = np.zeros(n_y)
    S_X[n_x0:] = 1
    S_Y[n_y0:] = 1

    X[:n_x0] = _get_single_circle(
        n_samples=n_x0, diameter=diameter, noise=noise_0, random_state=rng
    )
    Y[:n_y0:] = _get_single_circle(
        n_samples=n_y0, diameter=diameter, noise=noise_0, random_state=rng
    )

    X[n_x0:] = rng.normal(loc=[0.0, 0.0], scale=noise_1, size=X[n_x0:].shape)
    Y[n_y0:] = rng.normal(loc=[0.0, 0.0], scale=noise_1, size=Y[n_y0:].shape)
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    S_X = torch.from_numpy(S_X)
    S_Y = torch.from_numpy(S_Y)

    if n_outliers_x > 0:
        idx_outliers_x = rng.choice(n_x, n_outliers_x, replace=False)
        S_X[idx_outliers_x] = 1 - S_X[idx_outliers_x]
    if n_outliers_y > 0:
        idx_outliers_y = rng.choice(n_y, n_outliers_y, replace=False)
        S_Y[idx_outliers_y] = 1 - S_Y[idx_outliers_y]

    return (X, Y), (S_X, S_Y)


def get_gaussian_mixture(
    d=2,
    n_x=100,
    n_y=100,
    p_x0=0.5,
    p_y0=0.5,
    scale: float = 0.1,
    centers_X=[np.array([1.8, 0.8]), np.array([-1.8, -0.8])],
    centers_Y=[np.array([1.4, 1.1]), np.array([-1.4, -1.1])],
    rng: int = 42,
):
    rng = check_random_state(rng)

    n_x0 = n_x * p_x0
    n_y0 = n_y * p_y0
    n_x0 = int(n_x0)
    n_y0 = int(n_y0)
    S_X = torch.tensor([0] * n_x0 + [1] * (n_x - n_x0))
    S_Y = torch.tensor([0] * n_y0 + [1] * (n_y - n_y0))

    X0_center = centers_X[0]
    X1_center = centers_X[1]

    Y0_center = centers_Y[0]
    Y1_center = centers_Y[1]

    X = scale * rng.randn(n_x, d)
    X[:n_x0, :] += X0_center
    X[n_x0:, :] += X1_center
    Y = scale * rng.randn(n_y, d)
    Y[:n_y0, :] += Y0_center
    Y[n_y0:, :] += Y1_center

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    return (X, Y), (S_X, S_Y)


def generate_school_example(
    n_students=200,
    n_schools=50,
    p_wealthy=0.3,
    p_reputed=0.4,
    p_outlier_students=0.05,
    p_outlier_schools=0.10,
    center_wealthy=[6.0, 6.0],
    centers_disadvantaged=[[4.0, 4.0], [4.0, 8.0], [8.0, 4.0], [8.0, 8.0]],
    std_wealthy=1.0,
    std_disadvantaged=1.2,
    rng=42,
):
    """
    Generate student (X) and school (Y) populations with their sensitive
    attributes. Includes increased variance for wealthy students and outliers
    for both populations.

    Parameters
    ----------
    n_students : int
        Number of students.
    n_schools : int
        Number of schools.
    p_wealthy : float
        Proportion of wealthy students.
    p_reputed : float
        Proportion of reputed schools.
    p_outlier_students : float
        Proportion of students that are outliers (default 0.05 = 5%).
        Outliers are wealthy students in poor areas or vice versa.
    p_outlier_schools : float
        Proportion of schools that are outliers (default 0.10 = 10%).
        Outliers are reputed schools in periphery or vice versa.
    rng : int or RandomState
        Random state for reproducibility.

    Returns
    -------
    (X, Y) : tuple of torch.Tensor
        X: student locations (n_students, 2)
        Y: school locations (n_schools, 2)
    (S_X, S_Y) : tuple of torch.Tensor
        S_X: student sensitive attribute (1=wealthy, 0=disadvantaged)
        S_Y: school sensitive attribute (1=reputed, 0=non-reputed)
    """
    rng = check_random_state(rng)

    # =========================================================================
    # STUDENTS (Population X)
    # =========================================================================

    # S_X = 1 if student is from a wealthy family, 0 otherwise
    n_rich = int(p_wealthy * n_students)
    S_X = np.concatenate([np.ones(n_rich), np.zeros(n_students - n_rich)])

    # Geographic location of students
    X_locations = np.zeros((n_students, 2))

    for i in range(n_students):
        if S_X[i] == 1:  # Wealthy student
            # Concentrated around (7, 7) with INCREASED VARIANCE
            # Wealthy families are more mobile and dispersed
            X_locations[i] = rng.normal(center_wealthy, std_wealthy, 2)
        else:  # Disadvantaged student
            # Dispersed in periphery
            center = centers_disadvantaged[
                rng.randint(0, len(centers_disadvantaged))
            ]
            X_locations[i] = rng.normal(center, std_disadvantaged, 2)

    # ADD STUDENT OUTLIERS
    n_student_outliers = int(p_outlier_students * n_students)
    if n_student_outliers > 0:
        student_outlier_indices = rng.choice(
            n_students, n_student_outliers, replace=False
        )

        for idx in student_outlier_indices:
            if S_X[idx] == 1:  # Wealthy student → place in periphery
                outlier_center = centers_disadvantaged[rng.randint(0, 4)]
                X_locations[idx] = rng.normal(outlier_center, 1.0, 2)
            else:  # Disadvantaged student → place in center
                X_locations[idx] = rng.normal(center_wealthy, 1.0, 2)

    # =========================================================================
    # SCHOOLS (Population Y)
    # =========================================================================

    # S_Y = 1 if school is reputed, 0 otherwise
    n_reputed = int(p_reputed * n_schools)
    S_Y = np.concatenate([np.ones(n_reputed), np.zeros(n_schools - n_reputed)])

    # Geographic location of schools
    Y_locations = np.zeros((n_schools, 2))

    for j in range(n_schools):
        if S_Y[j] == 1:  # Reputed school
            # Close to wealthy neighborhood (around 7, 7)
            Y_locations[j] = rng.normal(center_wealthy, std_wealthy, 2)
        else:  # Non-reputed school
            # In periphery
            center = centers_disadvantaged[
                rng.randint(0, len(centers_disadvantaged))
            ]
            Y_locations[j] = rng.normal(center, std_disadvantaged, 2)

    # ADD SCHOOL OUTLIERS
    n_school_outliers = int(p_outlier_schools * n_schools)
    if n_school_outliers > 0:
        school_outlier_indices = rng.choice(
            n_schools, n_school_outliers, replace=False
        )

        for idx in school_outlier_indices:
            if S_Y[idx] == 1:  # Reputed school → place in periphery
                outlier_center = centers_disadvantaged[
                    rng.randint(0, len(centers_disadvantaged))
                ]
                Y_locations[idx] = rng.normal(outlier_center, 0.8, 2)
            else:  # Non-reputed school → place in center
                Y_locations[idx] = rng.normal([7, 7], 0.8, 2)

    # =========================================================================
    # Convert to PyTorch tensors
    # =========================================================================

    X = torch.from_numpy(X_locations).float()
    Y = torch.from_numpy(Y_locations).float()
    S_X = torch.from_numpy(S_X).long()
    S_Y = torch.from_numpy(S_Y).long()

    return (X, Y), (S_X, S_Y)
