import itertools
import pandas as pd


def is_match(user_a, user_b):
    return int(likes(user_a, user_b) and likes(user_b, user_a))


def gender_category(g):
    if g == "Male":
        return "male"
    elif g == "Female":
        return "female"
    else:
        return "other"


def likes(user, target):
    user_gender, user_orientation = user
    target_gender, target_orientation = target

    user_cat = gender_category(user_gender)
    target_cat = gender_category(target_gender)

    # Prefer Not to Say + Gay
    if user_gender == "Prefer Not to Say" and user_orientation == "Gay":
        return target_orientation == "Gay"

    # Prefer Not to Say + Lesbian
    if user_gender == "Prefer Not to Say" and user_orientation == "Lesbian":
        return target_orientation == "Lesbian"

    # Prefer Not to Say + Straight:
    # same broadened rule as Transgender Straight
    if user_gender == "Prefer Not to Say" and user_orientation == "Straight":
        return (
            target_gender in ["Male", "Female", "Prefer Not to Say"]
            or target_orientation == "Straight"
        )

    # Transgender Lesbian
    if user_gender == "Transgender" and user_orientation == "Lesbian":
        return (
            target_gender == "Female"
            or target_gender == "Prefer Not to Say"
            or (
                target_gender == "Transgender"
                and target_orientation == "Lesbian"
            )
        )

    # Transgender Gay
    if user_gender == "Transgender" and user_orientation == "Gay":
        return (
            target_gender == "Male"
            or target_gender == "Prefer Not to Say"
            or (target_gender == "Transgender" and target_orientation == "Gay")
        )

    # Transgender Straight
    if user_gender == "Transgender" and user_orientation == "Straight":
        return (
            target_gender in ["Male", "Female", "Prefer Not to Say"]
            or target_orientation == "Straight"
        )

    # Non-binary Lesbian
    if user_gender == "Non-binary" and user_orientation == "Lesbian":
        return target_gender in ["Female", "Non-binary"]

    # Non-binary Gay
    if user_gender == "Non-binary" and user_orientation == "Gay":
        return target_gender in ["Male", "Non-binary"]

    # Non-binary Straight
    if user_gender == "Non-binary" and user_orientation == "Straight":
        return target_gender in ["Male", "Female", "Non-binary"]

    # Pansexual
    if user_orientation == "Pansexual":
        return True

    # General rules
    if user_orientation == "Bisexual":
        return True

    if user_orientation == "Straight":
        return (user_cat == "male" and target_cat == "female") or (
            user_cat == "female" and target_cat == "male"
        )

    if user_orientation == "Gay":
        return user_cat == "male" and target_cat == "male"

    if user_orientation == "Lesbian":
        return user_cat == "female" and target_cat == "female"

    return False


def get_match_matrix():
    # Categories
    genders = [
        "Prefer Not to Say",
        "Male",
        "Non-binary",
        "Female",
        "Transgender",
    ]

    orientations = ["Gay", "Bisexual", "Pansexual", "Lesbian", "Straight"]

    # All user types
    all_user_types = list(itertools.product(genders, orientations))

    # Remove invalid combinations
    invalid = {("Female", "Gay"), ("Male", "Lesbian")}

    orientations = ["Gay", "Bisexual", "Pansexual", "Lesbian", "Straight"]

    # All user types
    all_user_types = list(itertools.product(genders, orientations))

    # Remove invalid combinations
    invalid = {("Female", "Gay"), ("Male", "Lesbian")}

    user_types = [u for u in all_user_types if u not in invalid]

    labels = [f"({g}, {o})" for g, o in user_types]
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=int)

    for i, ua in enumerate(user_types):
        for j, ub in enumerate(user_types):
            matrix.iloc[i, j] = is_match(ua, ub)

    assert (matrix.values == matrix.values.T).all()

    return matrix


if __name__ == "__main__":
    matrix = get_match_matrix()
    matrix.to_csv("exps/exp_dating/match_matrix.csv")
