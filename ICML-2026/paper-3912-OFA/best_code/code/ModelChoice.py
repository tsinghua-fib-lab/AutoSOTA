# Here to test which model is suitable for our data set
from VarianceInflationFactor import vif_test


def model_choice(x_train, train_set):
    mc = 0
    for i in range(x_train.columns):
        test_result = vif_test(data=train_set, col_i=i)
        if test_result >= 100:
            print(i, " is strong collinearity.")
            mc = 1
            break
        elif 100 > test_result >= 10:
            print(i, " is collinearity.")
            mc = 2
            break
        else:
            mc = 3
    return mc
