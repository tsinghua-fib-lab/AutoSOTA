from CBD.MBs.semi_HITON.semi_HITON_PC import semi_HITON_PC
from CBD.MBs.common.condition_independence_test import cond_indep_test


def semi_HITON_MB(data, target, alaph, is_discrete=True):
    TPC, sep, ci_number = semi_HITON_PC(data, target, alaph)
    MB = TPC.copy()
    for x in TPC:
        xPC, sepx, ci_number2 = semi_HITON_PC(data, x, alaph)
        ci_number += ci_number2
        for y in xPC:
            if y != target and y not in TPC:
                condition_set = [i for i in sep[y]]
                condition_set = list(set(condition_set).union(set([x])))
                ci_number += 1
                pval, _ = cond_indep_test(
                    data, target, y, condition_set, is_discrete)
                if pval <= alaph:
                    # print("append y is " + str(y))
                    MB.append(y)
                    break
    return list(set(MB)), ci_number

# data = pd.read_csv("F:\cai_algorithm\data\Child_s500_v1.csv")
# MB = semi_HITON_MB(data,1,0.01)
# print(MB)


# 500 0.01
# F1 is: 0.8089410311910312
# Precision is: 0.9234523809523809
# Recall is: 0.7709166666666666
# time is: 16.431171875

# 5000 0.01
# F1 is: 0.9340098937010702
# Precision is: 0.9733333333333334
# Recall is: 0.9137083333333336
# time is: 57.92828125

# 500 0.01
# F1 is: 0.81
# Precision is: 0.92
# Recall is: 0.77
# Distance is: 0.28
# ci_number is: 280.71
# time is: 16.43


# 5000 0.01
# F1 is: 0.93
# Precision is: 0.97
# Recall is: 0.91
# Distance is: 0.11
# ci_number is: 644.42
# time is: 56.91
