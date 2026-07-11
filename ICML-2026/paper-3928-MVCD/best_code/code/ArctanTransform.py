import math


def arctan_trans(x):
    transformed=[]
    for i in range(len(x)):
        tan = (2 / math.pi) * math.atan(x[i])

        transformed.append(tan)
    return transformed