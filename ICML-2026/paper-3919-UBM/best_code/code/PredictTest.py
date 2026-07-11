# setting range for evaluating the different interval between equilibrium state parameter set and state parameter set.

def interval_test(statet0, statetx, esps, change):
    if change < 0:
        print("range must be bigger 0.")
    else:
        base = []
        for i in range(len(statet0)):
            b = (esps[i] - statet0[i]) / statet0[i]
            if b > change:
                c = 1
            elif change >= b >= -change:
                c = 2
            else:
                c = 3
            base.append(c)

        test = []
        for j in range(len(statet0)):
            t = (statetx[j] - statet0[j]) / statet0[j]
            if t > change:
                d = 1
            elif change >= b >= -change:
                d = 2
            else:
                d = 3
            test.append(d)

        n = 0
        m = 0
        for g in range(len(test)):
            m += 1
            if test[g] == base[g]:
                n += 1

    return n/m



