import pandas as pd
import numpy as np

for i in range(11):
    j = 3
    if j == 1:
        snum = 500
    elif j == 2:
        snum = 1000
    elif j == 3:
        snum = 5000
    if i == 0:
        continue
    part_path = 'D:/data/alarm10_data/Alarm10_s'
    txt = np.loadtxt(part_path+str(snum)+"_v"+str(i)+'.txt')
    txtDF = pd.DataFrame(txt)
    txtDF.to_csv(part_path+str(snum)+"_v"+str(i)+'.csv', index=False)