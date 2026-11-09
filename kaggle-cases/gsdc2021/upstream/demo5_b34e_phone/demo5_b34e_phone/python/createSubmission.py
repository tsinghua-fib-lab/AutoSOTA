# convert baseline file into submission file

import numpy as np
import os

########### Input parameters ###############################

# specify data locations
BASELINE_FILE = 'baseline_locations_merged_PPK_test.csv'
OUT_FILE = 'submission_test_PPK.csv'
datapath = r'C:\gps\rtklib\bins\demo5_b34e_phone\Google'


############################################################


os.chdir(datapath)

# load baseline data 
d = np.genfromtxt(BASELINE_FILE, delimiter=',',invalid_raise=False, skip_header=1, dtype=str)

# save results to file
fout =open( OUT_FILE,'w')
fout.write('phone,millisSinceGpsEpoch,latDeg,lngDeg\n')
for i in range(len(d)):
    # write results to combined file
    fout.write('%s, %s, %3.15f, %3.15f\n' % (d[i,-1], d[i,2], float(d[i,3]), float(d[i,4])))
fout.close()