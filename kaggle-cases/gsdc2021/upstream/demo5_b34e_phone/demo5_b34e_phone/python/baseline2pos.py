'''
baseline2pos.py - convert Google baseline csv file to a folder of 
RTKLIB solution files
'''

import numpy as np
import os

########### Input parameters ###############################

# specify data locations
BASELINE_FILE = 'baseline_locations_merged_test_12_30.csv'
datapath = r'C:\gps\data\cellphone\Google'


############################################################



# create solution folder and header
os.chdir(datapath)
solpath = os.path.join('solutions', BASELINE_FILE[:-4])
os.makedirs(solpath, exist_ok=True)

hdr = '%  GPST          latitude(deg) longitude(deg)  height(m)   ' \
                   'Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio\n'

# load baseline data 
d = np.genfromtxt(BASELINE_FILE, delimiter=',',invalid_raise=False, skip_header=1, dtype=str)
times = d[:,2].astype(np.uint64)
llhs = d[:,3:6].astype(float)
if d.shape[1] > 7:
    stds = d[:,6:9].astype(float)
    qs = d[:,9].astype(int)
    nss = d[:,10].astype(int)
else:
    stds=np.ones((len(d), 3)) * 1.0
    qs=np.ones((len(d))) * 6.0
    nss=np.ones((len(d))) * 10.0

trips = np.unique(d[:,0])
phones = np.unique(d[:,1])
num_ph = len(phones)

# loop through trips
os.chdir(solpath)
for trip in trips:
    ixd = np.where(d[:,0] == trip)[0]
    for phone in phones:
        ix = ixd[np.where(d[ixd,1] == phone)[0]]
        if len(ix) == 0:
            continue
        # interpolate to even seconds for comparison to ground truth
        ts = np.arange(int(times[ix[0]]/1000),int(times[ix[-1]]/1000))
        llhs_i = []; stds_i = []
        for j in range(3):
            llhs_i.append(np.interp(ts, times[ix]/1000, llhs[ix,j]))
            stds_i.append(np.interp(ts, times[ix]/1000, stds[ix,j]))
        qsi = np.interp(ts, times[ix]/1000, qs[ix])
        nssi = np.interp(ts, times[ix]/1000, nss[ix])
        wk = (ts / (7 * 24 * 3600)).astype(int)
        sec = ts - wk*(7 * 24 *3600)
            
        # save results to file
        fout =open(trip + '_' + phone + '.pos','w')
        fout.write(hdr)
        for i in range(len(ts)):
            # write results to combined file
            fout.write('%4d %10.3f %14.9f %14.9f %10.4f %3d %3d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %6.2f %6.1f\n' % 
                       (wk[i], sec[i], llhs_i[0][i], llhs_i[1][i], llhs_i[2][i], 
                        qsi[i], nssi[i], stds_i[0][i], stds_i[1][i], stds_i[2][i],
                        0, 0, 0, 0, 0))
        fout.close()


