# create csv file from all ground truth files

import os
from os.path import join, isfile
import numpy as np

########### Input parameters ###############################

DATA_SET = 'test'
SOL_FILE = 'out_ppk_slac.pos'
datapath = r'C:\gps\rtklib\bins\demo5_b34e_phone\Google'


############################################################


# get timestamps from existing baseline file
os.chdir(datapath)
base_txt = np.genfromtxt('baseline_locations_' + DATA_SET + '.csv', delimiter=',',
                         invalid_raise=False, skip_header=1, dtype=str)
msecs_b = base_txt[:,2].astype(np.int64)
phones_b = base_txt[:,6]

# open output file
fout =open('baseline_locations_' + DATA_SET + '_PPK.csv','w')
fout.write('collectionName,phoneName,millisSinceGpsEpoch,latDeg,lngDeg \
           heightAboveWgs84EllipsoidM,stde, stdn, stdu, Q, ns, phone\n')

# get list of data sets in data path
os.chdir(join(datapath, DATA_SET))
trips = os.listdir()

# loop through data set folders
ix_b = 0
for trip in trips:
    if isfile(trip):
        continue
    phones = os.listdir(trip)
    # loop through phone folders
    for phone in phones:
        # check for valid folder and file
        folder = join(trip, phone)
        if isfile(folder):
            continue
        trip_phone = trip + '_' + phone
        print(trip_phone)
        sol_path = join(folder, SOL_FILE)
        if not isfile(sol_path):
            print('File not found: ', sol_path)
            continue

        # parse solution file
        fields = np.genfromtxt(sol_path, invalid_raise=False, skip_header=25)
        pos = fields[:,2:5]
        qs = fields[:,5].astype(int)
        nss = fields[:,6].astype(int)
        acc = fields[:,7:10]
        msecs = (1000 * (fields[:,0] * 7 * 24 * 3600 + fields[:,1])).astype(np.int64)
        
        # interpolate to baseline timestamps to fill in missing samples
        ix_b = np.where(phones_b == trip_phone)[0]
        llhs = []; stds = []
        for j in range(3):
            llhs.append(np.interp(msecs_b[ix_b], msecs, pos[:,j]))
            stds.append(np.interp(msecs_b[ix_b], msecs, acc[:,j],
                                left=1000, right=1000))
        qsi = np.interp(msecs_b[ix_b], msecs, qs)
        nssi = np.interp(msecs_b[ix_b], msecs, nss)

        # write results to combined file
        for i in range(len(ix_b)):
            fout.write('%s,%s,%d,%s,%s,%s,%6.3f,%6.3f,%6.3f,%d,%d,%s\n' % (trip, phone, msecs_b[ix_b[i]],
                    llhs[0][i], llhs[1][i], llhs[2][i], stds[0][i], stds[1][i], 
                    stds[2][i], qsi[i], nssi[i], trip_phone))
        
fout.close()
            
