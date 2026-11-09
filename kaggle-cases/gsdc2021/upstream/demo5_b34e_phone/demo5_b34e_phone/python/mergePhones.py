''' 
Calculate fused position for one phone then interpolate results back to other
phones on that ride
'''

import numpy as np
import os


########### Input parameters ###############################

# specify data locations
datapath = r'C:\gps\rtklib\bins\demo5_b34e_phone\Google'
DATA_SET = 'test'
BASELINE_PPK_FILE = 'baseline_locations_test_PPK.csv'
    
PPP_STD = 0.0 # standard deviation in meters to use for PPP data (0=disable)


############################################################

# load and parse RTKLIB and Google baseline data 
os.chdir(datapath)
BASELINE_PPP_FILE = 'baseline_locations_' + DATA_SET + '.csv'
d1 = np.genfromtxt(BASELINE_PPK_FILE, delimiter=',',invalid_raise=False, skip_header=1, dtype=str)
d2 = np.genfromtxt(BASELINE_PPP_FILE, delimiter=',',invalid_raise=False, skip_header=1, dtype=str)

times = d1[:,2].astype(np.uint64)
llhs = d1[:,3:6].astype(float)
llhs_ppp = d2[:,3:6].astype(float)
stds = d1[:,6:9].astype(float)
qs = d1[:,9].astype(int)
nss = d1[:,10].astype(int)

trips = np.unique(d1[:,0])
phones = np.unique(d1[:,1])
num_ph = len(phones)

# loop through trips
for trip in trips:
    ixd = np.where(d1[:,0] == trip)[0]
    ts = np.unique(times[ixd])
    ixp = []
    for phone in phones:
        ixp.append(ixd[np.where(d1[ixd,1] == phone)[0]])
    ix = 0
    llhs_i = np.zeros((2 * num_ph,len(ts),3))
    stds_i = np.zeros((2 * num_ph,len(ts),3))
    wghts_i = np.zeros((2 * num_ph,len(ts),3))
    qs_i = np.zeros((2 * num_ph,len(ts))).astype(int)
    nss_i = np.zeros((2 * num_ph,len(ts))).astype(int)
    for i in range(num_ph):
        if len(ixp[i]) > 0:
            # interpolate this phone to all timestamps
            for j in range(3):
                llhs_i[ix,:,j] = np.interp(ts, times[ixp[i]], llhs[ixp[i],j])
                stds_i[ix,:,j] = np.interp(ts, times[ixp[i]], stds[ixp[i],j],
                                           left=1000, right=1000)
                wghts_i[ix,:,j] = 1 / stds_i[ix,:,j]**2
                # throw out samples with zero or very low weights
                ii = np.where((wghts_i[ix,:,j] < 0.00001))[0] # | (nss_i < 2))
                wghts_i[ix,ii,j] = 1e-9
            qs_i[ix,:] = np.interp(ts, times[ixp[i]], qs[ixp[i]])
            nss_i[ix,:] = np.interp(ts, times[ixp[i]], nss[ixp[i]])
            ix += 1
            # include PPP data if not disabled
            if PPP_STD > 0:
                for j in range(3):
                    llhs_i[ix,:,j] = np.interp(ts, times[ixp[i]], llhs_ppp[ixp[i],j])
                    stds_i[ix,:,j] = PPP_STD
                    wghts_i[ix,:,j] = 1 / stds_i[ix,:,j]**2
                ix += 1

    # calc weighted average locations of phones for all timestamps
    llhs_w = np.average(llhs_i[:ix,:,:], weights=wghts_i[:ix,:,:],axis=0)
    stds_w = np.average(stds_i[:ix,:,:], weights=wghts_i[:ix,:,:],axis=0)
    nss_w = np.average(nss_i[:ix,:], weights=wghts_i[:ix,:,0],axis=0)
    qs_w = np.average(qs_i[:ix,:], weights=wghts_i[:ix,:,0],axis=0)
    # put weighted average back into individual phones
    ix = 0
    for i in range(num_ph):
        if len(ixp[i]) > 0:
            ii = np.searchsorted(ts,times[ixp[i]])
            llhs[ixp[i],:] = llhs_w[ii,:]
            stds[ixp[i],:] = stds_w[ii,:]
            qs[ixp[i]] = qs_w[ii]
            nss[ixp[i]] = nss_w[ii]
            ix += 1
    #break

# save results to file
fout =open('baseline_locations_merged_PPK_' + DATA_SET + '.csv','w')
fout.write('collectionName,phoneName,millisSinceGpsEpoch,latDeg,lngDeg, \
           heightAboveWgs84EllipsoidM,stde,stdn,stdu,Q,ns,phone\n')
for i in range(len(llhs)):
    # write results to combined file
    fout.write('%s,%s,%s,%13.9f,%13.9f,%6.3f,%6.3f,%6.3f,%6.3f,%s,%s,%s\n' % (d1[i,0], d1[i,1], d1[i,2],
            llhs[i,0], llhs[i,1], llhs[i,2], stds[i,0], stds[i,1], stds[i,2], qs[i], nss[i], d1[i,-1]))
fout.close()

    