# create single csv file from all ground truth files

import os
from os.path import join, isfile
import math
import time

########### Input parameters ###############################

datapath = r'C:\gps\rtklib\bins\demo5_b34e_phone\Google'


############################################################



GPS_TO_UTC = 315964782  # second

# open output file
os.chdir(datapath)
fout =open('../ground_truths_train.csv','w')
fout.write('collectionName,phoneName,millisSinceGpsEpoch,latDeg,lngDeg,heightAboveWgs84EllipsoidM,phone\n')

# get list of data sets in data path
datasets = os.listdir(datapath)

# loop through data set folders
for dataset in datasets:
    phones = os.listdir(join(datapath,dataset))
    for phone in phones:
        folder = join(datapath, dataset, phone)
        if isfile(folder):
            continue
        
        csv_file = join(folder, 'ground_truth.csv')
        if not isfile(csv_file):
            continue

        # parse ground truth file
        with open(csv_file) as f:
            lines = f.readlines()[1:]
        flag = 0
        for line in lines:
            d = line.split(',')
            
            # convert time stamp
            utc_sec = math.modf(float(d[2]) / 1000 + GPS_TO_UTC)
            t = time.gmtime(utc_sec[1])
            t = time.strftime('%Y-%m-%d %H:%M:%S', t)
            if flag == 0:
                print('%20s,%16s,%s' % (d[0], d[1], t))
                flag = 1
            # write results to combined file
            fout.write('%s,%s,%s,%s,%s,%5.2f,%s,%s\n' % (d[0], d[1], d[2], d[3],d[4],
                float(d[5]) - 61, d[0] + '_' + d[1], t))
        print('%20s %16s %s' % ('', '', t))
        
fout.close()