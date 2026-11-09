# batch_rnx2rtkp - run multiple simultaneous solutions of rnx2rtkp in Windows

import os
from os.path import join, isfile, isdir
import subprocess
import psutil
import time
import android_rinex.gnsslogger_to_rnx as rnx


################## INPUT PARAMETERS ###############################################

# set location of data and rnx2rtkp executable
datapath = r'C:\gps\rtklib\bins\demo5_b34e_phone\Google\test'
binpath  = r'C:\gps\rtklib\bins\demo5_b34e_phone\rnx2rtkp.exe'

# Set data set choices
PHONES = ['Pixel4', 'Pixel4XLModded', 'Pixel4XL', 'Pixel4Modded', 'Pixel5', 'Mi8', 'SamsungS20Ultra']
OVERWRITE_DATAGNSS = True
OVERWRITE_SOL = True

# set input/output files (in datapath/PHONE)
cfgs = ['ppk_slac']  # list of config files to run ('.conf' appended to each file name in list)
baseFile = 'slac*.*o' # base files with wild cards
navFile = 'BRDM*.rnx' # navigation files with wild cards
outFile = 'out'  # out file name will be appended with config file 

# set maximum number of simultaneous occurences of rnx2rtkp
max_windows = 10 

################################################################################


class args:
    def __init__(self):
        # Input parameters for conversion to rinex
        self.slip_mask = 0
        self.fix_bias = True
        self.timeadj = 1e-7
        self.pseudorange_bias = 0
        self.filter_mode = 'sync'
        # Header values for rinex files
        self.marker_name = 'UNK'
        self.observer = 'UNK'
        self.agency = 'UNK'
        self.receiver_number = 'UNK'
        self.receiver_type = 'UNK'
        self.receiver_version = 'UNK'
        self.antenna_number = 'UNK'
        self.antenna_type = 'UNK'

ARGS = args()

# get list of current threads running in Windows
current_process = psutil.Process()
num_start = len(current_process.children())

# get list of data sets in data path
datasets = os.listdir(datapath)

# loop through data set folders
for dataset in datasets:
    for phone in PHONES:
        rovFile = phone + '_GnssLog.obs'
        rawFile = phone + '_GnssLog.txt'
        folder = join(datapath, dataset, phone)
          
        # skip if no folder for this phone
        if not isdir(folder):  
            continue
          
        print('%s_%s' % (dataset, phone)) 
        os.chdir(folder)
    
        if OVERWRITE_DATAGNSS or not isfile(rovFile):
            # convert raw data file to rinex
            args.input_log = rawFile
            args.output = rovFile
            print('  ', args.input_log, ' -> ', args.output)
            rnx.convert2rnx(ARGS)
    
            
        for cfg in cfgs:
            if OVERWRITE_SOL or not isfile(outFile + '_' + cfg + '.pos'):
                # create command to run solution
                rtkcmd='%s -x 0 -y 2 -k ../../../%s.conf -o %s.pos %s ..\%s ..\%s' % \
                    (binpath, cfg, outFile + '_' + cfg, rovFile, baseFile, navFile)
                
                # run command
                print('    ', cfg)
                subprocess.Popen(rtkcmd)
    
    
        # if max windows open, wait for one to close
        while len(current_process.children())-num_start >= max_windows:
            time.sleep(1) #wait here for existing window to close

# Wait for all solutions to finish
print('Waiting for solutions to complete ...')  
while len(current_process.children())-num_start > 0:
    pass #wait here if max windows open        
print('Done')
