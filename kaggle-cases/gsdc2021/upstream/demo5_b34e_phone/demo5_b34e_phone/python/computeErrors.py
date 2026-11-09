# compute difference between baseline file and ground truth (Training data only)

import numpy as np
from os.path import join
from utm import from_latlon as ll2en

########### Input parameters ###############################

# specify data locations
BASELINE_FILE = 'baseline_locations_train.csv'
datapath = r'C:\gps\rtklib\bins\demo5_b34e_phone\Google'

CSV_FORMAT = True


############################################################

def show_errs(t_utm, r_utm, hvar, title):
    hErrs = np.sqrt((t_utm[0] - r_utm[0])**2 + (t_utm[1] - r_utm[1])**2)
    hErrs.sort(); 
    h50 = np.median(hErrs)
    h95 = hErrs[round(len(t_utm[0])*0.95)-1] 
    if np.mean(hvar) == 0:
        vacc = np.convolve(t_utm[2], [-1,2,-1])
        vacc[:2] = 0; vacc[-2:] = 0
        hstd = np.median(abs(vacc))
    else:
        hstd = np.sqrt(np.median(hvar))
    if CSV_FORMAT:
        print('%-36s, %.3f, %.3f, %.3f' % (title, h50, h95, hstd))
    else:
        print('%-36s: h50=%.3f h95=%.3f stde=%.3f' % (title, h50, h95, hstd))
    return((h50, h95, hstd))


# load baseline and ground truth data
test_s = np.genfromtxt(join(datapath, BASELINE_FILE), delimiter=',',
                         invalid_raise=False, skip_header=1, dtype=str)
ref_s = np.genfromtxt(join(datapath, 'ground_truths_train.csv'), delimiter=',',
                         invalid_raise=False, skip_header = 1, dtype=str)
if test_s.shape[1] >= 10: # includes stdevs
    test = test_s[:,2:9].astype(float)
    hvar = test[:,4]**2 + test[:,5]**2
else:    
    test = test_s[:,2:6].astype(float)
    hvar = np.zeros(len(test))
ref = ref_s[:,2:6].astype(float)

# check that time stamps match
if sum(test[:,0] - ref[:,0]) != 0:
    print('Time stamp mismatch')

# convert from llh to utm
t_utm = np.array((*ll2en(test[:,1], test[:,2])[0:2], test[:,3]))
r_utm = np.array((*ll2en(ref[:,1], ref[:,2])[0:2], ref[:,3]))

# calc errs for full set
print('')
show_errs(t_utm, r_utm, hvar, 'All datasets')
#calc_verrs(test, ref)

# calc errs for each phone
phones = np.unique(test_s[:,1])
print('')
errs = []
for phone in phones:
    ix = np.where(test_s[:,1] == phone)[0]
    errs.append(show_errs(t_utm[:,ix], r_utm[:,ix], hvar[ix], phone))
    
errs = np.array(errs)
h50 = np.mean(errs[:,0])
h95 = np.mean(errs[:,1])
print('\nMean of phones: h50=%.3f h95=%.3f h50_h95=%.3f' % (h50, h95, (h50 + h95) / 2))

# calc errs for each data set
datasets = np.unique(test_s[:,-1])
print('')
errs = []
for dataset in datasets:
    ix = np.where(test_s[:,-1] == dataset)[0]
    errs.append(show_errs(t_utm[:,ix], r_utm[:,ix], hvar[ix], dataset))
