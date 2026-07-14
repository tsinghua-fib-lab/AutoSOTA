import os
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
import numpy as np
import matplotlib.pyplot as plt

conditions=['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']

def get_vector_array(coords):
    """function for defining the vector features from each array of coordinates"""
    diff = np.diff(coords, axis=0)
    return diff


def fit_pca(rates, day, conditions, filter_data=True, pca_n=5):
    pos = []
    # loop over each condition on that day
    for c, cond in enumerate(conditions):

        # go cue at 500ms (500ms / 20ms bin = 250)
        # only take rates from bin 10 onwards
        data = rates[day][cond][:, :, 25:]

        # loop over all trials
        for t in range(data.shape[0]):

            # extract trial
            trial = data[t, :, :]

            # smooth trial over time
            if filter_data:
                trial = savgol_filter(trial, 9, 2)

            # store each trial as time x channels
            pos.append(trial.T)

    # stacking all trials into a single array (time x channels)
    pos = np.vstack(pos)

    # fit PCA to all data across all conditions on a given day simultaneously
    pca = PCA(n_components=pca_n)
    pca.fit(pos)
    
    return pca


def format_data(rates, trial_ids, day, conditions, pca=None, filter_data=True, go_cue=25, stack=True):
    # create empty list of lists for each condition
    pos = [[] for u in range(len(conditions))]
    vel = [[] for u in range(len(conditions))]
    timepoints = [[] for u in range(len(conditions))]
    condition_labels = [[] for u in range(len(conditions))]
    trial_indexes = [[] for u in range(len(conditions))]

    # loop over conditions
    for c, cond in enumerate(conditions):

        # go cue at 500ms (500ms / 50ms bin = 10)
        data = rates[day][cond][:, :, go_cue:]

        # loop over all trials
        for t in range(data.shape[0]):

            # extract trial
            trial = data[t, :, :]

            # smooth trial over time
            if filter_data:
                trial = savgol_filter(trial, 9, 2)

            # apply the transformation to a single trial
            if pca is not None:
                trial = pca.transform(trial.T)
            else:
                trial = trial.T

            # take all points except last
            pos[c].append(trial[:-1, :])

            # extract vectors between coordinates
            vel[c].append(get_vector_array(trial))
            timepoints[c].append(np.linspace(0, trial.shape[0] - 2, trial.shape[0] - 1))
            condition_labels[c].append(np.repeat(c, trial.shape[0] - 1))

            # adding trial id info (to match with kinematics decoding later)
            ind = np.repeat(trial_ids[day][cond][t], trial.shape[0] - 1)
            trial_indexes[c].append(ind)

    # stack the trials within each condition
    if stack:
        pos = [np.vstack(u) for u in pos]  # trials x time x channels
        vel = [np.vstack(u) for u in vel]  # trials x time x channels
        timepoints = [np.hstack(u) for u in timepoints]
        condition_labels = [np.hstack(u) for u in condition_labels]
        trial_indexes = [np.hstack(u) for u in trial_indexes]
        
    return pos, vel, timepoints, condition_labels, trial_indexes


def train_OLE(data, trial_ids, representation='lfads_factors'):
    
    X, Z = [], []
    unique_trial_ids = np.unique(trial_ids)
    for tr in unique_trial_ids:
        X.append(data[tr]['kinematics'])
        Z.append(data[tr][representation])

    X, Z = np.hstack(X), np.hstack(Z)

    X = X[:4,:] # take first four rows of kinematics (pos_x, pos_y, vel_x, vel_y)
    Z = np.vstack([Z,np.ones(Z.shape[1])])
    
    out  = np.linalg.lstsq(Z.T, X.T, rcond=None)
    Lw = out[0].T
    
    return Lw
    
    
def decode_kinematics(data, L, dt=20, representation='lfads_factors'):
    
    trial_emb = data[representation] # get trial embedding
    trial_kinematics = data['kinematics'] # get kinematics associated with trial

    trial_emb = np.vstack([trial_emb, np.ones(trial_emb.shape[1])])

    # empty array for decoding predictions
    trial_pred = np.empty([4, trial_kinematics.shape[1]])
    trial_pred[:] = np.nan

    trial_pred[:2,0] = trial_kinematics[:2,0] #first two entries are the x,y coordinates

    # predict velocity 
    z = np.matmul(L, trial_emb[:,0]); # decode
    trial_pred[[2,3],0] = z[[2,3]] #velocities

    # loop over each time point in trial
    for nt in range(1,trial_kinematics.shape[1]):

        neural = trial_emb[:,nt] # next point of embedding
        z = np.matmul(L, neural) # decode

        #trial_pred[:2,nt] = (1-alpha) * z[:2] + alpha * (trial_pred[:2,nt-1] + z[[2,3]]*dt/1000)
        trial_pred[:2,nt] = trial_pred[:2,nt-1] + z[[2,3]]*dt/1000
        trial_pred[[2,3],nt] = z[[2,3]]

    return trial_pred

# define a function for computing R-squared of kinematics
def correlation(data, trial_ids, representation='lfads_factors'):
    
    X, Z = [], []
    for tr in trial_ids:
        X.append(data[tr]['kinematics'])
        Z.append(data[tr][representation][:,:])

    X, Z = np.hstack(X), np.hstack(Z)

    r2_vel = np.mean([calcR2(X[2,:], Z[2,:]), calcR2(X[3,:], Z[3,:])])
    r2_pos = np.mean([calcR2(X[0,:], Z[0,:]), calcR2(X[1,:], Z[1,:])])
   
    return r2_pos, r2_vel

def calcR2(data, model):
    
    datavar = sum((data-np.mean(data))**2);
    errorvar = sum((model-data)**2);
    r2 = 1-errorvar/datavar   
    
    return r2

def plot_kinematics(data, session, trial_ids, representation='lfads_factors', ax=None, sz = 140):
    colors = plt.cm.viridis(np.linspace(0,1,7))        
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    for c,cond in enumerate(conditions):   
        for t in trial_ids:
            if data[session][t]['condition']==cond:
                meh = data[session][t][representation]
                ax.plot(meh[0,:],meh[1,:],c=colors[c])
                
    ax.set_title(representation)
    ax.set_xlim([-sz, sz])
    ax.set_ylim([-sz, sz])
    ax.set_axis_off()
    
    return ax

def fit_classifier(data, conditions, trials, representation):
    samples = []; labels = [];
    for c,cond in enumerate(conditions):   
        for t in trials:
            if data[t]['condition']==cond:
                sample = data[t][representation][:2,:].flatten()
                samples.append(sample)
                labels.append(c)

    X = np.vstack(samples)
    y = np.array(labels)
    clf = SVC().fit(X, y)

    return clf

def transform_classifier(clf, data, conditions, trials, representation):
    samples = []; labels = [];
    for c,cond in enumerate(conditions):   
        for t in trials:
            if data[t]['condition']==cond:
                sample = data[t][representation][:2,:].flatten()
                samples.append(sample)
                labels.append(c)

    X = np.vstack(samples)
    y = np.array(labels)
    return clf.score(X, y)