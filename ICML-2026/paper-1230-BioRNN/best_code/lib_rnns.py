import numpy as np
import matplotlib.pyplot as plt
import torch

#%%
def set_plot(ll = 7):
    plt.style.use('ggplot')

    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2  # height in inches
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True
     
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markeredgewidth'] = 0.003
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['font.size'] = 14#9
    plt.rcParams['legend.fontsize'] = 11#7.
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'
    
    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = 14#9
    plt.rcParams['xtick.labelsize'] = 11#7
    plt.rcParams['ytick.labelsize'] = 11#7
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    
    plt.rcParams['font.sans-serif'] = 'Arial'
    
    cls = np.zeros((ll,3))
    
    cl11 = np.array((102, 153, 255))/255.
    cl12 = np.array((53, 153, 53))/255.
    
    cl21 = np.array((255, 204, 51))/255.
    cl22 = np.array((204, 0, 0))/255.
    
    if ll==7:
        cls[0,:] = 0.4*np.ones((3,))
        
        cls[1,:] = cl11
        cls[2,:] = 0.5*cl11+0.5*cl12
        cls[3,:] = cl12
        
        cls[4,:] = cl21
        cls[5,:] = 0.5*cl21+0.5*cl22
        cls[6,:] = cl22
    elif ll == 5:
        cls[0,:] = 0.4*np.ones((3,))    
        
        cls[2,:] = cl12
        
        cls[3,:] = cl21
        
        cls[4,:] = cl22    
    return(cls)
    
def set_plot2(ll = 7):
    plt.style.use('ggplot')

    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2  # height in inches
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True
     
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markeredgewidth'] = 0.003
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['font.size'] = 14#9
    plt.rcParams['legend.fontsize'] = 11#7.
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'
    
    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = 14#9
    plt.rcParams['xtick.labelsize'] = 11#7
    plt.rcParams['ytick.labelsize'] = 11#7
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    
    plt.rcParams['font.sans-serif'] = 'Arial'
    
    cls = np.zeros((ll,3))
    
    cl11 = np.array((102, 153, 255))/255.
    cl12 = np.array((53, 153, 53))/255.
    
    cl21 = np.array((255, 204, 51))/255.
    cl22 = np.array((204, 0, 0))/255.
    
    if ll==7:
        cls[0,:] = 0.4*np.ones((3,))
        
        cls[1,:] = cl11
        cls[2,:] = 0.5*cl11+0.5*cl12
        cls[3,:] = cl12
        
        cls[4,:] = cl21
        cls[5,:] = 0.5*cl21+0.5*cl22
        cls[6,:] = cl22

        cls = cls[1:]
        cls = cls[::-1]
        
        c2 = [67/256, 90/256, 162/256]
        c1 = [220/256, 70/256, 51/256]
        cls[0,:]=c1
        cls[5,:]=c2
    elif ll == 5:
        cls[0,:] = 0.4*np.ones((3,))    
        
        cls[2,:] = cl12
        
        cls[3,:] = cl21
        
        cls[4,:] = cl22    
    return(cls)
# Create input 
    
def plot_readout(net_low,  time, Nt, time_com, amps, R_on, dt, string1, string2,
                 trials=100, plot_trace=True, plot_psycho=True, pl_min = 60, pl_max = 1350, mean=False, png=False, new=False):
    cls = set_plot2()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    Tp = np.zeros(len(amps))
    Tp_sd  = np.zeros(len(amps)) 
    thres = 0.35
    train_int = (0.5+thres)*time_com*dt
    for xx in range(len(amps)):
    
        trials = 100#100
        
        input_tr, output_tr, mask_tr, ct_tr, ct2_tr = create_inp_out2(trials, Nt, time_com, amps, R_on,  0., 
                                                                      just=xx, perc = 0.0)
    
        outp, traj = net_low.forward(input_tr, return_dynamics=True)
        outp = outp.detach().numpy()
        traj = traj.detach().numpy()
        
        avg_outp = np.mean(outp[:,:,0],0)
        if new==False:
            ax.plot(-R_on*dt/1000+time*dt/1000, avg_outp, color=cls[xx,:])
        else:
            ax.plot(-R_on*dt/1000+time*dt/1000, avg_outp, color=cls[xx,:], lw=2.)
        
        if mean == False:
            if new ==False:
                ax.plot(-R_on*dt/1000+time*dt/1000, outp[:,:,0].T, color=cls[xx,:], lw= 0.1)
            else:
                ax.plot(-R_on*dt/1000+time*dt/1000, outp[:,:,0].T, color=cls[xx,:], lw= 1.2)
        if new==False:
            ax.plot(time[mask_tr.detach().numpy()[xx,:,0]>0.]*dt/1000-R_on*dt/1000, output_tr.detach().numpy()[xx,mask_tr.detach().numpy()[xx,:]>0.], '--k', lw=1.0)
        else:
            ax.plot(time[mask_tr.detach().numpy()[xx,:,0]>0.]*dt/1000-R_on*dt/1000, output_tr.detach().numpy()[xx,mask_tr.detach().numpy()[xx,:]>0.], '--', c = [0.3, 0.3, 0.3], lw=1., zorder=2)
        
        outp_thres = outp[:,:,0]-thres
        time_cross = np.zeros(trials)
        for t in range(trials):
            t_ser = outp_thres[t,:]
            if len(time[t_ser>0])>0:
                time_cross[t] = (time[t_ser>0][0]-R_on)*dt
            else:
                time_cross[t] = np.nan
                
        Tp[xx] = np.mean(time_cross)-R_on*10
        Tp_sd[xx] = np.std(time_cross)-R_on*10
    xl = np.min( [time[-1]*dt/1000, 1.4*(np.max(time_com)+R_on)*dt/1000])
    ax.set_xlim([-0.6,  xl])    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('time after Set (s)')
    plt.ylabel('read out')
    if png==False:
        plt.savefig(string1)
    else:
        plt.savefig(string1, dpi=1200)
    plt.show()
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for xx in range(len(amps)):
        plt.errorbar(train_int[xx], Tp[xx], yerr = Tp_sd[xx], fmt='o', markersize=10, color=cls[xx,:])
    plt.plot(train_int, train_int, 'k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('trained interval [ms]')
    plt.ylabel(r'$t_p$ [ms]')
    if png==False:
        plt.savefig(string2)
    else:
        plt.savefig(string2, dpi=1200)
    plt.show()
    #plt.plot([pl_min, pl_max], [pl_min, pl_max], 'k')
    return(Tp, Tp_sd, train_int)
    
def plot_readout_thesis(net_low,  time, Nt, time_com, amps, R_on, dt, string1, string2,
                 trials=100, plot_trace=True, plot_psycho=True, pl_min = 60, pl_max = 1350, mean=False, png=False, new=False):
    cls = set_plot2()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    Tp = np.zeros(len(amps))
    Tp_sd  = np.zeros(len(amps)) 
    thres = 0.35
    train_int = (0.5+thres)*time_com*dt
    for xx in range(len(amps)):
    
        trials = 100#100
        
        input_tr, output_tr, mask_tr, ct_tr, ct2_tr = create_inp_out2(trials, Nt, time_com, amps, R_on,  0., 
                                                                      just=xx, perc = 0.0)
    
        outp, traj = net_low.forward(input_tr, return_dynamics=True)
        outp = outp.detach().numpy()
        traj = traj.detach().numpy()
        if xx ==len(amps)-1:
            ax.plot(time[mask_tr.detach().numpy()[xx,:,0]>0.]*dt/1000-R_on*dt/1000, output_tr.detach().numpy()[xx,mask_tr.detach().numpy()[xx,:]>0.], color=cls[xx,:], lw=2.1*1.5, zorder=3)
        else:
            ax.plot(time[mask_tr.detach().numpy()[xx,:,0]>0.]*dt/1000-R_on*dt/1000, output_tr.detach().numpy()[xx,mask_tr.detach().numpy()[xx,:]>0.], color=cls[xx,:], lw=2.1, alpha=0.5)
       
        outp_thres = outp[:,:,0]-thres
        time_cross = np.zeros(trials)
        for t in range(trials):
            t_ser = outp_thres[t,:]
            if len(time[t_ser>0])>0:
                time_cross[t] = (time[t_ser>0][0]-R_on)*dt
            else:
                time_cross[t] = np.nan
                
        Tp[xx] = np.mean(time_cross)-R_on*10
        Tp_sd[xx] = np.std(time_cross)-R_on*10
    xl = np.min( [time[-1]*dt/1000, 1.4*(np.max(time_com)+R_on)*dt/1000])
    ax.set_xlim([-0.6,  xl])    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('time after Set (s)')
    plt.ylabel('read out')
    if png==False:
        plt.savefig(string1)
    else:
        plt.savefig(string1, dpi=1200)
    plt.show()
        
    return(Tp, Tp_sd, train_int)
    
def plot_readout2(net_low,  time, Nt, time_com, amps, R_on, dt, string1, string2,
                 trials=100, plot_trace=True, plot_psycho=True, pl_min = 60, pl_max = 1350, mean=False):
    cls = set_plot()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #Tp = np.zeros(len(amps))
    #Tp_sd  = np.zeros(len(amps)) 
    #thres = 0.35
    #train_int = (0.5+thres)*time_com*dt
    for xx in range(len(amps)):
    
        trials = 100
        
        input_tr, output_tr, mask_tr, ct_tr, ct2_tr = create_inp_out2(trials, Nt, time_com, amps, R_on,  0., 
                                                                      just=xx, perc = 0.0)
    
        outp, traj = net_low.forward(input_tr, return_dynamics=True)
        outp = outp.detach().numpy()
        traj = traj.detach().numpy()
        
        #dist = np.diff(traj,axis=1)
        speed = np.sqrt(np.sum(np.diff(traj,axis=1)**2,-1))
        avg_outp = np.mean(speed,0)#np.mean(np.diff(outp[:,:,0])/dt,0)
        ax.plot(time[:-1]*dt, avg_outp, color=cls[xx,:])
        if mean == False:
            #ax.plot(time[:-1]*dt, np.diff(outp[:,:,0]).T, color=cls[xx,:], lw= 0.1)
            ax.plot(time[:-1]*dt, speed[:,:,0].T, color=cls[xx,:], lw= 0.1)
        #ax.plot(time[mask_tr.detach().numpy()[xx,:,0]>0.]*dt, output_tr.detach().numpy()[xx,mask_tr.detach().numpy()[xx,:]>0.], '--k', lw=1.0)
        
        #outp_thres = outp[:,:,0]-thres
        #time_cross = np.zeros(trials)
        #for t in range(trials):
        #    t_ser = outp_thres[t,:]
        #    if len(time[t_ser>0])>0:
        #        time_cross[t] = (time[t_ser>0][0]-R_on)*dt
        #    else:
        #        time_cross[t] = np.nan
                
        #Tp[xx] = np.mean(time_cross)
        #Tp_sd[xx] = np.std(time_cross)
    xl = np.min( [time[-1]*dt, 1.4*(np.max(time_com)+R_on)*dt])
    ax.set_xlim([0,  xl])    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('time [ms]')
    ax.set_ylim([0,6])
    #plt.yscale('log')
    plt.ylabel('speed')
    plt.savefig(string1)
    plt.show()
        
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for xx in range(len(amps)):
#        plt.errorbar(train_int[xx], Tp[xx], yerr = Tp_sd[xx], fmt='o', markersize=10, color=cls[xx,:])
#    plt.plot(train_int, train_int, 'k')
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.yaxis.set_ticks_position('left')
#    ax.xaxis.set_ticks_position('bottom')
#    plt.xlabel('trained interval [ms]')
#    plt.ylabel(r'$t_p$ [ms]')
#    plt.savefig(string2)
#    plt.show()
    #plt.plot([pl_min, pl_max], [pl_min, pl_max], 'k')
    return()

def create_inp_out(trials, Nt, tss, amps, R_on, SR_on, just=-1,  perc = 0.2):
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 2))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    
    r_inp = np.ones((trials, Nt))
    #r2_inp = np.ones((trials, Nt))
    s_inp =  np.zeros((trials, Nt))
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have the set cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    
    rnd = np.zeros(trials)
    if SR_on>0:
        rnd = np.random.randint(-SR_on, SR_on, trials)

    for itr in range(trials):            
        if  ct2[itr]:
            maskt[itr,:,0] = (time>R_on+1+rnd[itr])*(time<np.max(tss)+R_on+1+rnd[itr])
        else:
            maskt[itr,:,0] = (time>R_on+1+rnd[itr])*(time<tss[ct[itr]]+R_on+1+rnd[itr])
            s_inp[itr, time>R_on+rnd[itr]] = 100.
            s_inp[itr, time>1+R_on+rnd[itr]] = 0.
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -0.9*strt, int(sum(maskt[itr,:,0])), endpoint=True)
        #Include zero read-out in cost function
        if ct2[itr]:
            maskt[itr,:,0] = (time<np.max(tss)+R_on+1+rnd[itr])
        else:
            maskt[itr,:,0] = (time<tss[ct[itr]]+R_on+1+rnd[itr])
        if just==-1:
            r_inp[itr, :] = amps[ct[itr]]*r_inp[itr,:]    
        
    if just>-1:
        r_inp = amps[just]*r_inp
    
    inputt[:,:,0] = r_inp #cue
    inputt[:,:,1] = s_inp #set
    #inputt[:,:,2] = r2_inp
    #outputt = strt*np.ones((trials, Nt, 1))
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2)
    
def create_inp_out2(trials, Nt, tss, amps, R_on, SR_on=80, just=-1,  perc = 0.2):
    '''
    Missing
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 2))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    
    r_inp = np.ones((trials, Nt))
    #r2_inp = np.ones((trials, Nt))
    s_inp =  np.zeros((trials, Nt))
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have the set cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    
    rnd = np.zeros(trials)
    if SR_on>0:
        rnd = np.random.randint(-SR_on, SR_on, trials)

    for itr in range(trials):            
        if  ct2[itr]:
            maskt[itr,:,0] = (time>R_on+1+rnd[itr])*(time<np.max(tss)+R_on+1+rnd[itr])
        else:
            maskt[itr,:,0] = (time>R_on+1+rnd[itr])*(time<tss[ct[itr]]+R_on+1+rnd[itr])
            mask_aft = time>=tss[ct[itr]]+R_on+1+rnd[itr]
            s_inp[itr, time>R_on+rnd[itr]] = 10.
            s_inp[itr, time>1+R_on+rnd[itr]] = 0.
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
        #Include zero read-out in cost function
        if ct2[itr]:
            maskt[itr,:,0] = (time<Nt)#np.max(tss)+R_on+1+rnd[itr])
        else:
            maskt[itr,:,0] = (time<Nt)#tss[ct[itr]]+R_on+1+rnd[itr])
        if just==-1:
            r_inp[itr, :] = amps[ct[itr]]*r_inp[itr,:]    
        
    if just>-1:
        r_inp = amps[just]*r_inp
    
    inputt[:,:,0] = r_inp
    inputt[:,:,1] = s_inp
    #inputt[:,:,2] = r2_inp
    #outputt = strt*np.ones((trials, Nt, 1))
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2)

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

def Phi (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Prime (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def self_con(kappas, rho, means_z, var_z, w_z, means_x, var_x, w_x):
    G = np.zeros(len(kappas))
    for ikap, kap in enumerate(kappas):
        tot_w = np.zeros(len(w_z)*len(w_x))
        tot_mu = np.zeros(len(tot_w))
        tot_delta = np.zeros(len(tot_w))
        tot_a     = np.zeros(len(tot_w))
        tot_sig   = np.zeros(len(tot_w))
        fact1   = np.zeros(len(tot_w))
        fact2   = np.zeros(len(tot_w))
        
        
        
        count= 0
        for k in range(len(w_z)):
            for l in range(len(w_x)):
                tot_w[count] = w_z[k]*w_x[l]
                tot_mu[count] = kap*(means_x[l]+rho*means_z[k])
                tot_delta[count] = kap**2*(var_x[l]+rho**2*var_z[k])
                tot_a[count] = means_z[k]
                tot_sig[count] = np.sqrt(var_z[k])        
                
                fact1 += tot_w[count]*tot_a[count] * Phi(tot_mu[count], tot_delta[count])
                fact2 += kap*rho*tot_w[count]*tot_sig[count]*Prime(tot_mu[count], tot_delta[count])
                count +=1
        G[ikap] = np.sum(fact1)+np.sum(fact2)

    return(G)        

def self_condir(kappas, m, n, Icue, amp):
    G = np.zeros(len(kappas))
    for ikap, kap in enumerate(kappas):
        G[ikap] = np.mean(n*np.tanh(m*kap+Icue*amp))
    return(G) 

def self_condir2(kappas1, kappas2, m1, n1, m2, n2, Icue, amp):
    G = np.zeros((len(kappas1), len(kappas2),2))
    Gm = np.zeros((len(kappas1), len(kappas2),2))
    
    E = np.zeros((len(kappas1), len(kappas2)))
    
    K1 = np.zeros((len(kappas1), len(kappas2)))
    K2 = np.zeros((len(kappas1), len(kappas2)))
    for ikap1, kap1 in enumerate(kappas1):
        for ikap2, kap2 in enumerate(kappas2):    
            G[ikap1, ikap2,0] = np.mean(n1*np.tanh(m1*kap1+m2*kap2+Icue[:,0]*amp))
            G[ikap1, ikap2,1] = np.mean(n2*np.tanh(m1*kap1+m2*kap2+Icue[:,0]*amp))
            Gm[ikap1, ikap2,0] = G[ikap1, ikap2,0]-kap1
            Gm[ikap1, ikap2,1] = G[ikap1, ikap2,1]-kap2
            
            E[ikap1, ikap2] = (G[ikap1, ikap2,0]-kap1)**2+(G[ikap1, ikap2,1]-kap2)**2
            K1[ikap1, ikap2] = kap1
            K2[ikap1, ikap2] = kap2
            
    return(G, E, Gm, K1, K2)  

def self_condir2s(kappas1, m1, n1, m2, n2, Icue, amp):
    G = np.zeros((len(kappas1)))
    
    for ikap1, kap1 in enumerate(kappas1):
        G[ikap1] = np.mean(n1*np.tanh(m1*kap1+m2*kap1+Icue[:,0]*amp))+np.mean(n2*np.tanh(m1*kap1+m2*kap1+Icue[:,0]*amp))
    return(G)                
    
def angle2(vec1, vec2):    
    norm1 = np.sqrt(np.dot(vec1.T, vec1))
    norm2 = np.sqrt(np.dot(vec2.T, vec2))
    factor= 180/np.pi
    ang = np.arccos(np.dot(vec1.T, vec2)/(norm1*norm2))*factor
    return(ang)    
    
def find_kaps(E,Gm, kappas1, kappas2, tol_around = 0.1, tol = 0.05):
        
    Er = np.ravel(E)
    cand_ix0, cand_ix1 = np.unravel_index(np.argwhere(Er<tol), E.shape)
    
    real0 = []
    real1 = []
    stab = []
    eigs1 = []
    eigs2 = []
    c0s = []
    c1s = []
    dkap = kappas1[1]-kappas1[0]
    
    idcs_ar = (tol_around//dkap)+1
    
    for ic0, c0 in enumerate(cand_ix0):
        c0 = c0[0]
        c1 = cand_ix1[ic0][0]
        
        low1 = int(np.max((0, -idcs_ar+c0)))
        low2 = int(np.max((0, -idcs_ar+c1)))
        high1 = int(np.min((len(kappas1)-1, idcs_ar+c0)))
        high2 = int(np.min((len(kappas1)-1, idcs_ar+c1)))
        if E[c0,c1]==np.min(E[low1:high1, low2:high2]):
            if (Gm[c0+1,c1,0]*Gm[c0-1, c1,0])<0 and (Gm[c0,c1-1,1]*Gm[c0-1, c1+1,1])<0:
                    real0.append(c0)
                    real1.append(c1)
                    M = np.zeros((2,2))
                    M[0,0] = Gm[c0+1,c1,0]-Gm[c0-1,c1,0]
                    M[0,1] = Gm[c0,c1+1,0]-Gm[c0,c1-1,0]
                    M[1,1] = Gm[c0,c1+1,1]-Gm[c0,c1-1,1]
                    M[1,0] = Gm[c0+1,c1,1]-Gm[c0-1,c1,1]
                    largst, minst = np.linalg.eigvals(M/(2*dkap))
                    eigs1.append(largst)
                    eigs2.append(minst)
                    c0s.append(c0)
                    c1s.append(c1)
                    
                    
                    if np.max(np.real(np.linalg.eigvals(M)))<0:
                        stab.append(1)
                    else:
                        stab.append(0)
    kappa_x = np.array((kappas1[real0], kappas2[real1]))
    return(kappa_x, np.array(stab), np.array(eigs1), np.array(eigs2))#, M, c0s, c1s)

def create_inp_out_rsg(trials, Nt, tss, R1_on, SR1_on, just=-1,  perc = 0.1, perc1 = 0.1):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 3))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    
    r_inp = np.ones((trials, Nt))
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)

    for itr in range(trials):      
        redset = tss[ct[itr]]
        if  ct2[itr]:
            maskt[itr,:,0] = time<Nt
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
        if ct3[itr] and ~ct2[itr]:
            maskt[itr,:,0] = time<Nt
        else:
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset)*(time<2*redset+R1_on+1+rnd[itr])
            mask_aft = time>=2*redset+R1_on+1+rnd[itr]
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            s_inp_S[itr, time>R1_on+rnd[itr]+redset] = 10.
            s_inp_S[itr, time>1+R1_on+rnd[itr]+redset] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                


    inputt[:,:,0] = r_inp
    inputt[:,:,1] = s_inp_R
    inputt[:,:,2] = s_inp_S
    
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsgC(trials, Nt, tss, R1_on, SR1_on, just=-1,  perc = 0.1, perc1 = 0.1):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 3))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    
    r_inp = np.ones((trials, Nt))
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)

    for itr in range(trials):      
        redset = tss[ct[itr]]
        if  ct2[itr]:
            maskt[itr,:,0] = time<Nt
            s_inp_R[itr, time>R1_on+rnd[itr]] = 1.
            #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
        if ct3[itr] and ~ct2[itr]:
            maskt[itr,:,0] = time<Nt
        else:
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset)*(time<2*redset+R1_on+1+rnd[itr])
            mask_aft = time>=2*redset+R1_on+1+rnd[itr]
            s_inp_R[itr, time>R1_on+rnd[itr]] = 1.
            #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            s_inp_S[itr, time>R1_on+rnd[itr]+redset] = 10.
            s_inp_S[itr, time>1+R1_on+rnd[itr]+redset] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                


    inputt[:,:,0] = r_inp
    inputt[:,:,1] = s_inp_R
    inputt[:,:,2] = s_inp_S
    
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg1(trials, Nt, tss, R1_on, SR1_on, just=-1,  perc = 0.1, perc1 = 0.1):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 3))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    inc_mask = 20
    
    r_inp = np.ones((trials, Nt))
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)

    for itr in range(trials):      
        redset = tss[ct[itr]]
        if  ct2[itr]:
            maskt[itr,:,0] = time<Nt
            s_inp_R[itr, time>R1_on+rnd[itr]] = 1.
            #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
        if ct3[itr] and ~ct2[itr]:
            maskt[itr,:,0] = time<Nt
        else:
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset)*(time<2*redset+R1_on+1+rnd[itr])
            mask_aft = time>=2*redset+R1_on+1+rnd[itr]
            s_inp_R[itr, time>R1_on+rnd[itr]] = 1.
            s_inp_R[itr, time>R1_on+rnd[itr]+redset] = 0.
            s_inp_S[itr, time>R1_on+rnd[itr]+redset] = 10.
            s_inp_S[itr, time>1+R1_on+rnd[itr]+redset] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset-inc_mask)*(time<2*redset+R1_on+1+rnd[itr]+inc_mask)

    inputt[:,:,0] = r_inp
    inputt[:,:,1] = s_inp_R
    inputt[:,:,2] = s_inp_S
    
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg2(trials, Nt, tss, R1_on, SR1_on, just=-1,  perc = 0.1, perc1 = 0.1):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 3))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    inc_mask = 20
    
    r_inp = np.ones((trials, Nt))
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)

    for itr in range(trials):      
        redset = tss[ct[itr]]
        if  ct2[itr]:
            maskt[itr,:,0] = time<Nt
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            
            #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
        if ct3[itr] and ~ct2[itr]:
            maskt[itr,:,0] = time<Nt
        else:
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset)*(time<2*redset+R1_on+1+rnd[itr])
            mask_aft = time>=2*redset+R1_on+1+rnd[itr]
            s_inp_R[itr, time>R1_on+rnd[itr]] = 1.
            s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
            s_inp_S[itr, time>R1_on+rnd[itr]+redset] = 10.
            s_inp_S[itr, time>1+R1_on+rnd[itr]+redset] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset-inc_mask)*(time<2*redset+R1_on+1+rnd[itr]+inc_mask)

    inputt[:,:,0] = r_inp
    inputt[:,:,1] = s_inp_R
    inputt[:,:,2] = s_inp_S
    
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg3(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1,  perc = 0.1, perc1 = 0.1):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 3))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    inc_mask = 30
    
    r_inp = np.ones((trials, Nt))
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)

    for itr in range(trials):      
        redset = tss[ct[itr]]
        redset_comp = tss_comp[ct[itr]]
        if  ct2[itr]:
            maskt[itr,:,0] = time<Nt
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            
            #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
        if ct3[itr] and ~ct2[itr]:
            maskt[itr,:,0] = time<Nt
        else:
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp)*(time<redset_comp+redset+R1_on+1+rnd[itr])
            mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
            s_inp_S[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
            s_inp_S[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask)

    inputt[:,:,0] = r_inp
    inputt[:,:,1] = s_inp_R
    inputt[:,:,2] = s_inp_S
    
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg_2out(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1,  
                            perc = 0.1, perc1 = 0.1, noset=False, noready=False, align_set = False, align_go = False, fixGo = True):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 1))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    inc_mask = 30
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+interval, time<R1_on+rnd[itr]+2*interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp)*(time<redset_comp+redset+R1_on+1+rnd[itr])
                mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
                s_inp_S[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
                s_inp_S[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_comp-interval, time<R1_on+rnd[itr]+redset_comp)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
                maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask)
    
    elif align_set:
        
        if fixGo==True:
            fixT = R1_on + np.max(tss)
        else:
            fixT  = fixGo 
        
            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT] = 10.
                s_inp_S[itr, time>1+fixT] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT-interval, time<fixT)
                mask22 = np.logical_and(time>fixT+redset, time<fixT+redset_comp+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
    else:
        if fixGo:
            fixT = R1_on + np.max(tss)
            fixT2 = R1_on + np.max(tss)*2
        else:
            fixT  = fixGo - np.max(tss)
            fixT2 = fixGo
        
            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>fixT2-redset)*(time<fixT2)
                mask_aft = time>=fixT2
                s_inp_R[itr, time>fixT2-2*redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT2-2*redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT2-redset] = 10.
                s_inp_S[itr, time>1+fixT2-redset] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT2-redset-interval, time<fixT2-redset)
                mask22 = np.logical_and(time>fixT2, time<fixT2+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            
                    
                #maskt[itr,:,0] = (time>set_time-inc_mask)*(time<1+set_time+redset_comp+inc_mask)
    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,0] +=   s_inp_S
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg_2out2inp(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1,  
                            perc = 0.1, perc1 = 0.1, noset=False, noready=False, align_set = False, align_go = False, fixGo = True):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 2))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    inc_mask = 30
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+interval, time<R1_on+rnd[itr]+2*interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp)*(time<redset_comp+redset+R1_on+1+rnd[itr])
                mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
                s_inp_S[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
                s_inp_S[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_comp-interval, time<R1_on+rnd[itr]+redset_comp)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
                maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask)
    
    elif align_set:
        
        if fixGo==True:
            fixT = R1_on + np.max(tss)
        else:
            fixT  = fixGo 
        
            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT] = 10.
                s_inp_S[itr, time>1+fixT] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT-interval, time<fixT)
                mask22 = np.logical_and(time>fixT+redset, time<fixT+redset_comp+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
    else:
        if fixGo:
            fixT = R1_on + np.max(tss)
            fixT2 = R1_on + np.max(tss)*2
        else:
            fixT  = fixGo - np.max(tss)
            fixT2 = fixGo
        
            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>fixT2-redset)*(time<fixT2)
                mask_aft = time>=fixT2
                s_inp_R[itr, time>fixT2-2*redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT2-2*redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT2-redset] = 10.
                s_inp_S[itr, time>1+fixT2-redset] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT2-redset-interval, time<fixT2-redset)
                mask22 = np.logical_and(time>fixT2, time<fixT2+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            
                    
                #maskt[itr,:,0] = (time>set_time-inc_mask)*(time<1+set_time+redset_comp+inc_mask)
    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,1] +=   s_inp_S
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg_cont(trials, Nt, tss, R1_on, SR1_on, fact = 1, just=-1,  
                            perc = 0.1, perc1 = 0.1, noset=False, noready=False, align_set = False, align_go = False, fixGo = True, bayes=False):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    if len(tss_comp)>2:
        tss_comp = np.array((tss_comp[0], tss_comp[1]))
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 1))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    inc_mask = 30
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    time_shown = np.zeros(trials)
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = np.random.randint(tss_comp[0], tss_comp[1])#tss[ct[itr]]
            if bayes==False:
                redset_real = redset
            else:
                redset_real = redset + np.round(bayes*redset*np.random.randn())
            ct[itr] = redset
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+interval, time<R1_on+rnd[itr]+2*interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt                
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_real)*(time<redset_real+redset+R1_on+1+rnd[itr])
                mask_aft = time>=redset_real+redset+R1_on+1+rnd[itr]
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
                s_inp_S[itr, time>R1_on+rnd[itr]+redset_real] = 10.
                s_inp_S[itr, time>1+R1_on+rnd[itr]+redset_real] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_real-interval, time<R1_on+rnd[itr]+redset_real)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
                maskt[itr,:,0] = (time>redset_real+R1_on+1+rnd[itr]-inc_mask)*(time<redset+redset+R1_on+1+rnd[itr]+inc_mask)
            time_shown[itr] = redset_real
            
    elif align_set:
        
        if fixGo==True:
            fixT = R1_on + np.max(tss)
        else:
            fixT  = fixGo 
        
            
        for itr in range(trials):      
            redset = np.random.randint(tss_comp[0], tss_comp[1])#tss[ct[itr]]
            if bayes==False:
                redset_real = redset
            else:
                redset_real = redset + np.round(bayes*redset*np.random.randn())
            ct[itr] = redset
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-redset_real-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT-redset_real+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT] = 10.
                s_inp_S[itr, time>1+fixT] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT-interval, time<fixT)
                mask22 = np.logical_and(time>fixT+redset, time<fixT+redset+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            time_shown[itr] = redset_real
            
    else:
        if fixGo:
            fixT = R1_on + np.max(tss)
            fixT2 = R1_on + np.max(tss)*2
        else:
            fixT  = fixGo - np.max(tss)
            fixT2 = fixGo
        
            
        for itr in range(trials):      
            redset = np.random.randint(tss_comp[0], tss_comp[1])#tss[ct[itr]]
            if bayes==False:
                redset_real = redset
            else:
                redset_real = redset + np.round(bayes*redset*np.random.randn())
            ct[itr] = redset
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>fixT2-redset)*(time<fixT2)
                mask_aft = time>=fixT2
                s_inp_R[itr, time>fixT2-redset_real-redset-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT2-redset_real-redset+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT2-redset] = 10.
                s_inp_S[itr, time>1+fixT2-redset] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT2-redset-interval, time<fixT2-redset)
                mask22 = np.logical_and(time>fixT2, time<fixT2+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            time_shown[itr] = redset_real
            
            
    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,0] +=   s_inp_S
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    return(inputt, outputt, maskt, ct, ct2, ct3, time_shown)

def create_inp_out_rSSg(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1,  perc = 0.1, perc1 = 0.1, delayF = 0, delay_min = 20, delay_max = 250, noset=False, noready=False, align_set = False, align_go = False):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 1))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    inc_mask = 30
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S1 =  np.zeros((trials, Nt))
    s_inp_S2 =  np.zeros((trials, Nt))
    
    
    if delayF==0:
        delayF = np.round(np.mean((delay_min, delay_max)))
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            delay = np.random.randint(delay_min, delay_max)
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]+delay-interval, time<R1_on+rnd[itr]+delay)
                mask22 = np.logical_and(time>R1_on+rnd[itr]+delay+interval, time<R1_on+rnd[itr]+2*interval+delay)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+delay)
                mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]+delay
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
                s_inp_S1[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
                s_inp_S1[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
                
                s_inp_S2[itr, time>R1_on+rnd[itr]+redset_comp+delay] = 10.
                s_inp_S2[itr, time>1+R1_on+rnd[itr]+redset_comp+delay] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval+delay, time<R1_on+rnd[itr]+delay)
                mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_comp-interval+delay, time<R1_on+rnd[itr]+redset_comp+delay)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
                maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask+delay)
    
    elif align_set:
        fixT = R1_on + np.max(tss)


        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-redset_comp-rnd[itr]-delayF] = 10.
                s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]-delayF] = 0.
                
                s_inp_S1[itr, time>fixT-delayF] = 10.
                s_inp_S1[itr, time>1+fixT-delayF] = 0.
                
                
                s_inp_S2[itr, time>fixT] = 10.
                s_inp_S2[itr, time>1+fixT] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT-interval, time<fixT)
                mask22 = np.logical_and(time>fixT+redset, time<fixT+redset_comp+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
    else:
        fixT = R1_on + np.max(tss)
        fixT2 = R1_on + np.max(tss)*2
        
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>fixT2-redset)*(time<fixT2)
                mask_aft = time>=fixT2
                s_inp_R[itr, time>fixT2-2*redset_comp-rnd[itr]-delayF] = 10.
                s_inp_R[itr, time>fixT2-2*redset_comp+1-rnd[itr]-delayF] = 0.
                
                s_inp_S1[itr, time>fixT2-redset-delayF] = 10.
                s_inp_S1[itr, time>1+fixT2-redset-delayF] = 0.
                
                s_inp_S2[itr, time>fixT2-redset] = 10.
                s_inp_S2[itr, time>1+fixT2-redset] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT2-redset-interval, time<fixT2-redset)
                mask22 = np.logical_and(time>fixT2, time<fixT2+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            
                    
                #maskt[itr,:,0] = (time>set_time-inc_mask)*(time<1+set_time+redset_comp+inc_mask)
    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,0] +=   s_inp_S1
        inputt[:,:,0] +=   s_inp_S2
        
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)
    
def create_inp_out_rSSg_2in(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1,  perc = 0.1, perc1 = 0.1, delayF = 0, 
                            delay_min = 20, delay_max = 250, noset=False, noready=False, align_set = False, align_go = False):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 2))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    inc_mask = 30
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S1 =  np.zeros((trials, Nt))
    s_inp_S2 =  np.zeros((trials, Nt))
    
    
    if delayF==0:
        delayF = np.round(np.mean((delay_min, delay_max)))
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            delay = np.random.randint(delay_min, delay_max)
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]+delay-interval, time<R1_on+rnd[itr]+delay)
                mask22 = np.logical_and(time>R1_on+rnd[itr]+delay+interval, time<R1_on+rnd[itr]+2*interval+delay)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+delay)
                mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]+delay
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
                s_inp_S1[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
                s_inp_S1[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
                
                s_inp_S2[itr, time>R1_on+rnd[itr]+redset_comp+delay] = 10.
                s_inp_S2[itr, time>1+R1_on+rnd[itr]+redset_comp+delay] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval+delay, time<R1_on+rnd[itr]+delay)
                mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_comp-interval+delay, time<R1_on+rnd[itr]+redset_comp+delay)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
                maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask+delay)
    
    elif align_set:
        fixT = R1_on + np.max(tss)


        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-redset_comp-rnd[itr]-delayF] = 10.
                s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]-delayF] = 0.
                
                s_inp_S1[itr, time>fixT-delayF] = 10.
                s_inp_S1[itr, time>1+fixT-delayF] = 0.
                
                
                s_inp_S2[itr, time>fixT] = 10.
                s_inp_S2[itr, time>1+fixT] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT-interval, time<fixT)
                mask22 = np.logical_and(time>fixT+redset, time<fixT+redset_comp+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
    else:
        fixT = R1_on + np.max(tss)
        fixT2 = R1_on + np.max(tss)*2
        
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>fixT2-redset)*(time<fixT2)
                mask_aft = time>=fixT2
                s_inp_R[itr, time>fixT2-2*redset_comp-rnd[itr]-delayF] = 10.
                s_inp_R[itr, time>fixT2-2*redset_comp+1-rnd[itr]-delayF] = 0.
                
                s_inp_S1[itr, time>fixT2-redset-delayF] = 10.
                s_inp_S1[itr, time>1+fixT2-redset-delayF] = 0.
                
                s_inp_S2[itr, time>fixT2-redset] = 10.
                s_inp_S2[itr, time>1+fixT2-redset] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT2-redset-interval, time<fixT2-redset)
                mask22 = np.logical_and(time>fixT2, time<fixT2+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            
    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,0] +=   s_inp_S1
        inputt[:,:,1] +=   s_inp_S2
        
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg_justramp(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1, 
                                perc = 0.1, perc1 = 0.1, inc_mask = 30, 
                                noset=False, noready=False, align_set = False, align_go = False):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 1))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+interval, time<R1_on+rnd[itr]+2*interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp)*(time<redset_comp+2*redset+R1_on+1+rnd[itr])
                mask_aft = time>=redset_comp+2*redset+R1_on+1+rnd[itr]
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
                s_inp_S[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
                s_inp_S[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_comp-interval, time<R1_on+rnd[itr]+redset_comp)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
                maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask)*(time<redset_comp+redset+R1_on+2+rnd[itr]+inc_mask)
    
    elif align_set:
        fixT = R1_on + np.max(tss)

            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT] = 10.
                s_inp_S[itr, time>1+fixT] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT-interval, time<fixT)
                mask22 = np.logical_and(time>fixT+redset, time<fixT+redset_comp+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
    else:
        fixT = R1_on + np.max(tss)
        fixT2 = R1_on + np.max(tss)*2
            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>fixT2-redset)*(time<fixT2)
                mask_aft = time>=fixT2
                s_inp_R[itr, time>fixT2-2*redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT2-2*redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT2-redset] = 10.
                s_inp_S[itr, time>1+fixT2-redset] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT2-redset-interval, time<fixT2-redset)
                mask22 = np.logical_and(time>fixT2, time<fixT2+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, 0, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            
                    
                #maskt[itr,:,0] = (time>set_time-inc_mask)*(time<1+set_time+redset_comp+inc_mask)
    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,0] +=   s_inp_S
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    return(inputt, outputt, maskt, ct, ct2, ct3)
    
def create_inp_out_rsg_justramp2(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1, 
                                perc = 0.1, perc1 = 0.1, inc_mask = 30, FF = 1.25,
                                noset=False, noready=False, align_set = False, align_go = False,conti = True):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 1))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+interval, time<R1_on+rnd[itr]+2*interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp)*(time<redset_comp+FF*redset+R1_on+1+rnd[itr])
                mask_aft = time>=redset_comp+FF*redset+R1_on+1+rnd[itr]
                s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
                s_inp_S[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
                s_inp_S[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
                mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_comp-interval, time<R1_on+rnd[itr]+redset_comp)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                if conti:
                    maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask)*(time<redset_comp+FF*redset+R1_on+1+rnd[itr])
                else:
                    maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask)*(time<redset_comp+redset+R1_on+1+rnd[itr])
    
    elif align_set:
        fixT = R1_on + np.max(tss)

            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT] = 10.
                s_inp_S[itr, time>1+fixT] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT-interval, time<fixT)
                mask22 = np.logical_and(time>fixT+redset, time<fixT+redset_comp+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
    else:
        fixT = R1_on + np.max(tss)
        fixT2 = R1_on + np.max(tss)*2
            
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
                s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
                s_inp_R[itr, time>1+fixT-R1_on+rnd[itr]] = 0.

            if ct3[itr] and ~ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>fixT2-redset)*(time<fixT2)
                mask_aft = time>=fixT2
                s_inp_R[itr, time>fixT2-FF*redset_comp-rnd[itr]] = 10.
                s_inp_R[itr, time>fixT2-FF*redset_comp+1-rnd[itr]] = 0.
                
                s_inp_S[itr, time>fixT2-redset] = 10.
                s_inp_S[itr, time>1+fixT2-redset] = 0.
                
                
                outputt[itr,:,1] = 0.
                mask11 = np.logical_and(time>fixT2-redset-interval, time<fixT2-redset)
                mask22 = np.logical_and(time>fixT2, time<fixT2+interval)
                outputt[itr,mask11,1] = strt
                outputt[itr,mask22,1] = -strt
                maskt[itr,:,1] =np.logical_or(mask11, mask22)
                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, 0, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
            
                    
                #maskt[itr,:,0] = (time>set_time-inc_mask)*(time<1+set_time+redset_comp+inc_mask)
    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,0] +=   s_inp_S
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg4(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1,  perc = 0.1, perc1 = 0.1):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 3))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    inc_mask = 30
    
    r_inp = np.ones((trials, Nt))
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    fixT = R1_on + np.max(tss)
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)

    for itr in range(trials):      
        redset = tss[ct[itr]]
        redset_comp = tss_comp[ct[itr]]
        if  ct2[itr]:
            maskt[itr,:,0] = time<Nt
            s_inp_R[itr, time>fixT-R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
            
            #s_inp_R[itr, time>1+R1_on+rnd[itr]] = 0.
        if ct3[itr] and ~ct2[itr]:
            maskt[itr,:,0] = time<Nt
        else:
            maskt[itr,:,0] = (time>1+fixT)*(time<redset+1+fixT)
            mask_aft = time>=redset+1+fixT
            s_inp_R[itr, time>fixT-redset_comp-rnd[itr]] = 10.
            s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]] = 0.
            s_inp_S[itr, time>fixT] = 10. #R1_on+rnd[itr]+redset_comp
            s_inp_S[itr, time>1+fixT] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>fixT-inc_mask)*(time<redset+1+fixT+inc_mask)

    inputt[:,:,0] = r_inp
    inputt[:,:,1] = s_inp_R
    inputt[:,:,2] = s_inp_S
    
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_rsg_step(trials, Nt, tss, R1_on, SR1_on, fact = 5, just=-1,  perc = 0.1, perc1 = 0.0, noready=False):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 1))
    outputt = strt*np.ones((trials, Nt, 2))
    maskt   = np.zeros((trials, Nt, 2))
    interval = np.min(tss_comp)//2
    inc_mask = 30
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S =  np.zeros((trials, Nt))
    
    
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)

    for itr in range(trials):      
        redset = tss[ct[itr]]
        redset_comp = tss_comp[ct[itr]]

        if ct3[itr]:
            maskt[itr,:,0] = time<Nt
        else:
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp)*(time<redset_comp+redset+R1_on+1+rnd[itr])
            mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]
            s_inp_R[itr, time>R1_on+rnd[itr]] = 1.
            #s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
            #s_inp_S[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
            s_inp_R[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
            
            
            outputt[itr,:,1] = 0.
            mask11 = np.logical_and(time>R1_on+rnd[itr]-interval, time<R1_on+rnd[itr])
            mask22 = np.logical_and(time>R1_on+rnd[itr]+redset_comp-interval, time<R1_on+rnd[itr]+redset_comp)
            outputt[itr,mask11,1] = strt
            outputt[itr,mask22,1] = -strt
            maskt[itr,:,1] =np.logical_or(mask11, mask22)
            
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask)

    if noready ==False:
        inputt[:,:,0] += s_inp_R
    
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_csgdelay_2in(trials, Nt, tss, amps, R1_on, SR1_on, fact = 5, just=-1,  perc = 0.1, 
                                delayF = 0, delay_min = 20, delay_max = 250, noset=False, 
                                noready=False, align_set = False, align_go = False):
    '''
    trials: Number of trials
    Nt :    Number of time points
    tss :   Intervals between set and go
    R1_on:  Time of ready
    SR1_on: Standard deviation of the temporal onset of "Ready".
    perc:   Percentage of trials in which no transient inputs appear
    perc1:  Percentage of trials in which only the ready cue appears
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)
    
    strt = -0.5
    inputt  = np.zeros(( trials, Nt, 2))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    inc_mask = 30
    
    s_inp_R =  np.zeros((trials, Nt))
    s_inp_S1 =  np.zeros((trials, Nt))
    s_inp_S2 =  np.zeros((trials, Nt))
    
    length = 2*np.mean(tss) # I changed this in July20
    
    delays = np.zeros(trials)
    if delayF==0:
        delayF = np.round(np.mean((delay_min, delay_max)))
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
       
    else:
        ct = just*np.ones(trials, dtype = np.int8)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)
    if not align_set and not align_go:
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            amp_c = amps[ct[itr]]
            delay = np.random.randint(delay_min, delay_max)
            delays[itr] = delay
            if  ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+length+delay)*(time<redset_comp+length+R1_on+1+rnd[itr]+delay)
                mask_aft = time>=redset_comp+length+R1_on+1+rnd[itr]+delay
                s_inp_R[itr, time>R1_on+rnd[itr]] = amp_c
                s_inp_R[itr, time>1+R1_on+rnd[itr]+length] = 0.
                
                s_inp_S2[itr, time>R1_on+rnd[itr]+length+delay] = 10.
                s_inp_S2[itr, time>1+R1_on+rnd[itr]+length+delay] = 0.
                                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    
                maskt[itr,:,0] = (time>length+R1_on+1+rnd[itr]-inc_mask+delay)*(time<length+redset+R1_on+1+rnd[itr]+inc_mask+delay)
    elif align_set:
        fixT = R1_on + np.max(tss)
        for itr in range(trials):      
            redset = tss[ct[itr]]
            redset_comp = tss_comp[ct[itr]]
            amp_c = amps[ct[itr]]

            if ct2[itr]:
                maskt[itr,:,0] = time<Nt
            else:
                maskt[itr,:,0] = (time>1+fixT)*(time<redset+fixT+1)
                mask_aft = time>=redset+1+fixT
                s_inp_R[itr, time>fixT-length-delayF] = amp_c
                s_inp_R[itr, time>1+fixT-delayF] = 0.
                
                
                s_inp_S2[itr, time>fixT] = 10.
                s_inp_S2[itr, time>1+fixT] = 0.

                
                if sum(maskt[itr,:,0]):
                    outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                    outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                    

    
    if noready ==False:
        inputt[:,:,0] += s_inp_R
    if noset==False:
        inputt[:,:,0] +=   s_inp_S1
        inputt[:,:,1] +=   s_inp_S2
        
        
    
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, delays)