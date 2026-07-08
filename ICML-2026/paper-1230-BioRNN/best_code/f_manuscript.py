import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import funcs_Sphere as fs
from matplotlib import cm
        # hellooo

rank = 3

dt = 10#ms
tau = 100#ms

alpha = dt/tau
std_noise_rec = np.sqrt(2*alpha)*0.1

input_size = 4
hidden_size = 1000
output_size = 1
size_f = np.sqrt(10)

#initial connectivity
sigma_mn = 0.85

trials_train = 500
trials_test = 100

R_on  = 1000//dt#500//dt


#%%
tss = np.array((800, 1550))
tss2 = np.array(( 800,  1050, 1300, 1550))
tss22 = 2*np.array(( 800,  1050, 1300, 1550))

tss3 = np.linspace(800, 3000, 20)
tss32 = 1.5*np.linspace(800, 3000, 20)

N_steps = 5


Tss  = fs.gen_intervals(tss,N_steps)
Tss2 = fs.gen_intervals(tss2,N_steps)


#%%
Nt = 1100 
time = np.arange(Nt)

Nt2 = 1300
time2 = np.arange(Nt2)

#%%
# Parameters of task
SR_on = 60
factor = 1
dela = 150
repeat = 1#7 #20 number of examples

# Colors
cls2 = fs.set_plot()

cls2[1,:] = cls2[2,:]
cls2[2,:] = cls2[4,:]
cls2[3,:] = cls2[5,:]

initial_h0 = False


cls = np.zeros((7,3))
cl11 = np.array((102, 153, 255))/255.
cl12 = np.array((53, 153, 53))/255.

cl21 = np.array((255, 204, 51))/255.
cl22 = np.array((204, 0, 0))/255.

cls[0,:] = 0.4*np.ones((3,))

cls[1,:] = cl11
cls[2,:] = 0.5*cl11+0.5*cl12
cls[3,:] = cl12

cls[4,:] = cl21
cls[5,:] = 0.5*cl21+0.5*cl22
cls[6,:] = cl22

CLL = np.zeros((8,3))
CLL[0,:] = np.array((150, 53, 47))/256.
CLL[1,:] = np.array((207, 84, 57))/256.
CLL[2,:] = np.array((212, 104, 81))/256.
CLL[3,:] = np.array((220, 135, 64))/256.

CLL[4,:] = np.array((78, 104, 172))/256.
CLL[5,:] = np.array((85, 131, 198))/256.
CLL[6,:] = np.array((102, 185, 224))/256.
CLL[7,:] = np.array((150, 198, 219))/256.
CLL2 = CLL

CLL[0,:] = np.array((203, 80, 71))/256.

CLL[3,:] = np.array((53, 153, 53))/256.
CLL[1,:] = 0.75*CLL[0,:] + 0.25*CLL[3,:]
CLL[2,:] = 0.25*CLL[0,:] + 0.75*CLL[3,:]

CLL[4,:] = np.array((255, 204, 51))/256.
CLL[7,:] = np.array((71, 89, 156))/256.
CLL[5,:] = 0.75*CLL[4,:] + 0.25*CLL[7,:]
CLL[6,:] = 0.1*CLL[4,:] + 0.9*CLL[7,:]

def plot_output_MWG(net_low_all, tss2, tss22, dt, time, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, tss_ref2=0, dela = 150, give_inps=False,
                    plot_sev = np.ones(2), s1 = 0., s2 = 0.1, rem_prod=False, rem_meas = False, non_avg=False, R_on= False):
    if fr==False:
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
    if plot:
        fig_width = 1.5*2.2 # width in inches
        fig_height = 0.8*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        
    CLL = CLL2
    if gener==True:
        evenly_spaced_interval = np.linspace(0, 1, len(tss2)+10)
        cls2 = [cm.viridis(x) for x in evenly_spaced_interval]
        CLL = cls2
    trials = 10
    Trajs = np.zeros((len(time), rank, len(tss2)))
    if non_avg == False:
        Inps  = np.zeros((len(time), net_low_all.hidden_size, len(tss2)))
    else:
        Inps  = np.zeros((trials, len(time), net_low_all.hidden_size, len(tss2)))
    
    if sum(plot_sev)>1:
        Trajs2 = np.zeros((len(time), rank, len(tss2)))
        if non_avg == False:
            Inps2  = np.zeros((len(time), net_low_all.hidden_size, len(tss2)))
        else:
            Inps2  = np.zeros((trials, len(time), net_low_all.hidden_size, len(tss2)))
        
    T0s_lr = []
    T0s_lr2 = []
    
    if R_on==False:
        R_on = 1000//dt
    if len(tss2)>1:
        dtss = tss2[1]-tss2[0]
    Nt = len(time)
    T0 = 6000
    if plot_sev[0] == 1.: 
        for xx in range(len(tss2)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True )
            if rem_prod:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True, rem_prod=True)                
            if rem_meas:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True, rem_meas=True )                               
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()
            traj = traj.detach().numpy()
            #if non_avg==False:
            mtraj  = np.mean(traj,0)
            # else:
            #     mtraj = traj
            if np.shape(mtraj)[0]>Nt:
                mtraj2 = mtraj[:-1,:]
                traj = traj[:,:-1,:]
            else:
                mtraj2 = mtraj
            if fr==False:

                k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
                k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
                k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
                Trajs[:,0,xx] = k1_traj[:-1]
                Trajs[:,1,xx] = k2_traj[:-1]
                Trajs[:,2,xx] = k3_traj[:-1]
            if give_inps==True:
                if non_avg==False:
                    Inps[:,:,xx] = mtraj2
                else:
                    Inps[:,:,:,xx] = traj
                
            outp2 = np.copy(outp)
            
            outp3 = np.copy(outp2[:,1:,0])
            outp3[:,np.diff(low_pass(np.mean(outp2[:,:,0],0)))<0]=5.
            
            outp3[:, time[1:]*dt<4000] = 5. 
            outp2 = outp3
    
            tt0s = time[np.argmin(np.abs(outp2-0.35),1)]*dt-R_on*dt-np.max((np.max(tss2),np.max(tss22)))-dela*dt
            T0s_lr.append(tt0s)
            avg_outp0 = np.mean(outp[:,:,0],0)#np.mean(outp3[:,:,0],0)
            if plot:
                        
                if gener==False:
                    ax.plot(time*dt-T0, avg_outp0, color=CLL[xx,:], lw=2)    
                    ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=CLL[xx,:], alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=CLL[xx,:], alpha=0.5)
                else:
                    if np.min(np.abs(tss2[xx]-tss_ref))<0.5*dtss:
                        ax.plot(time*dt-T0, avg_outp0, color='k', lw=2)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color='k')
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color='k')
                    
                    else:
                        ax.plot(time*dt-T0, avg_outp0, color=CLL[xx], lw=1.5)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=CLL[xx], alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.5)
                    
    # Doing long stimuli
    if plot_sev[1] == 1.:
        if len(tss22)>1:
            dtss = tss22[1]-tss22[0]
        for xx in range(len(tss22)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, perc = 0.,
                                                                                 perc1 = 0., just2=1, align_set=True,  s1 = s1, s2 = s2)
            if rem_prod:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=1, s1 = s1, s2 = s2, align_set=True, rem_prod=True)                
            if rem_meas:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=1, s1 = s1, s2 = s2, align_set=True, rem_meas=True )  
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()
            
            traj = traj.detach().numpy()
            traj2 = traj
            
            mtraj  = np.mean(traj,0)
            if np.shape(mtraj)[0]>Nt:
                mtraj2 = mtraj[:-1,:]
                traj2 = traj2[:,:-1,:]
            else:
                mtraj2 = mtraj
            if fr==False:
                k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
                k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
                k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
                if sum(plot_sev)==1:
                    Trajs[:,0,xx] = k1_traj[:-1]
                    Trajs[:,1,xx] = k2_traj[:-1]
                    Trajs[:,2,xx] = k3_traj[:-1]
                else:
                    Trajs2[:,0,xx] = k1_traj[:-1]
                    Trajs2[:,1,xx] = k2_traj[:-1]
                    Trajs2[:,2,xx] = k3_traj[:-1]

            
            if give_inps==True:
                if sum(plot_sev)==1:
                    if non_avg==False:
                        Inps[:,:,xx] = mtraj2
                    else:
                        Inps[:,:,:,xx] = traj2
                else:
                    if non_avg ==False:
                        Inps2[:,:,xx] = mtraj2
                    else:
                        Inps2[:,:,:,xx] = traj2
            outp2 = np.copy(outp)
            
            outp3 = np.copy(outp2[:,1:,0])
            outp3[:,np.diff(low_pass(np.mean(outp2[:,:,0],0), ints=10))<0]=5.
            
            outp3[:, time[1:]*dt<R_on*dt+np.max((np.max(tss2),np.max(tss22)))+dela*dt] = 5. 
            outp3[:, time[1:]*dt<R_on*dt+np.max((np.max(tss2),np.max(tss22)))+dela*dt] = 5. 
            
            outp2 = outp3
    
            tt0s = time[np.argmin(np.abs(outp2-0.35),1)]*dt-R_on*dt-np.max((np.max(tss2),np.max(tss22)))-dela*dt
            T0s_lr2.append(tt0s)
    
            avg_outp0 = np.mean(outp[:,:,0],0) #np.mean(outp3[:,:,0],0)
            if plot:
                
                if gener==False:
                    
                    nC = CLL[xx+4,:]#lighten_color(CLL[xx,:], amount=0.5)
                    ax.plot(time*dt-T0, avg_outp0, color=nC, lw=2)    
                    ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=nC, alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=nC, alpha=0.5)
                else:
                    nC = CLL[xx]#lighten_color(CLL[xx], amount=0.5)
                    if np.min(np.abs(tss22[xx]-tss_ref2))<0.5*dtss:
                        ax.plot(time*dt-T0, avg_outp0, color='k', lw=2)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k',alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color='k')
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color='k')
                    else:
                        ax.plot(time*dt-T0, avg_outp0, color=nC, lw=1.5)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k',alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=nC, alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=nC, alpha=0.5)
                
    if plot:                    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time (ms)')
        plt.ylabel('read out')
        plt.ylim([-0.6, 0.6])
        plt.yticks([-0.5, 0, 0.5])
        plt.xlim([-800, 3500])
    if give_trajs==True:
        return(Trajs)
    
    if give_inps==True and fr == True and sum(plot_sev)==1.:
        return(Inps)
    elif give_inps==True and fr == False and sum(plot_sev)==1.:
        return(Inps, Trajs)
    elif give_inps==True and fr == False and sum(plot_sev)==2.:
        return(Inps, Trajs, Inps2, Trajs2)
    elif give_inps==True and fr == True and sum(plot_sev)==2.:
        return(Inps, Inps2)
    elif t0s==True:
        return(fig, ax, T0s_lr, T0s_lr2)
    else:
        return(fig, ax)
    
    
def plot_output_MWG_Inp(net_low_all, tss2, tss22, dt, time, rank=3, give_trajs=False, plot=True,
                    t0s=False, gener=False, tss_ref= 0, tss_ref2=0, dela = 150,
                    plot_sev = np.ones(2), s1 = 0., s2 = 0.1):
    M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
    Ib = np.copy(I_pre[3,:])

    for r in range(rank):
        P = np.dot(M_pre[:,r,np.newaxis], M_pre[:,r,np.newaxis].T)/np.dot(M_pre[:,r].T, M_pre[:,r])
        Ib = Ib - P.dot(Ib)
    CLL = cls

    trials = 10
    Trajs = np.zeros((len(time), rank+1, len(tss2)))
    Trajs2 = np.zeros((len(time), rank+1, len(tss2)))
    
    T0s_lr = []
    T0s_lr2 = []
    
    R_on = 1000//dt
    dtss = tss2[1]-tss2[0]
    Nt = len(time)
    if plot_sev[0] == 1.: 
        for xx in range(len(tss2)):
            #input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss2//dt, 
            #                                                        R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela, inp_size=3)
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delay_min = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True )
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()

            traj = traj.detach().numpy()
            
            mtraj  = np.mean(traj,0)
            k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
            k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
            k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
            kI_traj = Ib.dot(mtraj.T)/np.sum(Ib**2)
            
            Trajs[:,0,xx] = k1_traj[:-1]
            Trajs[:,1,xx] = k2_traj[:-1]
            Trajs[:,2,xx] = k3_traj[:-1]
            Trajs[:,3,xx] = kI_traj[:-1]
                
                    
    # Doing long stimuli
    if plot_sev[1] == 1.:
        dtss = tss22[1]-tss22[0]
        for xx in range(len(tss22)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delay_min = dela, inp_size=input_size, perc = 0.,
                                                                                 perc1 = 0., just2=1, align_set=True,  s1 = s1, s2 = s2)
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()

            traj = traj.detach().numpy()
            
            mtraj  = np.mean(traj,0)
            k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
            k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
            k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
            kI_traj = Ib.dot(mtraj.T)/np.sum(Ib**2)
            Trajs2[:,0,xx] = k1_traj[:-1]
            Trajs2[:,1,xx] = k2_traj[:-1]
            Trajs2[:,2,xx] = k3_traj[:-1]
            Trajs2[:,3,xx] = kI_traj[:-1]

            
    return(Trajs, Trajs2)

def plot_inputs_MWG(net_low_all, tss2, tss22,  dt, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, tss_ref2 = 0, dela = 150, 
                    only_perf=False, dist1=True, dist2 =True, gener2 =False):

    if plot:
        fig_width = 1.5*2.2 # width in inches
        fig_height = 1.2*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        
    CLL = CLL2
    if gener==True or gener2==True:
        evenly_spaced_interval = np.linspace(0, 1, len(tss2)+10)
        cls2 = [cm.viridis(x) for x in evenly_spaced_interval]
        CLL = cls2
    trials = 10

    

    dtss = tss2[1]-tss2[0]
    if dist1:
        for xx in range(len(tss2)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                    R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela, inp_size=3)
    
            T0=2500
            if plot:
    
                if gener==False and gener2==False:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx,:], alpha=1, lw=2)
                elif gener2==False:
                    if np.min(np.abs(tss2[xx]-tss_ref))<0.5*dtss:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color='k', lw=2)
                    else:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.8, lw=1)
                else:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.8, lw=1)
        if gener2==False:
            ax3.plot(time*dt-T0, 0*np.ones_like(time), lw=4, color='C0', alpha=0.7)
        
    if dist2:
        for xx in range(len(tss22)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss22//dt, 
                                                                    R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela, inp_size=3)
    
            T0=2500
            if plot:
                if gener==False and gener2==False:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx+4,:], alpha=1, lw=2)
                elif gener2==False:
                    if np.min(np.abs(tss2[xx]-tss_ref2))<0.5*dtss:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color='k', lw=2)
                    else:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx+4], alpha=0.8, lw=1)
        if gener2==False:          
            ax3.plot(time*dt-T0, 0.1*np.ones_like(time), lw=4, color='k', alpha=0.3)
    if gener2:
        ax3.plot(time*dt-T0, 0.1*np.ones_like(time), lw=3, color='k')
        ax3.plot(time*dt-T0, 0.*np.ones_like(time), lw=3, color='C0')
        ax3.plot(time*dt-T0, 0.05*np.ones_like(time), lw=2, color='C1', alpha=0.6)
        ax3.plot(time*dt-T0, 0.15*np.ones_like(time), lw=2, color='C3', alpha=0.6)
        ax3.plot(time*dt-T0, 0.2*np.ones_like(time), lw=2, color='C4', alpha=0.6)
        
    if plot:                    
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')
        ax3.set_xlabel('time (ms)')
        ax1.set_ylabel('input 1')
        ax2.set_ylabel('input 2')
        ax3.set_ylabel('input ctx')
        
        ax1.set_ylim([-0.05, 1])
        ax2.set_ylim([-0.05, 1])
        if gener2==False:
            ax3.set_ylim([-0.02, 0.12])
        else:
            ax3.set_ylim([-0.05, 0.25])
        
        if gener2==False:
            if gener==False:
                ax2.set_xlim([-100, 3500])
                ax1.set_xlim([-100, 3500])
                ax3.set_xlim([-100, 3500])
                
            else:
                ax2.set_xlim([-100, 5100])
                ax1.set_xlim([-100, 5100])
                ax3.set_xlim([-100, 5100])
        else:
            
            ax2.set_xlim([-400, 6400])
            ax1.set_xlim([-400, 6400])
            ax3.set_xlim([-400, 6400])
        if gener2==False:
            if gener==False:
                
                ax1.set_xticks([0, 1000, 2000,3000])
                ax2.set_xticks([0,  1000, 2000, 3000])
                ax3.set_xticks([0,  1000, 2000, 3000])
                
                ax1.set_xticklabels(['','','',''])
                ax2.set_xticklabels(['','','',''])
            else:
                ax1.set_xticks([0, 2000, 4000])
                ax2.set_xticks([0, 2000, 4000])
                ax3.set_xticks([0, 2000, 4000])
                
                ax1.set_xticklabels(['','',''])
                ax2.set_xticklabels(['','',''])
        else:
            ax1.set_xticks([0, 3000, 6000])
            ax2.set_xticks([0, 3000, 6000])
            ax3.set_xticks([0, 3000, 6000])
            
            ax1.set_xticklabels(['','',''])
            ax2.set_xticklabels(['','',''])            
            
        
        
    return(fig, ax1, ax2, ax3)

def filter_loss(loss, vals = 15):
    nloss = np.zeros_like(loss)
    nloss[0] = loss[0]
    i_s = 1
    count = 1
    for i in range(len(loss)-1):
        ip = i+1
        if ip//15 == ip/15.:
            #print(str(i_s) + '  - ' +str(i))
            nloss[count] = np.mean(loss[i_s:i])
            count += 1
            i_s = i
    nloss = nloss[0:count]
    return(nloss)

import math as m
def low_pass(x, ints=5):
    y = np.copy(x)
    for ix in range(len(x)):
        lmin = np.max((0, ix-ints))
        lmax = np.min((len(x), ix+ints))
        y[ix] = np.mean(x[lmin:lmax])
    return(y)

def to_polar(Traj):
    x = Traj[:,0,:]
    y = Traj[:,1,:]
    z = Traj[:,2,:]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2) 
    elev = 0.5*np.pi-np.arctan(z/np.sqrt(XsqPlusYsq)) 
    maskE=  elev>3.141592
    elev[maskE] = 0
    az = np.arctan2(y,x)
    maskAz =  az<-np.pi/2
    az[maskAz] += 2*np.pi
    return(r, az, elev)