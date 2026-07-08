import numpy as np
import math as m
import lib_rnns as lr
import torch
import modules2 as md
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from mpl_toolkits.basemap import Basemap
hidden_size =   150
#%%

def dynamics(time, k0, corr, vecM2):
    x = np.zeros((len(time), len(k0)))
    x[0,:] = k0
    dt = time[1]-time[0]
    for it, t in enumerate(time[:-1]):
        delta = np.sum((x[it,:]*vecM2)**2)
        prim = lr.Prime(0,delta)
        x[it+1,:] = x[it,:] + dt*(-x[it,:] + prim*np.dot(corr, x[it,:]))
    return(x)

def dynamics_noise(time, k0, corr, vecM2, alpha):
    x = np.zeros((len(time), len(k0)))
    x[0,:] = k0
    dt = time[1]-time[0]
    for it, t in enumerate(time[:-1]):
        delta = np.sum((x[it,:]*vecM2)**2)
        prim = lr.Prime(0,delta)
        x[it+1,:] = x[it,:] + dt*(-x[it,:] + prim*np.dot(corr, x[it,:]))+np.sqrt(dt)*alpha*np.random.randn(len(x[it,:]))
    return(x)
    
def dynamics_rnn(time, k0, M, N):
    x = np.zeros((len(time), len(k0)))
    x[0,:] = k0
    dt = time[1]-time[0]
    for it, t in enumerate(time[:-1]):
        dd, e =def_field_net(x[it,:], M, N)
        x[it+1,:] = x[it,:] + dt*dd
    return(x)
    
def def_field(k0, corr, vecM2):
    delta = np.sum((k0*vecM2)**2)
    prim = lr.Prime(0,delta)
    sol = -k0 + prim*np.dot(corr, k0)
    E = np.sum(sol**2)
    return(sol, E)

def def_field_net(k0, M, N):    
    sol = -k0 
    
    sol[0] +=   np.mean(N[0,:]*np.tanh(np.dot(M,k0)))
    sol[1] +=   np.mean(N[1,:]*np.tanh(np.dot(M,k0)))
    sol[2] +=   np.mean(N[2,:]*np.tanh(np.dot(M,k0)))
    
    E = np.sqrt(np.sum(sol**2))
    return(sol, E)

def def_field_net2(k0, M, N, hidden_size=150):    
    sol = -np.dot(M,k0) + np.dot(np.dot(M,N),np.tanh(np.dot(M,k0)))/hidden_size
    sol2 = np.zeros((3,))
    #print(np.shape(sol[:,0]*M[:,0]))
    if len(np.shape(sol))>1:
        sol2[0] =  np.mean(sol[:,0]*M[:,0])/np.mean(M[:,0]**2)
        sol2[1] =  np.mean(sol[:,0]*M[:,1])/np.mean(M[:,1]**2)
        sol2[2] =  np.mean(sol[:,0]*M[:,2])/np.mean(M[:,2]**2)
    else:
        sol2[0] =  np.mean(sol*M[:,0])/np.mean(M[:,0]**2)
        sol2[1] =  np.mean(sol*M[:,1])/np.mean(M[:,1]**2)
        sol2[2] =  np.mean(sol*M[:,2])/np.mean(M[:,2]**2)
    E = np.sqrt(np.sum(sol2**2))
    return(sol2, E)

def ellipsoid(th, p, R1, R2, R3):
    x = R1*np.sin(p)*np.sin(th)
    y = R2*np.sin(p)*np.sin(th)
    z = R3*np.cos(p)
    arr =np.array((x,y,z))
    return(arr)
    
def sph2cart(direc):
    sol = np.zeros(3,)
    sol[0] = direc[0]*np.sin(direc[2])*np.cos(direc[1])
    sol[1] = direc[0]*np.sin(direc[2])*np.sin(direc[1])
    sol[2] = direc[0]*np.cos(direc[2])
    return(sol)

def cart2sph(direc):

    x=direc[0]
    y=direc[1]
    z=direc[2]
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = 0.5*np.pi-m.atan(z/m.sqrt(XsqPlusYsq))     # theta
    if elev>3.141592:
        elev=0.
    az = m.atan2(y,x)                           # phi
    if az<-np.pi/2:
        az+=2*np.pi
        
    return r, az, elev

def plane_orthogonal(Ns, point,  R1, R2, R3, point_a=np.zeros(3,)):
    v1 = np.array((point[1], point[0], 0))
    v1 = v1/np.sqrt(np.sum(v1**2))
    v2 = np.array((point[0]*point[2], point[1]*point[2], -point[0]**2-point[1]**2))
    v2 = v2/np.sqrt(np.sum(v2**2))
    
    points = np.linspace(0.001, 2*np.pi, Ns)
    sols = np.zeros(Ns)
    basis_t = np.zeros(Ns)
    basis_p = np.zeros(Ns)
    for iP, P in enumerate(points):

        if iP == 0:
            direc = v1*np.cos(P)+v2*np.sin(P)+point_a
            rSS = np.linspace(0,2,200)
            dist = np.zeros(len(rSS))
            PP = np.random.rand()*2*np.pi
            for irS, rS in enumerate(rSS):
                cord = rS*v1*np.cos(PP)+rS*v2*np.sin(PP)+point_a
                r_d, t_d, p_d = cart2sph(cord)
                sol = ellipsoid(t_d, p_d, R1, R2, R3)
                dist[irS] = np.sum((sol-cord)**2)
            rS = rSS[np.nanargmin(dist)]
            print(rS)
        direc = rS*v1*np.cos(P)+rS*v2*np.sin(P)+point_a
        r_d, t_d, p_d = cart2sph(direc)
        sol = ellipsoid(t_d, p_d, R1, R2, R3)
        r_s2, t_d2, p_d2 = cart2sph(sol)
        sols[iP] = r_s2
        basis_t[iP] = t_d
        basis_p[iP] = p_d
    return(sols, basis_t, basis_p)

def initialize(rank,hidden_size, input_size, output_size, std_noise_rec, alpha, rho, repeat=False):
    
    sigma_mn = 0.85
    dtype = torch.FloatTensor  
    z_i =  sigma_mn*np.random.randn(hidden_size,rank).T
    mrec_i = z_i+np.random.randn(hidden_size,rank).T
    nrec_i = z_i+np.random.randn(hidden_size,rank).T
    mrec_i += np.random.randn(hidden_size,rank).T
    nrec_i += np.random.randn(hidden_size,rank).T
    mrec_I = torch.from_numpy(mrec_i.T).type(dtype)
    nrec_I = torch.from_numpy(nrec_i.T).type(dtype)
    
    #initial inputs and outputs fixed
    inp_i = np.random.randn(hidden_size,input_size).T
    randm_coeff = np.random.randn(1,2)/sigma_mn
    inp_i[0,:] = np.dot(randm_coeff, nrec_i[0:2,:])[0,:]
    inp_i[0,:] = inp_i[0,:]/np.sqrt(2)
    randm_coeff = np.random.randn(1,rank//2)/sigma_mn
    inp_i[1,:] = np.dot(randm_coeff, nrec_i[2:,:])[0,:]
    inp_i[1,:] = inp_i[1,:]/np.sqrt(rank//2)
    
    inp_i += np.random.randn(hidden_size,1).T
    
    out_i = np.random.randn(hidden_size, output_size)
    randm_coeff = np.random.randn(1,rank)/sigma_mn
    out_i[:,0] = np.dot(randm_coeff, mrec_i)[0,:]
    out_i[:,0] = out_i[:,0]/np.sqrt(rank)
    out_i += np.random.randn(hidden_size,output_size)
    
    
    inp_I = torch.from_numpy(inp_i).type(dtype)
    out_I = torch.from_numpy(out_i).type(dtype)
    
    net_low = md.LowRankLeakyNoisyRNN(input_size, hidden_size, output_size, std_noise_rec, 
                                      alpha, rank=rank, rho=rho, train_wi = False, 
                                      train_wo = False, train_h0=True, initial_wi=inp_I, 
                                      initial_wo=out_I, initial_mrec=mrec_I, initial_nrec=nrec_I)
    if repeat==False:
        net_low.load_state_dict(torch.load("rsg_rank"+str(rank)+"_prog_"+str(4)+"_rSSg_4int.pt", map_location='cpu'))
    else:
        net_low.load_state_dict(torch.load("rsg_rank"+str(rank)+"_prog_"+str(4)+"_rep_"+str(repeat)+"_rSSg_4int_2steps.pt", map_location='cpu'))
    return(net_low)

def initializeMWG( rank,hidden_size, input_size, output_size, std_noise_rec, alpha, rho, repeat=False, i=4 ):
    sigma_mn = 0.85
    dtype = torch.FloatTensor  
    z_i =  sigma_mn*np.random.randn(hidden_size,rank).T
    mrec_i = z_i+np.random.randn(hidden_size,rank).T
    nrec_i = z_i+np.random.randn(hidden_size,rank).T
    mrec_i += np.random.randn(hidden_size,rank).T
    nrec_i += np.random.randn(hidden_size,rank).T
    mrec_i = mrec_i/np.sqrt(hidden_size)
    nrec_i = nrec_i/np.sqrt(hidden_size)
    
    mrec_I = torch.from_numpy(mrec_i.T).type(dtype)
    nrec_I = torch.from_numpy(nrec_i.T).type(dtype)
    
    #initial inputs and outputs fixed
    inp_i = np.random.randn(hidden_size,input_size).T
    randm_coeff = np.random.randn(1,2)/sigma_mn
    inp_i[0,:] = np.dot(randm_coeff, nrec_i[0:2,:])[0,:]
    inp_i[0,:] = inp_i[0,:]/np.sqrt(2)
    randm_coeff = np.random.randn(1,rank//2)/sigma_mn
    inp_i[1,:] = np.dot(randm_coeff, nrec_i[2:,:])[0,:]
    inp_i[1,:] = inp_i[1,:]/np.sqrt(rank//2)
    
    inp_i += np.random.randn(hidden_size,1).T
    
    out_i = np.random.randn(hidden_size, output_size)
    randm_coeff = np.random.randn(1,rank)/sigma_mn
    out_i[:,0] = np.dot(randm_coeff, mrec_i)[0,:]
    out_i[:,0] = out_i[:,0]/np.sqrt(rank)
    out_i += np.random.randn(hidden_size,output_size)
    out_i = out_i/hidden_size
    
    inp_I = torch.from_numpy(inp_i).type(dtype)
    out_I = torch.from_numpy(out_i).type(dtype)
    
    net_low = mdn.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, 
                                  alpha, rank=rank, rho=rho, train_wi = False, 
                                  train_wo = False, train_h0=True, wi_init=inp_I, 
                                  wo_init=out_I, m_init=mrec_I, n_init=nrec_I)
    if repeat==False:
        net_low.load_state_dict(torch.load("MWG_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(repeat)+".pt", map_location='cpu'))
    else:
        net_low.load_state_dict(torch.load("MWG_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(repeat)+".pt", map_location='cpu'))
    return(net_low)

def Rs_ellipsoid(E, theta, phi, rs):
    th = 0
    p = np.pi/2    
    R1 = rs[np.argmin(E[:,np.argmin(np.abs(th-theta)), np.argmin(np.abs(p-phi))])]

    th = np.pi/2
    p=np.pi/2 
    R2 = rs[np.argmin(E[:,np.argmin(np.abs(th-theta)), np.argmin(np.abs(p-phi))])]
    
    th=0
    p = 0
    R3 = rs[np.argmin(E[:,np.argmin(np.abs(th-theta)), np.argmin(np.abs(p-phi))])]
    return(R1, R2, R3)
    
def give_fields_Es(rs, theta, phi, M, N, corr, vecM2, verbal=False, hidden_size=hidden_size):
    E = np.zeros((len(rs), len(theta), len(phi)))
    UX = np.zeros((len(rs), len(theta), len(phi), 3))
    
    En = np.zeros((len(rs), len(theta), len(phi)))
    UXn = np.zeros((len(rs), len(theta), len(phi), 3))
    
    for ir, r in enumerate(rs):
        if verbal:
            if 100*(ir/len(rs))==100*ir//len(rs):
                print('Accomplished: '+str(100*ir//len(rs))+' %')
        for it, th in enumerate(theta):
            for ip , p in enumerate(phi):
                k0 = r*np.array((np.sin(p)*np.cos(th), np.sin(p)*np.sin(th), np.cos(p) ))
                UX[ir, it, ip,:], E[ir, it, ip] =  def_field(k0, corr, vecM2)
                UXn[ir, it, ip,:], En[ir, it, ip] =  def_field_net2(k0, M, N, hidden_size=hidden_size)
    return(UX, E, UXn, En)
    
def give_manif(theta, phi, rs, E, UX, En, UXn):
    np.seterr(divide='ignore', invalid='ignore')
    E_manif = np.zeros((len(theta), len(phi)))
    U_manif = np.zeros_like(E_manif)
    V_manif = np.zeros_like(E_manif)
    
    E_manif_net = np.zeros((len(theta), len(phi)))
    U_manif_net = np.zeros_like(E_manif)
    V_manif_net = np.zeros_like(E_manif)
    R_manif = np.zeros((len(theta), len(phi)))
    R_manif_net = np.zeros((len(theta), len(phi)))
    
    for it, th in enumerate(theta):
        for ip , p in enumerate(phi):
            ix = np.argmin(E[:,it, ip])
            R = rs[ix]
            E_manif[it, ip] = E[ix,it,ip] 
            R_manif[it, ip] = R
            
            vec = UX[ix,it,ip,:]
            
            X = rs[ix]*np.sin(p)*np.cos(th)
            Y = rs[ix]*np.sin(p)*np.sin(th)
            Z = rs[ix]*np.cos(p)
            vec_r = np.array((X, Y, Z))
            vec_r = vec_r/np.sqrt(np.sum(vec_r**2))
            vec_theta = np.array((-Y, X, 0))
            #if np.sqrt(np.sum(vec_theta**2))>0:
            vec_theta = vec_theta/np.sqrt(np.sum(vec_theta**2))
            vec_phi   = np.array((X*Z, Y*Z, -X**2-Y**2))
            #if np.sqrt(np.sum(vec_phi**2))    >0:
            vec_phi = vec_phi/np.sqrt(np.sum(vec_phi**2))        
            
            U_manif[it, ip] =np.sum(vec*vec_theta)  
            V_manif[it, ip] = np.sum(vec*vec_phi)
            
            ix = np.argmax(np.diff(np.sign(np.diff(En[:,it, ip])))>0)+1#np.argmin(R)
            if ix==1:
                print('attention. Maybe r bounds are too narrow')
                plt.plot(rs, En[:,it, ip])
                
            R = rs[ix]
            E_manif_net[it, ip] = En[ix,it,ip] 
            R_manif_net[it, ip] = R
            
            vec = UXn[ix,it,ip,:]
            
            X = rs[ix]*np.sin(p)*np.cos(th)
            Y = rs[ix]*np.sin(p)*np.sin(th)
            Z = rs[ix]*np.cos(p)
            vec_r = np.array((X, Y, Z))
            vec_r = vec_r/np.sqrt(np.sum(vec_r**2))
            vec_theta = np.array((-Y, X, 0))
            if np.sqrt(np.sum(vec_theta**2))>0:
                vec_theta = vec_theta/np.sqrt(np.sum(vec_theta**2))
            
            vec_phi   = np.array((X*Z, Y*Z, -X**2-Y**2))
            if np.sqrt(np.sum(vec_phi**2))    >0:
                vec_phi = vec_phi/np.sqrt(np.sum(vec_phi**2))      
            
            U_manif_net[it, ip] = np.sum(vec*vec_theta)  
            V_manif_net[it, ip] = np.sum(vec*vec_phi)
    return(E_manif, R_manif, U_manif, V_manif, E_manif_net, R_manif_net, U_manif_net, V_manif_net)

#%%
def project_sin(theta, phi, E_manif, U_manif, V_manif):
    new_E_manif = np.nan*np.ones_like(E_manif)
    count = np.ones_like(E_manif)
    new_U_manif = np.nan*np.zeros_like(U_manif)
    new_V_manif = np.nan*np.zeros_like(V_manif)
    for ip, p in enumerate(phi):
        for it, th in enumerate(theta):
            new_th =  (th-np.mean(theta))*np.sin(p)+np.mean(theta)
            new_ith = np.argmin(np.abs(new_th-theta))
            count[new_ith, ip] += 1
            new_E_manif[new_ith, ip] = E_manif[it, ip]
            new_U_manif[new_ith, ip] = U_manif[it, ip]
            new_V_manif[new_ith, ip] = V_manif[it, ip]
    new_E_manif = new_E_manif/count
    new_U_manif = new_U_manif/count
    new_V_manif = new_V_manif/count
    return(new_E_manif, new_U_manif, new_V_manif)

def plot_field(E_manif_net, En, U_manif_net, V_manif_net, theta, phi, rs, lw=1, cb=True, alpha = 1, s_fp = 70):
    PS, TH = np.meshgrid(phi, theta) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mins = []
    Efake = np.tile(E_manif_net, [3,3])
    
    nums=8
    for it, th in enumerate(theta):
        for ip , p in enumerate(phi):
            sub = Efake[(it+len(theta)-nums):(it+len(theta)+nums),(ip+len(phi)-nums):(ip+len(phi)+nums)]
            if E_manif_net[it,ip]==np.nanmin(sub):
                if it>0 and it<len(theta)-1:
                    if np.isnan(E_manif_net[it+1,ip])==False and np.isnan(E_manif_net[it-1,ip])==False:
                        ix = np.argmax(np.diff(np.sign(np.diff(En[:,it, ip])))>0)+1  
                        mins.append([ix, it, ip])
                    
    plt.pcolor(TH, PS, np.log(E_manif_net), alpha=alpha, vmin = -5, vmax=-1, shading='auto')
    if cb:
        cbar = plt.colorbar()
        cbar.set_ticks([-5, -3, -1])
    plt.streamplot(theta, phi, U_manif_net.T, V_manif_net.T, color='w', linewidth=lw)
    #for x in range(len(mins)):
    #    plt.scatter(theta[mins[x][1]], phi[mins[x][2]], color='w', edgecolor='k', s=s_fp)
    #plt.plot(theta, 0.5*np.pi*theta/theta, '--',c='w', lw=2)
    
    #fp = np.array((rs[mins[0][0]], theta[mins[0][1]], phi[mins[0][2]]))
    plt.xlim(np.min(theta), np.max(theta))
    plt.ylim(np.min(phi), np.max(phi))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')
    
    return(fig, ax)

#%%
def plot_manifold(R_manif, theta, phi, n_c=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', azim=-141, elev = 34)
    for it, th in enumerate(theta):
        if n_c ==0:
            for ip , p in enumerate(phi[::4]):
                dot = sph2cart(np.array((R_manif[it, ip], th, p)))
                if dot[0]<0 and dot[1]<0 and dot[2]>0:
                    ax.scatter(dot[0], dot[1], dot[2], color='k')
                else:
                    ax.scatter(dot[0], dot[1], dot[2], color=[0.3, 0.3, 0.3], s=2)
        else:
            int_c = len(theta)//n_c
            for ip , p in enumerate(phi[::int_c]):
                dot = sph2cart(np.array((R_manif[it, ip], th, p)))
                if dot[0]<0 and dot[1]<0 and dot[2]>0:
                    ax.scatter(dot[0], dot[1], dot[2], color='k')
                else:
                    ax.scatter(dot[0], dot[1], dot[2], color=[0.3, 0.3, 0.3], s=2)
    ax.scatter(0,0,0, color='r')
    
    #ax.set_xticks([-1,0,1])
    #ax.set_yticks([-1,0,1])
    #ax.set_zticks([-1,0,1])
    
    #ax.set_xlim([-1,1])
    #ax.set_ylim([-1,1])
    #ax.set_zlim([-1,1])
    plt.xlabel(r'$\kappa_1$')
    plt.ylabel(r'$\kappa_2$')
    ax.set_zlabel(r'$\kappa_3$')
    ax.dist=12
    return(fig, ax)

#%%
def plot_dist_to_manif(R_manif, theta, phi, rank, corr, vecM2,M, N, mf, R=False):

    time = np.linspace(0, 12, 2000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    trials = 30
    fp = np.zeros((trials, rank))
    if R==False:
        R = 0.9
    
    
    
    for tr in range(trials):
        R = 0.8*np.random.rand()+0.3
        th = 10*np.random.randn()
        th2 = 10*np.random.randn()
        k0 = R*np.array((np.cos(th)*np.cos(th2), np.cos(th)*np.sin(th2), np.sin(th)   ))
        if mf ==True:
            k = dynamics(time, k0, corr, vecM2)
        else:
            k = dynamics_rnn(time, k0, M, N)
        dis = np.zeros(len(time))
        for it, t in enumerate(time):
            rR, rT, rP = cart2sph( k[it,:])
            dis[it] = rR - R_manif[np.argmin(np.abs(rT-theta)), np.argmin(np.abs(rP-phi))]
        plt.plot(time, dis, lw=0.5, color=[0.5, 0.5, 0.5])
        if tr==0:
            disM = dis/trials
        else:
            disM +=dis/trials
        
        fp[tr,:] = k[-1,:]
    plt.plot(time, disM, lw=2, color='k')
    plt.xlabel(r'time')
    plt.ylabel(r'$r\left(t\right)-r_m\left(\theta(t), \phi(t)\right)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return(fig, ax, dis, k)

#%%
def randomtraj(R_manif_net, theta, phi, corr, vecM2,M, N, mf, Rr=True):
    time = np.linspace(0, 100, 3000)
    lr.set_plot()
    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2  # height in inches
    fig_size =  [fig_width,fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d', azim=-63, elev=14)
    trials = 30
    
    R = 0.9
    fx_f = []
    
    for tr in range(trials):
        Ith = np.random.randint(0,len(theta))
        Ip = np.random.randint(0,len(phi))
        if Rr==True:
            R = R_manif_net[Ith, Ip]
        else:
            R = Rr
        th = theta[Ith]
        th2 = phi[Ip]
        k0 = R*np.array((np.cos(th)*np.cos(th2), np.cos(th)*np.sin(th2), np.sin(th)   ))
        if mf:
            k = dynamics(time, k0, corr, vecM2)
        else:
            k = dynamics_rnn(time, k0, M, N)
    
    
        ax.plot(k[:,0], k[:,1],k[:,2], lw=0.5, color=[0.5, 0.5, 0.5])
        argtim = np.argmin(np.abs(time - 2))    
        ax.scatter(k[argtim,0], k[argtim,1], k[argtim,2], s = 15, color=[0.6, 0.6, 0.6], edgecolor='k')
        
        argtim = np.argmin(np.abs(time - 4))    
        ax.scatter(k[argtim,0], k[argtim,1], k[argtim,2], s = 15, color=[0.3, 0.3, 0.3], edgecolor='k')
        
        ax.scatter(k[0,0], k[0,1], k[0,2], s = 20, color='w', edgecolor='k')
        ax.scatter(k[-1,0], k[-1,1], k[-1,2], s = 50, color='k', edgecolor='w')
        fx_f.append([k[-1,0], k[-1,1], k[-1,2]])
    
    for tr in range(trials):
        ax.scatter(fx_f[tr][0], fx_f[tr][1],fx_f[tr][2], s = 50, color='k', edgecolor='w')
        
    pis = np.linspace(0, 2*np.pi)
    R=1.25*np.mean(R_manif_net)
    ax.plot(R*np.cos(pis), R*np.sin(pis), 0*np.sin(pis), 'k', lw=0.7)
    ax.plot(R*np.cos(pis), 0*np.cos(pis), R*np.sin(pis), 'k', lw=0.7)
    
        
    plt.xlabel(r'$\kappa_1$')
    plt.ylabel(r'$\kappa_2$')
    ax.set_zlabel(r'$\kappa_3$')
    ax.dist= 10.
    ax.set_xticks([-0.5,0.5])
    ax.set_yticks([-0.5,0.5])
    ax.set_zticks([-0.5,0.5])
    plt.tight_layout()
    return(fig, ax)

#%% 
def get_avgtraj(net_low, ints, trials_test, dt, Nt, R_on, dela, phi, theta, M, N, align_set = True, show_set = True):
    
    I1 = np.zeros((Nt, len(ints)))
    I2 = np.zeros((Nt, len(ints)))
    O = np.zeros((Nt, len(ints)))
    
    Sol_rad  = np.zeros((3,  Nt, len(ints)))
    Sol      = np.zeros_like(Sol_rad)
    exc = np.zeros((1, Nt, len(ints)))
    factor = 1
    
    for xx in range(len(ints)):
        if align_set:
            input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = lr.create_inp_out_rSSg_2in(trials_test, Nt, ints//dt, 
                                                                            R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela)
        else:
            input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = lr.create_inp_out_rSSg_2in(trials_test, Nt, ints//dt, 
                                                                            R_on, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela)
        if show_set ==False:
            input_tr[:,:,1] = 0.*input_tr[:,:,1]
        outp, traj = net_low.forward(input_tr, return_dynamics=True)
        outp = outp.detach().numpy()
        traj = traj.detach().numpy()
        
        #avg_outp = np.mean(outp[:,:,0],0)
        #std_outp = np.std(outp[:,:,0],0)
         
        proj_m =  np.dot(M.T,np.mean(traj[:,:,:],0).T).T/np.diag(np.dot(M.T,M))#hidden_size
        exc = np.std(np.dot(proj_m, M.T)-np.mean(traj[:,:,:],0),1)
        proj_m = proj_m.T
        
        I1[:,xx] = input_tr[0,:,0].detach().numpy()
        I2[:,xx] = input_tr[0,:,1].detach().numpy()
        O[:,xx] = output_tr[0,:,0].detach().numpy()
        Sol[:,:,xx] = proj_m[:,0:Nt]
        for it in range(Nt):
            dot = proj_m[:,it]
            sol = cart2sph(dot)
            Sol_rad[:,it, xx] = sol
    return(Sol, Sol_rad, exc, I1, I2, O, proj_m)

#%% 
def get_avgtraj_pert1(net_low, ints, trials_test, dt, Nt, R_on, dela, phi, theta, M, N, align_set = True, show_set = True):
    
    I1 = np.zeros((Nt, len(ints)))
    I2 = np.zeros((Nt, len(ints)))
    O = np.zeros((Nt, len(ints)))
    
    Sol_rad  = np.zeros((3,  Nt, len(ints)))
    Sol      = np.zeros_like(Sol_rad)
    exc = np.zeros((1, Nt, len(ints)))
    factor = 1
    
    for xx in range(len(ints)):
        if align_set:
            input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = lr.create_inp_out_rSSg_2in(trials_test, Nt, ints//dt, 
                                                                            R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela)
        else:
            input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = lr.create_inp_out_rSSg_2in(trials_test, Nt, ints//dt, 
                                                                            R_on, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela)
        if show_set ==False:
            input_tr[:,:,1] = 0.*input_tr[:,:,1]
        outp, traj, noise = net_low.forward(input_tr, return_dynamics=True, return_noise=True)
        outp = outp.detach().numpy()
        traj = traj.detach().numpy()
        noise = noise.detach().numpy()
        #print(np.shape(traj))
        #avg_outp = np.mean(outp[:,:,0],0)
        #std_outp = np.std(outp[:,:,0],0)
         
        proj_m =  np.dot(M.T,np.mean(traj[:,:,:],0).T).T/np.diag(np.dot(M.T,M))#hidden_size
        exc = np.std(np.dot(proj_m, M.T)-np.mean(traj[:,:,:],0),1)
        proj_m = proj_m.T
        
        I1[:,xx] = input_tr[0,:,0].detach().numpy()
        I2[:,xx] = input_tr[0,:,1].detach().numpy()
        O[:,xx] = output_tr[0,:,0].detach().numpy()
        Sol[:,:,xx] = proj_m[:,0:Nt]
        for it in range(Nt):
            dot = proj_m[:,it]
            sol = cart2sph(dot)
            Sol_rad[:,it, xx] = sol
    return(Sol, Sol_rad, exc, I1, I2, O, proj_m, noise, traj, outp)
#%%

def clean_theta(x):
    ix_ = np.arange(len(x)-1)+1
    idcs = ix_[np.abs(np.diff(x))>0.95*2*np.pi]
    x[idcs] = np.nan
    return(x)

def clean_phi(x):
    ix_ = np.arange(len(x)-1)+1
    idcs = ix_[np.abs(np.diff(x))>0.95*np.pi]
    x[idcs] = np.nan
    return(x)

def map_traj_sin(traj, mu_th):
    new_traj = np.copy(traj)
    for xx in range(np.shape(traj)[2]):
        for it in range(np.shape(traj)[1]):
            new_traj[1,it,xx] = (new_traj[1,it,xx]-mu_th)*np.sin(new_traj[2,it,xx])+mu_th
    return(new_traj)
#%%

def plot_traj_field(Sol_rad, xx, t1, t2):
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
    plt.plot(clean_theta(Sol_rad[1,t1:t2,xx]), clean_phi(Sol_rad[2,t1:t2,xx]), lw=1.5, color='k', zorder=1)
    plt.plot(clean_theta(Sol_rad[1,t1:t2,xx]), clean_phi(Sol_rad[2,t1:t2,xx]), color=cls[xx,:], zorder=2)
    plt.scatter(clean_theta(Sol_rad[1,t1:t2:20,xx]), clean_phi(Sol_rad[2,t1:t2:20,xx]), s=8, edgecolor='w', color=cls[xx,:], zorder=4)
    return()
#%%

def rotate_flow(Enet, theta,  Dt):
    Dt_ = Dt*np.pi/180
    Enet2= np.zeros_like(Enet)

    iDt = np.int(Dt_/(theta[1]-theta[0]))
    Enet2 = np.copy(Enet)
    Enet2 = np.roll(Enet2, iDt, axis=0)
    return(Enet2)

def rotate_trajG(traj1, theta, dt):    
    traj1F = traj1 + dt
    mask_b = traj1F>np.max(theta)
    if sum(mask_b)>0:
        traj1F[mask_b] = traj1F[mask_b]-360
    mask_c = traj1F<np.min(theta)
    if sum(mask_c)>0:
        traj1F[mask_c] = traj1F[mask_c]+360
    return(traj1F)

#%%
def create_gif_SG(Sol_rad, th0, I2, O, E_manif_net, folderP, last=False, l0 = False, v0=False, phi=-2, theta=-2, label=[0]):
    plt.figure()
    #ax = fig.add_subplot(111)
    if np.sum(phi+2)==0:

        Phi = np.linspace(0,np.pi,70)#50
    else:
        
        Phi=phi
    if np.sum(theta+2)==0:
        Theta = np.linspace(-np.pi/2, 2*np.pi+th0, 70)#60
    else:
        Theta = theta
    
    
    nrows, ncols = (len(Theta),len(Phi))
    
    X = np.linspace(0,360,nrows)
    Y = np.linspace(-90,90,ncols)
    lon, lat = np.meshgrid(X, Y)
    
    cls = give_cls()
    
    DT0 = 2
    DT = DT0
    iDT = 1
    if last==False:
        for tsteps in range(170):
            
            t1 = np.argmax(I2[:,0])-1#137
            t2 = t1+DT
    
            xx = 0
            for xx in range(4):
                rt2 = np.min((t2, np.argmax(O[:,xx])))
                if xx ==0:
                    lon_tr = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)/4
                    #lon_tr2 = tm.rotate_trajG(lon_tr, theta*180/np.pi, Dthet)
                    lat_tr = (Sol_rad[2,rt2,xx]*180/np.pi - 90)/4
                else:
                    lon_tr += ((Sol_rad[1,rt2,xx]*180/np.pi)+90)/4
                    lat_tr += (Sol_rad[2,rt2,xx]*180/np.pi - 90)/4
                #x1, y1 = map(lon_tr,lat_tr)
            
            map = Basemap(projection='ortho', lat_0=lat_tr-20, lon_0=lon_tr)
            # draw lat/lon grid lines every 30 degrees.
            map.drawmeridians(np.arange(0, 360, 60),c='w')
            map.drawparallels(np.arange(-90, 90, 30), color='w')
            
            lon = Theta*180/np.pi+90
            lat = (Phi*180/np.pi)-90
            Lon, Lat = np.meshgrid(lon , lat)
            # compute native map projection coordinates of lat/lon grid.
            x, y = map(Lon, Lat)
            
            cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0, shading='auto')
            cs.set_edgecolor('face')
            
            for xx in range(4):
                rt2 = np.min((t2, np.argmax(O[:,xx])))
                
                lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
                #lon_tr2 = tm.rotate_trajG(lon_tr, theta*180/np.pi, Dthet)
                lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
                x1, y1 = map(lon_tr,lat_tr)
                #map.streamplot(lon, lat, U_manif_net, V_manif_net, latlon=True)
                X1 = np.hstack((x1, x1[-1]))
                mask = np.log(np.abs(np.diff(X1)))>20
                
                x1[mask] = np.nan
                y1[mask] = np.nan
                map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
                map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
                
                map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
                
                map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
                if rt2==t2:
                    map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
                else:
                    map.scatter(x1[-1],y1[-1], s = 120, c = [0.7, 0.7, 0.7], edgecolor='k', marker='^', zorder=6)
                
                
        
            DT = DT  + iDT
            plt.title('t='+str(DT*10/1000)+'s')
        
            plt.savefig(folderP+'prj_'+str(tsteps)+'.png', dpi=200)
            plt.savefig(folderP+'prj_'+str(tsteps)+'_thesis.pdf', dpi=200)
            
            if len(label)>1:
                plt.savefig(folderP+label, dpi=200)

            
            plt.show()
    else:
        tsteps= 170
        
        t1 = np.argmax(I2[:,0])-1#137
        t2 = t1+tsteps*DT

        xx = 0
        for xx in range(4):
            rt2 = np.min((t2, np.argmax(O[:,xx])))
            if xx ==0:
                lon_tr = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)/4
                #lon_tr2 = tm.rotate_trajG(lon_tr, theta*180/np.pi, Dthet)
                lat_tr = (Sol_rad[2,rt2,xx]*180/np.pi - 90)/4
            else:
                lon_tr += ((Sol_rad[1,rt2,xx]*180/np.pi)+90)/4
                lat_tr += (Sol_rad[2,rt2,xx]*180/np.pi - 90)/4
            #x1, y1 = map(lon_tr,lat_tr)
        print(lat_tr-20)
        print(lon_tr)
        if l0==False:
            map = Basemap(projection='ortho', lat_0=lat_tr-20, lon_0=lon_tr)
        else:
            map = Basemap(projection='ortho', lat_0=l0+40, lon_0=v0)
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0, 360, 60),c='w')
        map.drawparallels(np.arange(-90, 90, 30), color='w')
        
        lon = Theta*180/np.pi+90
        lat = (Phi*180/np.pi)-90
        Lon, Lat = np.meshgrid(lon , lat)
        # compute native map projection coordinates of lat/lon grid.
        x, y = map(Lon, Lat)
        
        cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.6, linewidth=0)
        cs.set_edgecolor('face')
        #plt.colorbar(ticks=[-4, -2])
        
        
        for xx in range(4):
            rt2 = np.min((t2, np.argmax(O[:,xx])))
            
            lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
            #lon_tr2 = tm.rotate_trajG(lon_tr, theta*180/np.pi, Dthet)
            lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
            x1, y1 = map(lon_tr,lat_tr)
            #map.streamplot(lon, lat, U_manif_net, V_manif_net, latlon=True)
            X1 = np.hstack((x1, x1[-1]))
            mask = np.log(np.abs(np.diff(X1)))>20
            
            x1[mask] = np.nan
            y1[mask] = np.nan
            map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
            map.plot(x1,y1, lw = 4.1, c = 'w', zorder=1)
            
            
            map.scatter(x1[0],y1[0], s = 120, c = cls[xx,:], edgecolor='w', linewidth=2, zorder=5)
            
            map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
            if rt2==t2:
                map.scatter(x1[-1],y1[-1], s = 120, c = cls[xx,:], edgecolor='w', marker='^',linewidth=2, zorder=6)
            else:
                map.scatter(x1[-1],y1[-1], s = 120, c = [0.7, 0.7, 0.7], edgecolor='k', marker='^', linewidth=2,zorder=6)
            
            
    
        DT = DT  + iDT
        #plt.title('t='+str(DT*10/1000)+'s')
    
        plt.savefig(folderP+'prj_'+str(tsteps)+'_last.png', dpi=200)
        plt.savefig(folderP+'prj_'+str(tsteps)+'_thesis.pdf')
        plt.savefig(folderP+'prj_'+str(tsteps)+'_thesis.eps')
        
        if len(label)>1:
                plt.savefig(folderP+label, dpi=200)

        plt.show()
        
    #%%

def create_gif_SG_many(Sol_rad, rt2s, th0, I2, O, E_manif_net, folderP, phi=-2, theta=-2):
    plt.figure()
    #ax = fig.add_subplot(111)
    if np.sum(phi+2)==0:

        Phi = np.linspace(0,np.pi,70)#50
    else:
        
        Phi=phi
    if np.sum(theta+2)==0:
        Theta = np.linspace(-np.pi/2, 2*np.pi+th0, 70)#60
    else:
        Theta = theta
    
    
    nrows, ncols = (len(Theta),len(Phi))
    
    X = np.linspace(0,360,nrows)
    Y = np.linspace(-90,90,ncols)
    lon, lat = np.meshgrid(X, Y)
    
    
    trajs = np.shape(Sol_rad)[2]
    
    DT0 = 2
    DT = DT0
    iDT = 1
    for tsteps in range(220):
        
        t1 = 0#137
        t2 = t1+DT

        xx = 0
        for xx in range(trajs):
            rt2 = np.min((t2, int(rt2s[xx])))
            if xx ==0:
                lon_tr = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)/trajs
                #lon_tr2 = tm.rotate_trajG(lon_tr, theta*180/np.pi, Dthet)
                lat_tr = (Sol_rad[2,rt2,xx]*180/np.pi - 90)/trajs
            else:
                lon_tr += ((Sol_rad[1,rt2,xx]*180/np.pi)+90)/trajs
                lat_tr += (Sol_rad[2,rt2,xx]*180/np.pi - 90)/trajs
            #x1, y1 = map(lon_tr,lat_tr)
        #print(lat_tr)
        map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr)
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0, 360, 60),c='w')
        map.drawparallels(np.arange(-90, 90, 30), color='w')
        
        lon = Theta*180/np.pi+90
        lat = (Phi*180/np.pi)-90
        Lon, Lat = np.meshgrid(lon , lat)
        # compute native map projection coordinates of lat/lon grid.
        x, y = map(Lon, Lat)
        
        cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
        cs.set_edgecolor('face')
        
        for xx in range(trajs):
            cl = 0.7*(xx/trajs)*np.ones(3)+0.3*np.array((0.9, 0.1, 0.3))
            rt2 = np.min((t2, int(rt2s[xx])))
            
            lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
            #lon_tr2 = tm.rotate_trajG(lon_tr, theta*180/np.pi, Dthet)
            lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
            x1, y1 = map(lon_tr,lat_tr)
            #map.streamplot(lon, lat, U_manif_net, V_manif_net, latlon=True)
            X1 = np.hstack((x1, x1[-1]))
            mask = np.log(np.abs(np.diff(X1)))>20
            
            x1[mask] = np.nan
            y1[mask] = np.nan
            map.plot(x1,y1, lw = 3, c = cl, zorder=2)
            map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
            
            map.scatter(x1[0],y1[0], s = 100, c = cl, edgecolor='w', zorder=5)
            
            map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
            if rt2==t2:
                map.scatter(x1[-1],y1[-1], s = 150, c = cl, edgecolor='w', marker='^', zorder=6)
            else:
                map.scatter(x1[-1],y1[-1], s = 120, c = [0.7, 0.7, 0.7], edgecolor='k', marker='^', zorder=6)
            
            
    
        DT = DT  + iDT
        plt.title('t='+str(DT*10/1000)+'s')
    
        plt.savefig(folderP+'many_prj_'+str(tsteps)+'.png', dpi=200)
        plt.show()
        
#%%
def create_gif_RS(Sol_rad, th0, I1, I2, O, E_manif_net, folderP, last=False, phi=-2, theta=-2, lat_tr=False, lon_tr2=False):
    DT0 = 2
    DT = DT0
    iDT = 2
    t1 = 0#137
  
    if np.sum(phi+2)==0:

        Phi = np.linspace(0,np.pi,70)#50
    else:
        
        Phi=phi
    if np.sum(theta+2)==0:
        Theta = np.linspace(-np.pi/2, 2*np.pi+th0, 70)#60
    else:
        Theta = theta
    
    
    
    cls = give_cls()
    if last==False:
        for tsteps in range(190):
            t1 = 50#137
            t2 = t1+DT
                
            xx = 0
            Lon_tr = np.zeros(4)
            Lat_tr = np.zeros(4)
            for xx in range(4):
                rt2 = np.min((t2,np.argmax(I2[:,0])))
                
                Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
                Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
              
                print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
            print('-')
            if tsteps==0:
                plon_tr = np.mean(Lon_tr)
                plat_tr = np.mean(Lat_tr)
            eta = 0.95
            if np.std(Lon_tr)<20:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
            else:
                Lon_tr1 = Lon_tr
                #break
                Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
                Lon_tr2 = Lon_tr
                Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
                
                if np.std(Lon_tr1)<np.std(Lon_tr2):
                
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
                else:
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
            plon_tr = lon_tr
            print(lon_tr)
            
            if lat_tr==False:
                lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
                plat_tr = lat_tr        
                print(lat_tr)
                
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr)
            else:
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr2)
            # draw lat/lon grid lines every 30 degrees.
            map.drawmeridians(np.arange(0, 360, 60),c='w')
            map.drawparallels(np.arange(-90, 90, 30), color='w')
            
            lon = Theta*180/np.pi+90
            lat = (Phi*180/np.pi)-90
            Lon, Lat = np.meshgrid(lon , lat)
            # compute native map projection coordinates of lat/lon grid.
            x, y = map(Lon, Lat)
            
        
            cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
            cs.set_edgecolor('face')
            
            for xx in range(4):
                val = np.argmax(I1[:,xx])
                cI1 = I1
                cI1[val,xx] = -1
                
                rt2 = np.argmax(cI1[:,xx])-1#np.min((t2, np.argmax(I2[:,0])))
                
                lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
                lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
                print(lat_tr)
                x1, y1 = map(lon_tr,lat_tr)
                X1 = np.hstack((x1, x1[-1]))
                mask = np.log(np.abs(np.diff(X1)))>20
                
                x1[mask] = np.nan
                y1[mask] = np.nan
                map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
                map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
                
                map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
                
                map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
                map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
               
                
        
            DT = DT  + iDT
            plt.title('t='+str(DT*10/1000)+'s')
        
            plt.savefig(folderP+'prj2_'+str(tsteps)+'.png', dpi=200)
            plt.show()
    else:
        tsteps=190
        t1 = 50#137
        t2 = t1+tsteps*DT
            
        xx = 0
        Lon_tr = np.zeros(4)
        Lat_tr = np.zeros(4)
        for xx in range(4):
            val = np.argmax(I1[:,xx])
            cI1 = I1
            cI1[val,xx] = -1
                
            rt2 = np.argmax(cI1[:,xx])-10#np.min((t2, np.argmax(I2[:,0])))rt2 = np.argmax(I2[:,0])#np.min((t2,np.argmax(I2[:,0])))
            
            Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
            Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
            print(rt2)
            #print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
        print('-')
        plon_tr = np.mean(Lon_tr)
        plat_tr = np.mean(Lat_tr)
        eta = 0.95
        if np.std(Lon_tr)<20:
            lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
        else:
            Lon_tr1 = Lon_tr
            #break
            Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
            Lon_tr2 = Lon_tr
            Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
            
            if np.std(Lon_tr1)<np.std(Lon_tr2):
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
            else:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
        plon_tr = lon_tr
        print(lon_tr)
        
        
        lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
        plat_tr = lat_tr        
        print(lat_tr)

        map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr+30)
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0, 360, 60),c='w')
        map.drawparallels(np.arange(-90, 90, 30), color='w')
        
        lon = Theta*180/np.pi+90
        lat = (Phi*180/np.pi)-90
        Lon, Lat = np.meshgrid(lon , lat)
        # compute native map projection coordinates of lat/lon grid.
        x, y = map(Lon, Lat)
        
    
        cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
        cs.set_edgecolor('face')
        
        for xx in range(4):
            rt2 = np.argmax(cI1[:,xx])-1
            
            lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
            lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
            x1, y1 = map(lon_tr,lat_tr)
            X1 = np.hstack((x1, x1[-1]))
            mask = np.log(np.abs(np.diff(X1)))>20
            
            x1[mask] = np.nan
            y1[mask] = np.nan
            map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
            map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
            
            map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
            
            map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
            map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
           
            
    
        DT = DT  + iDT
    
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last.png', dpi=200)
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_RS_thesis.eps')
        print(folderP+'prj2_'+str(tsteps)+'_last_thesis_RS.pdf')
        plt.show()

#%%
def create_gif_RS2(Sol_rad, th0, I1, I2, O, E_manif_net, folderP, last=False, phi=-2, theta=-2, lat_tr=False, lon_tr2=False):
    DT0 = 2
    DT = DT0
    iDT = 2
    t1 = 0#137
  
    if np.sum(phi+2)==0:

        Phi = np.linspace(0,np.pi,70)#50
    else:
        
        Phi=phi
    if np.sum(theta+2)==0:
        Theta = np.linspace(-np.pi/2, 2*np.pi+th0, 70)#60
    else:
        Theta = theta
    
    
    
    cls = give_cls()
    if last==False:
        for tsteps in range(190):
            t1 = 50#137
            t2 = t1+DT
                
            xx = 0
            Lon_tr = np.zeros(4)
            Lat_tr = np.zeros(4)
            for xx in range(4):
                rt2 = np.min((t2,np.argmax(I2[:,0])))
                
                Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
                Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
              
                print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
            print('-')
            if tsteps==0:
                plon_tr = np.mean(Lon_tr)
                plat_tr = np.mean(Lat_tr)
            eta = 0.95
            if np.std(Lon_tr)<20:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
            else:
                Lon_tr1 = Lon_tr
                #break
                Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
                Lon_tr2 = Lon_tr
                Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
                
                if np.std(Lon_tr1)<np.std(Lon_tr2):
                
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
                else:
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
            plon_tr = lon_tr
            print(lon_tr)
            
            if lat_tr==False:
                lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
                plat_tr = lat_tr        
                print(lat_tr)
                
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr)
            else:
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr2)
            # draw lat/lon grid lines every 30 degrees.
            map.drawmeridians(np.arange(0, 360, 60),c='w')
            map.drawparallels(np.arange(-90, 90, 30), color='w')
            
            lon = Theta*180/np.pi+90
            lat = (Phi*180/np.pi)-90
            Lon, Lat = np.meshgrid(lon , lat)
            # compute native map projection coordinates of lat/lon grid.
            x, y = map(Lon, Lat)
            
        
            cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
            cs.set_edgecolor('face')
            
            for xx in range(4):
                val = np.argmax(I1[:,xx])
                cI1 = I1
                cI1[val,xx] = -1
                
                rt2 = np.argmax(cI1[:,xx])-1#np.min((t2, np.argmax(I2[:,0])))
                
                lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
                lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
                print(lat_tr)
                x1, y1 = map(lon_tr,lat_tr)
                X1 = np.hstack((x1, x1[-1]))
                mask = np.log(np.abs(np.diff(X1)))>20
                
                x1[mask] = np.nan
                y1[mask] = np.nan
                map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
                map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
                
                map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
                
                map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
                map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
               
                
        
            DT = DT  + iDT
            plt.title('t='+str(DT*10/1000)+'s')
        
            plt.savefig(folderP+'prj2_'+str(tsteps)+'.png', dpi=200)
            plt.show()
    else:
        tsteps=190
        t1 = 50#137
        t2 = t1+tsteps*DT
            
        xx = 0
        Lon_tr = np.zeros(4)
        Lat_tr = np.zeros(4)
        for xx in range(4):
            val = np.argmax(I1[:,xx])
            cI1 = I1
            cI1[val,xx] = -1
                
            rt2 = np.argmax(cI1[:,xx])-10#np.min((t2, np.argmax(I2[:,0])))rt2 = np.argmax(I2[:,0])#np.min((t2,np.argmax(I2[:,0])))
            
            Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
            Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
            print(rt2)
            #print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
        print('-')
        plon_tr = np.mean(Lon_tr)
        plat_tr = np.mean(Lat_tr)
        eta = 0.95
        if np.std(Lon_tr)<20:
            lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
        else:
            Lon_tr1 = Lon_tr
            #break
            Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
            Lon_tr2 = Lon_tr
            Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
            
            if np.std(Lon_tr1)<np.std(Lon_tr2):
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
            else:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
        plon_tr = lon_tr
        print(lon_tr)
        
        
        lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
        plat_tr = lat_tr        
        print(lat_tr)

        map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr+30)
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0, 360, 60),c='w')
        map.drawparallels(np.arange(-90, 90, 30), color='w')
        
        lon = Theta*180/np.pi+90
        lat = (Phi*180/np.pi)-90
        Lon, Lat = np.meshgrid(lon , lat)
        # compute native map projection coordinates of lat/lon grid.
        x, y = map(Lon, Lat)
        
    
        cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
        cs.set_edgecolor('face')
        
        for xx in range(4):
            rt2 = np.argmax(cI1[:,xx])-1
            rt1 = np.argmax(I2[:,xx])-1
            print('')
            print(rt2)
            print(rt1)
            print('')
            
            lon_tr = (Sol_rad[1,rt2:rt1,xx]*180/np.pi)+90
            lat_tr = Sol_rad[2,rt2:rt1,xx]*180/np.pi - 90
            x1, y1 = map(lon_tr,lat_tr)
            X1 = np.hstack((x1, x1[-1]))
            mask = np.log(np.abs(np.diff(X1)))>20
            
            x1[mask] = np.nan
            y1[mask] = np.nan
            map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
            map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
            
            map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
            
            map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
            map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
           
            
    
        DT = DT  + iDT
    
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last.png', dpi=200)
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_RS2_thesis.eps')
        print(folderP+'prj2_'+str(tsteps)+'_last_thesis_RS2.pdf')
        plt.show()

#%%
def create_gif_delay1(Sol_rad, th0, I1, I2, O, E_manif_net, folderP, last=False, phi=-2, theta=-2, lat_tr=False, lon_tr2=False):
    DT0 = 2
    DT = DT0
    iDT = 2
    t1 = 0#137
  
    if np.sum(phi+2)==0:

        Phi = np.linspace(0,np.pi,70)#50
    else:
        
        Phi=phi
    if np.sum(theta+2)==0:
        Theta = np.linspace(-np.pi/2, 2*np.pi+th0, 70)#60
    else:
        Theta = theta
    
    
    
    cls = give_cls()
    if last==False:
        for tsteps in range(190):
            t1 = 50#137
            t2 = t1+DT
                
            xx = 0
            Lon_tr = np.zeros(4)
            Lat_tr = np.zeros(4)
            for xx in range(4):
                rt2 = np.min((t2,np.argmax(I2[:,0])))
                
                Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
                Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
              
                print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
            print('-')
            if tsteps==0:
                plon_tr = np.mean(Lon_tr)
                plat_tr = np.mean(Lat_tr)
            eta = 0.95
            if np.std(Lon_tr)<20:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
            else:
                Lon_tr1 = Lon_tr
                #break
                Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
                Lon_tr2 = Lon_tr
                Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
                
                if np.std(Lon_tr1)<np.std(Lon_tr2):
                
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
                else:
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
            plon_tr = lon_tr
            print(lon_tr)
            
            if lat_tr==False:
                lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
                plat_tr = lat_tr        
                print(lat_tr)
                
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr)
            else:
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr2)
            # draw lat/lon grid lines every 30 degrees.
            map.drawmeridians(np.arange(0, 360, 60),c='w')
            map.drawparallels(np.arange(-90, 90, 30), color='w')
            
            lon = Theta*180/np.pi+90
            lat = (Phi*180/np.pi)-90
            Lon, Lat = np.meshgrid(lon , lat)
            # compute native map projection coordinates of lat/lon grid.
            x, y = map(Lon, Lat)
            
        
            cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
            cs.set_edgecolor('face')
            
            for xx in range(4):
                val = np.argmax(I1[:,xx])
                cI1 = I1
                cI1[val,xx] = -1
                
                rt2 = np.argmax(cI1[:,xx])-1#np.min((t2, np.argmax(I2[:,0])))
                
                lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
                lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
                print(lat_tr)
                x1, y1 = map(lon_tr,lat_tr)
                X1 = np.hstack((x1, x1[-1]))
                mask = np.log(np.abs(np.diff(X1)))>20
                
                x1[mask] = np.nan
                y1[mask] = np.nan
                map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
                map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
                
                map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
                
                map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
                map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
               
                
        
            DT = DT  + iDT
            plt.title('t='+str(DT*10/1000)+'s')
        
            plt.savefig(folderP+'prj2_'+str(tsteps)+'.png', dpi=200)
            plt.show()
    else:
        tsteps=190
        t1 = 50#137
        t2 = 100#t1+tsteps*DT
            
        xx = 0
        Lon_tr = np.zeros(4)
        Lat_tr = np.zeros(4)
        for xx in range(4):
            val = np.argmax(I1[:,xx])
            cI1 = I1
            cI1[val,xx] = -1
                
            rt2 = np.argmax(cI1[:,xx])-10#np.min((t2, np.argmax(I2[:,0])))rt2 = np.argmax(I2[:,0])#np.min((t2,np.argmax(I2[:,0])))
            
            Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
            Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
            print(rt2)
            #print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
        print('-')
        plon_tr = np.mean(Lon_tr)
        plat_tr = np.mean(Lat_tr)
        eta = 0.95
        if np.std(Lon_tr)<20:
            lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
        else:
            Lon_tr1 = Lon_tr
            #break
            Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
            Lon_tr2 = Lon_tr
            Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
            
            if np.std(Lon_tr1)<np.std(Lon_tr2):
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
            else:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
        plon_tr = lon_tr
        print(lon_tr)
        
        
        lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
        plat_tr = lat_tr        
        print(lat_tr)

        map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr+30)
        Lttr = lat_tr
        Lnoo = lon_tr
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0, 360, 60),c='w')
        map.drawparallels(np.arange(-90, 90, 30), color='w')
        
        lon = Theta*180/np.pi+90
        lat = (Phi*180/np.pi)-90
        Lon, Lat = np.meshgrid(lon , lat)
        # compute native map projection coordinates of lat/lon grid.
        x, y = map(Lon, Lat)
        
    
        cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
        cs.set_edgecolor('face')
        
        for xx in range(4):
            rt2 = np.argmax(cI1[:,xx])-1
            rt1 = np.argmax(cI1[:,xx])-1+70#np.argmax(I2[:,xx])-1
            print('')
            print(rt2)
            print(rt1)
            print('')
            
            lon_tr = (Sol_rad[1,rt2:rt1,xx]*180/np.pi)+90
            lat_tr = Sol_rad[2,rt2:rt1,xx]*180/np.pi - 90
            x1, y1 = map(lon_tr,lat_tr)
            X1 = np.hstack((x1, x1[-1]))
            mask = np.log(np.abs(np.diff(X1)))>20
            
            x1[mask] = np.nan
            y1[mask] = np.nan
            map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
            map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
            
            map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
            
            map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
            map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
           
        DT = DT  + iDT
    
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_delay.png', dpi=200)
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_RS2_thesis_delay.eps')
        print(folderP+'prj2_'+str(tsteps)+'_last_thesis_RS2.pdf')
        plt.show()
# =============================================================================
#       SECOND
# =============================================================================
        plt.figure()
        map2 = Basemap(projection='ortho', lat_0=Lttr, lon_0=Lnoo+30)
        
        # draw lat/lon grid lines every 30 degrees.
        map2.drawmeridians(np.arange(0, 360, 60),c='w')
        map2.drawparallels(np.arange(-90, 90, 30), color='w')
        
        lon = Theta*180/np.pi+90
        lat = (Phi*180/np.pi)-90
        Lon, Lat = np.meshgrid(lon , lat)
        # compute native map projection coordinates of lat/lon grid.
        x, y = map2(Lon, Lat)
        
        
        cs = map2.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
        cs.set_edgecolor('face')
        
        for xx in range(4):
            rt2 = np.argmax(I2[:,xx])-130
            #rt1 = np.argmax(cI1[:,xx])-1+40
            rt1 = np.argmax(I2[:,xx])-1
            print('')
            print(rt2)
            print(rt1)
            print('')
            
            lon_tr = (Sol_rad[1,rt2:rt1,xx]*180/np.pi)+90
            lat_tr = Sol_rad[2,rt2:rt1,xx]*180/np.pi - 90
            x1, y1 = map2(lon_tr,lat_tr)
            X1 = np.hstack((x1, x1[-1]))
            mask = np.log(np.abs(np.diff(X1)))>20
            
            x1[mask] = np.nan
            y1[mask] = np.nan
            map2.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
            map2.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
            
            map2.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
            
            map2.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
            map2.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
           
        DT = DT  + iDT
    
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_delay2.png', dpi=200)
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_RS2_thesis_delay2.eps')
        print(folderP+'prj2_'+str(tsteps)+'_last_thesis_RS2_2.pdf')
        plt.show()

#%%
def create_gif_delay2(Sol_rad, th0, I1, I2, O, E_manif_net, folderP, last=False, phi=-2, theta=-2, lat_tr=False, lon_tr2=False):
    DT0 = 2
    DT = DT0
    iDT = 2
    t1 = 0#137
  
    if np.sum(phi+2)==0:

        Phi = np.linspace(0,np.pi,70)#50
    else:
        
        Phi=phi
    if np.sum(theta+2)==0:
        Theta = np.linspace(-np.pi/2, 2*np.pi+th0, 70)#60
    else:
        Theta = theta
    
    
    
    cls = give_cls()
    if last==False:
        for tsteps in range(190):
            t1 = 50#137
            t2 = t1+DT
                
            xx = 0
            Lon_tr = np.zeros(4)
            Lat_tr = np.zeros(4)
            for xx in range(4):
                rt2 = np.min((t2,np.argmax(I2[:,0])))
                
                Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
                Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
              
                print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
            print('-')
            if tsteps==0:
                plon_tr = np.mean(Lon_tr)
                plat_tr = np.mean(Lat_tr)
            eta = 0.95
            if np.std(Lon_tr)<20:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
            else:
                Lon_tr1 = Lon_tr
                #break
                Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
                Lon_tr2 = Lon_tr
                Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
                
                if np.std(Lon_tr1)<np.std(Lon_tr2):
                
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
                else:
                    lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
            plon_tr = lon_tr
            print(lon_tr)
            
            if lat_tr==False:
                lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
                plat_tr = lat_tr        
                print(lat_tr)
                
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr)
            else:
                map = Basemap(projection='ortho', lat_0=lat_tr, lon_0=lon_tr2)
            # draw lat/lon grid lines every 30 degrees.
            map.drawmeridians(np.arange(0, 360, 60),c='w')
            map.drawparallels(np.arange(-90, 90, 30), color='w')
            
            lon = Theta*180/np.pi+90
            lat = (Phi*180/np.pi)-90
            Lon, Lat = np.meshgrid(lon , lat)
            # compute native map projection coordinates of lat/lon grid.
            x, y = map(Lon, Lat)
            
        
            cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
            cs.set_edgecolor('face')
            
            for xx in range(4):
                val = np.argmax(I1[:,xx])
                cI1 = I1
                cI1[val,xx] = -1
                
                rt2 = np.argmax(cI1[:,xx])-1#np.min((t2, np.argmax(I2[:,0])))
                
                lon_tr = (Sol_rad[1,t1:rt2,xx]*180/np.pi)+90
                lat_tr = Sol_rad[2,t1:rt2,xx]*180/np.pi - 90
                print(lat_tr)
                x1, y1 = map(lon_tr,lat_tr)
                X1 = np.hstack((x1, x1[-1]))
                mask = np.log(np.abs(np.diff(X1)))>20
                
                x1[mask] = np.nan
                y1[mask] = np.nan
                map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
                map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
                
                map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
                
                map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
                map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
               
                
        
            DT = DT  + iDT
            plt.title('t='+str(DT*10/1000)+'s')
        
            plt.savefig(folderP+'prj2_'+str(tsteps)+'.png', dpi=200)
            plt.show()
    else:
        tsteps=190
        t1 = 50#137
        t2 = 100#t1+tsteps*DT
            
        xx = 0
        Lon_tr = np.zeros(4)
        Lat_tr = np.zeros(4)
        for xx in range(4):
            val = np.argmax(I1[:,xx])
            cI1 = I1
            cI1[val,xx] = -1
                
            rt2 = np.argmax(cI1[:,xx])-10#np.min((t2, np.argmax(I2[:,0])))rt2 = np.argmax(I2[:,0])#np.min((t2,np.argmax(I2[:,0])))
            
            Lon_tr[xx] = ((Sol_rad[1,rt2,xx]*180/np.pi)+90)
            Lat_tr[xx] = (Sol_rad[2,rt2,xx]*180/np.pi - 90)
            print(rt2)
            #print( ((Sol_rad[1,rt2,xx]*180/np.pi)+90))
        print('-')
        plon_tr = np.mean(Lon_tr)
        plat_tr = np.mean(Lat_tr)
        eta = 0.95
        if np.std(Lon_tr)<20:
            lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr)
        else:
            Lon_tr1 = Lon_tr
            #break
            Lon_tr1[Lon_tr>180] = Lon_tr1[Lon_tr>180]-360
            Lon_tr2 = Lon_tr
            Lon_tr2[Lon_tr<180] = Lon_tr2[Lon_tr<180]+360
            
            if np.std(Lon_tr1)<np.std(Lon_tr2):
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr1)
            else:
                lon_tr = (1-eta)*plon_tr+eta*np.nanmean(Lon_tr2)
        plon_tr = lon_tr
        print(lon_tr)
        
        
        lat_tr = (1-eta)*plat_tr+eta*np.mean(Lat_tr)
        plat_tr = lat_tr        
        print(lat_tr)

        map = Basemap(projection='ortho', lat_0=69, lon_0=37)#lat_tr, lon_0=lon_tr+30)
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0, 360, 60),c='w')
        map.drawparallels(np.arange(-90, 90, 30), color='w')
        
        lon = Theta*180/np.pi+90
        lat = (Phi*180/np.pi)-90
        Lon, Lat = np.meshgrid(lon , lat)
        # compute native map projection coordinates of lat/lon grid.
        x, y = map(Lon, Lat)
        
    
        cs = map.pcolor(x, y, np.log(E_manif_net).T,  alpha=0.8, linewidth=0)
        cs.set_edgecolor('face')
        
        for xx in range(4):
            rt2 = np.argmax(I2[:,xx])-40
            rt1 = np.argmax(I2[:,xx])-1
            print('')
            print(rt2)
            print(rt1)
            print('')
            
            lon_tr = (Sol_rad[1,rt2:rt1,xx]*180/np.pi)+90
            lat_tr = Sol_rad[2,rt2:rt1,xx]*180/np.pi - 90
            
            
            x1, y1 = map(lon_tr,lat_tr)
            X1 = np.hstack((x1, x1[-1]))
            mask = np.log(np.abs(np.diff(X1)))>20
            
            x1[mask] = np.nan
            y1[mask] = np.nan
            map.plot(x1,y1, lw = 3, c = cls[xx,:], zorder=2)
            map.plot(x1,y1, lw = 3.8, c = 'w', zorder=1)
            
            map.scatter(x1[0],y1[0], s = 100, c = cls[xx,:], edgecolor='w', zorder=5)
            
            map.scatter(x1[::20],y1[::20], s = 20, c = 'k', edgecolor='w', zorder=4)
            map.scatter(x1[-1],y1[-1], s = 150, c = cls[xx,:], edgecolor='w', marker='^', zorder=6)
           
        DT = DT  + iDT
    
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_delay2.png', dpi=200)
        plt.savefig(folderP+'prj2_'+str(tsteps)+'_last_RS2_thesis_delay2.eps')
        print(folderP+'prj2_'+str(tsteps)+'_last_thesis_RS2.pdf')
        plt.show()

#%%
def give_cls():
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
    cls = lr.set_plot2()
    
    cls[1,:] = cls[2,:]
    cls[2,:] = cls[4,:]
    cls[3,:] = cls[5,:]
    return(cls)

#%%
def inter_3dtraj(Sol, E_manif_net, R_manif_net, exc, I2, O, th0, phi=-2, theta=-2):
    from matplotlib import cm
    t1 = np.argmax(I2[:,0])-1

    Theta = np.linspace(-np.pi/2, 2*np.pi+th0, 70)#60

    if np.sum(phi+2)==0:

        Phi = np.linspace(0,np.pi,70)#50
        phi = Phi#np.linspace(0, np.pi, 100)
    else:
        
        Phi=phi
    if np.sum(theta+2)==0:
        theta = Theta#np.linspace(0, 2*np.pi, 100)
    else:
        Theta = theta
    phi, theta = np.meshgrid(phi, theta)
    
    # The Cartesian coordinates of the unit sphere
    
    x = np.zeros_like(phi)
    y = np.zeros_like(phi)
    z = np.zeros_like(phi)
    for ip, p in enumerate(Phi):
        for it, t in enumerate(Theta):
            x[it, ip] = R_manif_net[it, ip]*np.sin(p)*np.cos(t)
            y[it, ip] = R_manif_net[it, ip]*np.sin(p)*np.sin(t)
            z[it, ip] = R_manif_net[it, ip]*np.cos(p)
    
    cls = give_cls()
    
    
    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = np.log(E_manif_net)
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)
    
    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d', azim= -104, elev=58)
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.viridis(fcolors),antialiased=False)
    for xx in range(4):
        rt2 =  np.argmax(O[:,xx])
        Xx = Sol[0, t1+1:rt2,xx]
        Yy = Sol[1, t1+1:rt2,xx]
        Zz = Sol[2, t1+1:rt2,xx]
        
        pert = np.array((Xx, Yy, Zz))
        per = pert/np.sqrt(np.sum(pert**2,0))*exc[t1+1:rt2]
        ax.plot(Xx,Yy,Zz, '--', lw = 2, c = cls[xx,:], zorder=2)
        ax.plot(Xx+per[0],Yy+per[1],Zz+per[2], lw = 3, c = cls[xx,:], zorder=2)
        
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.viridis(fcolors), alpha=1.)
    # Turn off the axis planes
    ax.set_axis_off()
    plt.show()

def plot_1Dapprox(l_atr, azim=60, elev=-20):
    mu = np.mean(l_atr,1)  
    dat = l_atr.T-mu
    dat = dat.T
    
    v, w = np.linalg.eig(np.dot(dat, dat.T))
    
    tms = 0.5
    t = np.linspace(-tms, tms)
    line = np.zeros((3, len(t)))
    
    for it, tt in enumerate(t):
        line[:,it] = tt*w[:,0]+mu
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev)
    for xx in range(4):
        ax.scatter(l_atr[0,xx], l_atr[1,xx], l_atr[2,xx], s=100)
    W  = np.expand_dims(w[:,0], 1)
    
    proj_lin = np.dot(W.T, dat)
    ax.plot(line[0,:], line[1,:], line[2,:], lw=2, c = [0.7, 0.7, 0.7])
    
    cls = give_cls()
    for xx in range(4):
        ax.scatter(l_atr[0,xx], l_atr[1,xx], l_atr[2,xx], s=70,  c = cls[xx,:])
        ax.scatter(proj_lin[0,xx]*W[0]+mu[0], proj_lin[0,xx]*W[1]+mu[1], proj_lin[0,xx]*W[2]+mu[2], s=20, c = cls[xx,:])
        ax.plot([l_atr[0,xx], proj_lin[0,xx]*W[0]+mu[0]], [l_atr[1,xx], proj_lin[0,xx]*W[1]+mu[1] ] , [l_atr[2,xx], proj_lin[0,xx]*W[2]+mu[2] ] , '-k' )
    var_e = v[0]/np.sum(v)
    return(mu, w[:,0], var_e, proj_lin, fig, ax)

#%%
def get_avgtraj_pert(net_low, M, ic, w_l, mu_l, t_B, itt, Nt, ints, dt, dela, R_on, thrs = 0.4, Rot = 0):# ints, trials_test, dt, Nt, R_on, dela, phi, theta, M, N, align_set = True, show_set = True):
    '''
    t_B: time of perturbation
    itt: time of stimulus
    
    '''
    

    #exc = np.zeros((1, Nt, len(ints)))
    #factor = 1
    
    time_ = np.arange(Nt)
    time_mask = time_[time_>t_B]
    rNt= len(time_mask)
    Sol_rad  = np.zeros((3,  rNt, len(ic)))
    Sol      = np.zeros_like(Sol_rad)
    
    if np.size(Rot) ==1:
        Rot = np.eye(np.shape(M)[1])

    out_     = np.zeros((rNt, len(ic)))
    trials_test = 30
    thrs_cs  = np.zeros((trials_test, len(ic)))
    dtype = torch.FloatTensor  
    
    time = time_[time_mask]
    time = time-time[0]
    for iC, C in enumerate(ic):

        input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = lr.create_inp_out_rSSg_2in(trials_test, Nt, ints//dt, 
                                                                            R_on+dela, 1, just=0, perc=0., perc1=0., fact=1, align_set = True, delayF = dela)
        
        rh0 = np.dot(M, C*w_l+mu_l)
        
        net_low.h0 = torch.nn.Parameter(torch.from_numpy(rh0).type(dtype))

        input_tr[:,:,0] = 0.*input_tr[:,:,0]
        outp, traj = net_low.forward(input_tr[:,time_mask,:], return_dynamics=True)
        outp = outp.detach().numpy()
        traj = traj.detach().numpy()
        
        
        
        out_[:, iC] = np.mean(outp[:,:,0],0)
        I2 = input_tr[0,time_mask,1].detach().numpy()
        thrs_cs[:,iC] = (time[np.argmin(np.abs((outp[:,:,0]-thrs)),1)]-time[np.argmax(I2)])/(0.5+thrs)
        
        proj_m =  np.dot(M.T,np.mean(traj[:,:,:],0).T).T/np.diag(np.dot(M.T,M))#hidden_size
        proj_m = proj_m.T
        
        Sol[:,:,iC] = np.dot(Rot,proj_m[:,0:rNt])
        for it in range(rNt):
            dot = np.dot(Rot,proj_m[:,it])
            sol = cart2sph(dot)
            Sol_rad[:,it, iC] = sol
        
    I1 =input_tr[:,time_mask,0].detach().numpy()
    I2 = input_tr[:,time_mask,1].detach().numpy()

    return(out_, time, Sol, Sol_rad, I1, I2, thrs_cs)
    
#%% 
def get_avgtraj_perf(net_low, ints, trials_test, dt, Nt, R_on, dela, phi, theta, M, N, thrs= 0.4, align_set = True, show_set = True, Rot=0):
    
    I1 = np.zeros((Nt, len(ints)))
    I2 = np.zeros((Nt, len(ints)))
    O = np.zeros((Nt, len(ints)))
    
    Sol_rad  = np.zeros((3,  Nt, len(ints)))
    Sol      = np.zeros_like(Sol_rad)
    exc = np.zeros((1, Nt, len(ints)))
    factor = 1
    thrs_cs  = np.zeros((trials_test, len(ints)))
    
    if np.size(Rot) ==1:
        Rot = np.eye(np.shape(M)[1])
    
    time = np.arange(Nt)
    time = time*dt
    for xx in range(len(ints)):
        if align_set:
            input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = lr.create_inp_out_rSSg_2in(trials_test, Nt, ints//dt, 
                                                                            R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela)
        else:
            input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = lr.create_inp_out_rSSg_2in(trials_test, Nt, ints//dt, 
                                                                            R_on, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela)
        if show_set ==False:
            input_tr[:,:,1] = 0.*input_tr[:,:,1]
        outp, traj = net_low.forward(input_tr, return_dynamics=True)
        outp = outp.detach().numpy()
        traj = traj.detach().numpy()
        
         
        proj_m =  np.dot(M.T,np.mean(traj[:,:,:],0).T).T/np.diag(np.dot(M.T,M))#hidden_size
        exc = np.std(np.dot(proj_m, M.T)-np.mean(traj[:,:,:],0),1)
        proj_m = proj_m.T
        
        
        
        I1[:,xx] = input_tr[0,:,0].detach().numpy()
        I2[:,xx] = input_tr[0,:,1].detach().numpy()
        Outp = np.copy(outp[:,:,0])
        max_idcs = np.argmax(outp[:,:,0],1)
        
        for rr in range(np.shape(Outp)[0]):
            Outp[rr,max_idcs[rr]:] = Outp[rr,max_idcs[rr]]
        thrs_cs[:,xx] = (time[np.argmin(np.abs((Outp-thrs)),1)]-time[np.argmax(I2[:,xx])])/(0.5+thrs)
        O[:,xx] = output_tr[0,:,0].detach().numpy()
        Sol[:,:,xx] = np.dot(Rot,proj_m[:,0:Nt])
        for it in range(Nt):
            dot = np.dot(Rot,proj_m[:,it])
            sol = cart2sph(dot)
            Sol_rad[:,it, xx] = sol
    return(Sol, Sol_rad, exc, I1, I2, O, proj_m, thrs_cs, outp[:,:,0])