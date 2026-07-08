import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def gen_intervals(ts_final,N_steps, ts_min=0.4):
    '''
    Generates time intervals for training schedule, from faster to slower intervals

    Parameters
    ----------
    ts_final : np.array 2D
        Final time intervals to be trained on
    N_steps : int
        Number of intermediate training steps
    ts_min : double, optional
        Fastest interval, as a fraction of final interval. The default is 0.4.

    Returns
    -------
    Tss : 2D array, with the time intervals for intervals at each time step

    '''
    Tss = np.zeros((len(ts_final), N_steps))
    stp_pr = np.linspace(ts_min, 1.0,N_steps)
    for i in range(N_steps):
        Tss[:,i] = stp_pr[i]*ts_final
    return(Tss)

def set_plot(ll = 7):
    '''
    Set plotting parameters. Returns colors for plots

    Parameters
    ----------
    ll : int, optional
        Number of colors. 5 or 7. The default is 7.

    Returns
    -------
    clS : colors

    '''
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
    
    clS = np.zeros((ll,3))
    
    cl11 = np.array((102, 153, 255))/255.
    cl12 = np.array((53, 153, 53))/255.
    
    cl21 = np.array((255, 204, 51))/255.
    cl22 = np.array((204, 0, 0))/255.
    
    if ll==7:
        clS[0,:] = 0.4*np.ones((3,))
        
        clS[1,:] = cl11
        clS[2,:] = 0.5*cl11+0.5*cl12
        clS[3,:] = cl12
        
        clS[4,:] = cl21
        clS[5,:] = 0.5*cl21+0.5*cl22
        clS[6,:] = cl22
        
        clS = clS[1:]
        clS = clS[::-1]
        
        c2 = [67/256, 90/256, 162/256]
        c1 = [220/256, 70/256, 51/256]
        clS[0,:]=c1
        clS[5,:]=c2
    elif ll == 5:
        clS[0,:] = 0.4*np.ones((3,))    
        
        clS[2,:] = cl12
        
        clS[3,:] = cl21
        
        clS[4,:] = cl22    
    return(clS)

def fromSigma_to_BigSigma(sigma_mn, sigma_m):
    '''
    Generates the 2Rx2R covariance matrix of the loadings based on the 
    RxR covariance matrix of the loadings

    Parameters
    ----------
    sigma_mn : TYPE
        DESCRIPTION.
    sigma_m : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    dims = np.shape(sigma_m)[0]
    bigSigma = np.zeros((dims+dims,dims+dims))
    
    for d in range(dims):
        bigSigma[d,d] = sigma_m[d]#m_i^2
        bigSigma[d+dims, d+dims] = sigma_m[d]
    
    bigSigma[dims:,0:dims] = sigma_mn
    bigSigma[0:dims,dims:] = sigma_mn
    
    it_max = 1000
    ite = 0
    while np.min(np.linalg.eigvals(bigSigma))<1e-7 and ite<it_max:
        bigSigma[dims:, dims:] = 1.02*bigSigma[dims:, dims:]
        ite +=1
    bigSigma[dims:, dims:] = 1.01*bigSigma[dims:, dims:]
    return(bigSigma)

def initialize_sphere(hidden_size, s_mn=2, Delta=0.01, do=0.1, sm = 1.,dims=3, Delta2=np.nan, run_mat = 100):
    sigma_mn = s_mn*np.eye(dims)
    sigma_m = sm*np.ones((dims))
    
    sigma_mn[0,0] = s_mn
    if dims>1:
        sigma_mn[1,1] -= Delta
        
    if dims>2:
        if np.isnan(Delta2):
            Delta2=Delta
            sigma_mn[2,2] = s_mn-Delta2
    
        sigma_mn[2,1] = 0.
        sigma_mn[0,2] = do
    
    bigSigma = fromSigma_to_BigSigma(sigma_mn, sigma_m)
    mean = np.zeros((dims+dims))
    
    err_min = 1000.0
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    for tr in range(run_mat):
        X = np.random.multivariate_normal(mean, bigSigma, size = hidden_size)
        err_tr = np.std(np.cov(X.T)-bigSigma)
        if err_tr<err_min:
            err_min = err_tr
            X_save = X
        
    X = X_save
    M = X[:,0:dims]
    N = X[:,dims:]
    J = np.dot(M, N.T)/hidden_size
    
    return(sigma_mn, sigma_m, J, M, N, err_min, bigSigma, np.cov(X.T))

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

def give_manif_Inp(theta, phi, rs, En, UXn):
    np.seterr(divide='ignore', invalid='ignore')
    
    E_manif_net = np.zeros((len(theta), len(phi)))
    U_manif_net = np.zeros_like(E_manif_net)
    V_manif_net = np.zeros_like(E_manif_net)
    R_manif_net = np.zeros((len(theta), len(phi)))
    
    for it, th in enumerate(theta):
        for ip , p in enumerate(phi):
            # ix = np.argmin(E[:,it, ip])
            # R = rs[ix]
            # E_manif[it, ip] = E[ix,it,ip] 
            # R_manif[it, ip] = R
            
            # vec = UX[ix,it,ip,:]
            
            # X = rs[ix]*np.sin(p)*np.cos(th)
            # Y = rs[ix]*np.sin(p)*np.sin(th)
            # Z = rs[ix]*np.cos(p)
            # vec_r = np.array((X, Y, Z))
            # vec_r = vec_r/np.sqrt(np.sum(vec_r**2))
            # vec_theta = np.array((-Y, X, 0))
            # #if np.sqrt(np.sum(vec_theta**2))>0:
            # vec_theta = vec_theta/np.sqrt(np.sum(vec_theta**2))
            # vec_phi   = np.array((X*Z, Y*Z, -X**2-Y**2))
            # #if np.sqrt(np.sum(vec_phi**2))    >0:
            # vec_phi = vec_phi/np.sqrt(np.sum(vec_phi**2))        
            
            # U_manif[it, ip] =np.sum(vec*vec_theta)  
            # V_manif[it, ip] = np.sum(vec*vec_phi)
            
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
    return( E_manif_net, R_manif_net, U_manif_net, V_manif_net)

def give_fieldsMF2D(rs, theta, phi, sigma_mn, sigma_m, sigma_nI = None, sigma_I = None, verbose = False, vm = -5):
    E = np.zeros((len(rs), len(phi)))
    UX = np.zeros((len(rs), len(phi), 3))
    th = theta
    for ir, r in enumerate(rs):
        if verbose:
            if 100*(ir/len(rs))==100*ir//len(rs):
                print('Accomplished: '+str(100*ir//len(rs))+' %')
        
        for ip , p in enumerate(phi):
            
            k0 = r*np.array((np.sin(p)*np.cos(th), np.sin(p)*np.sin(th), np.cos(p) ))
           
            if sigma_nI is None:
                UX[ir, ip,:], E[ir, ip] =  def_field(k0, sigma_mn, sigma_m)
            else:
                UX[ir, ip,:], E[ir, ip] =  def_fieldInp(k0, sigma_mn, sigma_m, sigma_nI, sigma_I)
    return(UX, E)

def give_fieldsMF2Dhor(rs, theta, sigma_mn, sigma_m, sigma_nI = None, sigma_I = None, verbose = False, vm = -5):
    E = np.zeros((len(rs), len(theta)))
    UX = np.zeros((len(rs), len(theta), 3))
    
    for ir, r in enumerate(rs):
        if verbose:
            if 100*(ir/len(rs))==100*ir//len(rs):
                print('Accomplished: '+str(100*ir//len(rs))+' %')
        
        for it , th in enumerate(theta):

            k0 = r*np.array((np.cos(th), np.sin(th), 0. ))
            if sigma_nI is None:
                UX[ir, it,:], E[ir, it] =  def_field(k0, sigma_mn, sigma_m)
            else:
                UX[ir, it,:], E[ir, it] =  def_fieldInp(k0, sigma_mn, sigma_m, sigma_nI, sigma_I)
    return(UX, E)

def give_fields(rs, theta, sigma_mn, sigma_m, sigma_nI = None, sigma_I = None, verbose = False, vm = -5):
    E = np.zeros((len(rs), len(theta)))
    UX = np.zeros((len(rs), len(theta), 3))
    
    for ir, r in enumerate(rs):
        if verbose:
            if 100*(ir/len(rs))==100*ir//len(rs):
                print('Accomplished: '+str(100*ir//len(rs))+' %')
        
        for it , th in enumerate(theta):

            k0 = r*np.array((np.cos(th), np.sin(th), 0. ))
            if sigma_nI is None:
                UX[ir, it,:], E[ir, it] =  def_field(k0, sigma_mn, sigma_m)
            else:
                UX[ir, it,:], E[ir, it] =  def_fieldInp(k0, sigma_mn, sigma_m, sigma_nI, sigma_I)
    return(UX, E)

def give_manif2D(theta, phi, rs, E, UX):
    np.seterr(divide='ignore', invalid='ignore')
    E_manif = np.zeros( len(phi))
    U_manif = np.zeros_like(E_manif)
    V_manif = np.zeros_like(E_manif)
    R_manif = np.zeros_like(E_manif)
    
   
    th = theta
    for ip , p in enumerate(phi):
        ix = np.argmin(E[:, ip])
        R = rs[ix]
        E_manif[ ip] = E[ix,ip] 
        R_manif[ ip] = R
        
        vec = UX[ix,ip,:]
        
        X = rs[ix]*np.sin(p)*np.cos(th)
        Y = rs[ix]*np.sin(p)*np.sin(th)
        Z = rs[ix]*np.cos(p)
        vec_r = np.array((X, Y, Z))
        vec_r = vec_r/np.sqrt(np.sum(vec_r**2))
        vec_theta = np.array((-Y, X, 0))
        vec_theta = vec_theta/np.sqrt(np.sum(vec_theta**2))
        vec_phi   = np.array((X*Z, Y*Z, -X**2-Y**2))
        #if np.sqrt(np.sum(vec_phi**2))    >0:
        vec_phi = vec_phi/np.sqrt(np.sum(vec_phi**2))        
        
        U_manif[ ip] =np.sum(vec*vec_theta)  
        V_manif[ ip] = np.sum(vec*vec_phi)
        
        
    return(E_manif, R_manif, U_manif, V_manif)

def give_manif2Dhor(theta, rs, E, UX):
    np.seterr(divide='ignore', invalid='ignore')
    E_manif = np.zeros( len(theta))
    U_manif = np.zeros_like(E_manif)
    V_manif = np.zeros_like(E_manif)
    R_manif = np.zeros_like(E_manif)
    
   

    for it , th in enumerate(theta):
        ix = np.argmin(E[:, it])
        R = rs[ix]
        E_manif[ it] = E[ix,it] 
        R_manif[ it] = R
        
        vec = UX[ix,it,:]
        
        X = rs[ix]*np.cos(th)
        Y = rs[ix]*np.sin(th)
        Z = rs[ix]*0
        vec_r = np.array((X, Y, Z))
        vec_r = vec_r/np.sqrt(np.sum(vec_r**2))
        vec_theta = np.array((-Y, X, 0))
        vec_theta = vec_theta/np.sqrt(np.sum(vec_theta**2))
        vec_phi   = np.array((X*Z, Y*Z, -X**2-Y**2))
        #if np.sqrt(np.sum(vec_phi**2))    >0:
        vec_phi = vec_phi/np.sqrt(np.sum(vec_phi**2))        
        
        U_manif[ it] =np.sum(vec*vec_theta)  
        V_manif[ it] = np.sum(vec*vec_phi)
        
        
    return(E_manif, R_manif, U_manif, V_manif)

def give_manifMF(theta, phi, rs, E, UX):
    np.seterr(divide='ignore', invalid='ignore')
    E_manif = np.zeros((len(theta), len(phi)))
    U_manif = np.zeros_like(E_manif)
    V_manif = np.zeros_like(E_manif)
    
    R_manif = np.zeros((len(theta), len(phi)))
    
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
            vec_theta = vec_theta/np.sqrt(np.sum(vec_theta**2))
            vec_phi   = np.array((X*Z, Y*Z, -X**2-Y**2))
            vec_phi = vec_phi/np.sqrt(np.sum(vec_phi**2))        
            
            U_manif[it, ip] =np.sum(vec*vec_theta)  
            V_manif[it, ip] = np.sum(vec*vec_phi)
                
            R = rs[ix]
            
    return(E_manif, R_manif, U_manif, V_manif)

def def_field(k0, sigma_mn, sigma_m):
    delta = np.sum((k0*sigma_m)**2)
    prim = Prime(0,delta)
    sol = -k0 + prim*np.dot(sigma_mn, k0)
    E = np.sum(sol**2)
    return(sol, E)

def def_fieldInp(k0, sigma_mn, sigma_m, sigma_nI, sigma_I):
    delta = np.sum((k0*sigma_m)**2)+sigma_I**2
    prim = Prime(0,delta)
    sol = -k0 + prim*(np.dot(sigma_mn, k0)+sigma_nI)
    E = np.sum(sol**2)
    return(sol, E)

def def_field_net(k0, M, N):    
    sol = -k0 
    dims = len(k0)
    for d in range(dims):
        sol[d] +=   np.mean(N[:,d]*np.tanh(np.dot(M,k0)))

    E = np.sqrt(np.sum(sol**2))
    return(sol, E)

def give_fields_Inp(rs, theta, phi, M, N, I, verbose = False, vm = -5):
    
    En = np.zeros((len(rs), len(theta), len(phi)))
    UXn = np.zeros((len(rs), len(theta), len(phi), 3))
    
    for ir, r in enumerate(rs):
        if verbose:
            if 100*(ir/len(rs))==100*ir//len(rs):
                print('Accomplished: '+str(100*ir//len(rs))+' %')
        for it, th in enumerate(theta):
            for ip , p in enumerate(phi):
                k0 = r*np.array((np.sin(p)*np.cos(th), np.sin(p)*np.sin(th), np.cos(p) ))
                UXn[ir, it, ip,:], En[ir, it, ip] =  def_field_net_Inp(k0, M, N,I)
    return( UXn, En)

def give_fields(rs, theta, phi, M, N, sigma_mn, sigma_m, verbose = False, vm = -5):
    E = np.zeros((len(rs), len(theta), len(phi)))
    UX = np.zeros((len(rs), len(theta), len(phi), 3))
    
    En = np.zeros((len(rs), len(theta), len(phi)))
    UXn = np.zeros((len(rs), len(theta), len(phi), 3))
    
    for ir, r in enumerate(rs):
        if verbose:
            if 100*(ir/len(rs))==100*ir//len(rs):
                print('Accomplished: '+str(100*ir//len(rs))+' %')
        for it, th in enumerate(theta):
            for ip , p in enumerate(phi):
                k0 = r*np.array((np.sin(p)*np.cos(th), np.sin(p)*np.sin(th), np.cos(p) ))
                UX[ir, it, ip,:], E[ir, it, ip] =  def_field(k0, sigma_mn, sigma_m)
                UXn[ir, it, ip,:], En[ir, it, ip] =  def_field_net(k0, M, N)
    return(UX, E, UXn, En)


def give_fieldsMF(rs, theta, phi, sigma_mn, sigma_m, sigma_nI = None, sigma_I = None, verbose = False, vm = -5):
    E = np.zeros((len(rs), len(theta), len(phi)))
    UX = np.zeros((len(rs), len(theta), len(phi), 3))
    

    for ir, r in enumerate(rs):
        if verbose:
            if 100*(ir/len(rs))==100*ir//len(rs):
                print('Accomplished: '+str(100*ir//len(rs))+' %')
        for it, th in enumerate(theta):
            for ip , p in enumerate(phi):
                k0 = r*np.array((np.sin(p)*np.cos(th), np.sin(p)*np.sin(th), np.cos(p) ))
                if sigma_nI is None:
                    UX[ir, it, ip,:], E[ir, it, ip] =  def_field(k0, sigma_mn, sigma_m)
                else:
                    UX[ir, it, ip,:], E[ir, it, ip] =  def_fieldInp(k0, sigma_mn, sigma_m, sigma_nI, sigma_I)
    return(UX, E)

def give_fieldsMF_cart(xs, ys, zs, sigma_mn, sigma_m, sigma_nI = None, sigma_I = None, verbose = False, vm = -5):
    E = np.zeros((len(xs), len(ys), len(zs)))
    UX = np.zeros((len(xs), len(ys), len(zs), 3))
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz , z in enumerate(zs):
                k0 = np.array((x, y , z))
                if sigma_nI is None:
                    UX[ix, iy, iz,:], E[ix, iy, iz] =  def_field(k0, sigma_mn, sigma_m)
                else:
                    UX[ix, iy, iz,:], E[ix, iy, iz] =  def_fieldInp(k0, sigma_mn, sigma_m, sigma_nI, sigma_I)
    return(UX, E)

def run_randtraj_mf(sigma_mn, sigma_m, T = 80, dt = 0.2, trajs=20):
    dims = len(sigma_m)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    time = np.arange(0, T, dt)
    
    for tr in range(trajs):
        ks = np.zeros((dims, len(time)))
        k0 = 0.1*np.random.randn(dims)
        ks[:,0]= k0
        for it, ti in enumerate(time[:-1]):
            ks[:,it+1] = ks[:,it] + dt*(def_field(ks[:,it], sigma_mn, sigma_m))[0]
        ax.plot(ks[0,:], ks[1,:], ks[2,:], c=0.6*np.ones(3))    
        ax.scatter(ks[0,0], ks[1,0], ks[2,0], s=5, c='k') 
        ax.scatter(ks[0,-1], ks[1,-1], ks[2,-1], s=20, edgecolor='w', facecolor='k') 
    plt.xlabel(r'$\kappa_1$')
    plt.ylabel(r'$\kappa_2$')
    ax.set_zlabel(r'$\kappa_3$')
    
    return(fig, ax)

def run_randtraj_mf_all(sigma_mn, sigma_m, T = 80, dt = 0.2, trajs=20):
    dims = len(sigma_m)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    time = np.arange(0, T, dt)
    Trajs = np.zeros((3, len(time), trajs))
    Times = np.zeros(trajs)
    iTimes = np.zeros(trajs)
    for tr in range(trajs):
        ks = np.zeros((dims, len(time)))
        k0 = 0.1*np.random.randn(dims)
        ks[:,0]= k0
        stop = False
        for it, ti in enumerate(time[:-1]):
            if stop==False:
                sp = (def_field(ks[:,it], sigma_mn, sigma_m))[0]
                if np.sqrt(np.sum(sp**2))<1e-3 and ti>4. and np.sqrt(np.sum(ks[:,it]**2))>0.1:
                    fp = ks[:,it] + dt*sp
                    ks[0,it:]=fp[0]
                    ks[1,it:]=fp[1]
                    ks[2,it:]=fp[2]
                    Times[tr]  = ti 
                    iTimes[tr] = it
                    stop=True
                else:
                    ks[:,it+1] = ks[:,it] + dt*sp
        if stop==False:
            Times[tr] = ti
            iTimes[tr] = it
        Trajs[:,:,tr] = ks 
        ax.plot(ks[0,:], ks[1,:], ks[2,:], c=0.6*np.ones(3))    
        ax.scatter(ks[0,0], ks[1,0], ks[2,0], s=5, c='k') 
        ax.scatter(ks[0,-1], ks[1,-1], ks[2,-1], s=20, edgecolor='w', facecolor='k') 
    plt.xlabel(r'$\kappa_1$')
    plt.ylabel(r'$\kappa_2$')
    ax.set_zlabel(r'$\kappa_3$')
    
    return(fig, ax, Trajs, Times, iTimes, time)

def run_randtraj_fs(M, N, T = 80, dt = 0.2, trajs=20):
    dims = np.shape(M)[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    time = np.arange(0, T, dt)
    
    for tr in range(trajs):
        ks = np.zeros((dims, len(time)))
        k0 = 0.1*np.random.randn(dims)
        ks[:,0]= k0
        for it, ti in enumerate(time[:-1]):
            ks[:,it+1] = ks[:,it] + dt*(def_field_net(ks[:,it], M, N))[0]
        ax.plot(ks[0,:], ks[1,:], ks[2,:], c=0.6*np.ones(3))    
        ax.scatter(ks[0,0], ks[1,0], ks[2,0], s=5, c='k') 
        ax.scatter(ks[0,-1], ks[1,-1], ks[2,-1], s=20, edgecolor='w', facecolor='k') 
    plt.xlabel(r'$\kappa_1$')
    plt.ylabel(r'$\kappa_2$')
    ax.set_zlabel(r'$\kappa_3$')
    
    return(fig, ax)

def run_randtraj_fs_all(M, N, T = 80, dt = 0.2, trajs=20, lw=2):
    dims = np.shape(M)[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', azim=50, elev=15)
    time = np.arange(0, T, dt)
    Trajs = np.zeros((3, len(time), trajs))
    Times = np.zeros(trajs)
    iTimes = np.zeros(trajs)
    for tr in range(trajs):
        ks = np.zeros((dims, len(time)))
        k0 = 0.1*np.random.randn(dims)
        ks[:,0]= k0
        stop=False
        for it, ti in enumerate(time[:-1]):
            ks[:,it+1] = ks[:,it] + dt*(def_field_net(ks[:,it], M, N))[0]
            if stop==False:
                sp = (def_field_net(ks[:,it], M, N))[0]
                if np.sqrt(np.sum(sp**2))<1e-3 and ti>4.:
                    fp = ks[:,it] + dt*sp
                    ks[0,it:]=fp[0]
                    ks[1,it:]=fp[1]
                    ks[2,it:]=fp[2]
                    Times[tr]  = ti 
                    iTimes[tr] = it
                    stop=True
                else:
                    ks[:,it+1] = ks[:,it] + dt*sp
        
        Trajs[:,:,tr] = ks 
        ax.plot(ks[0,:], ks[1,:], ks[2,:], c=0.4*np.ones(3), lw=lw)    
        ax.scatter(ks[0,0], ks[1,0], ks[2,0], s=30, marker='^',edgecolor='k', color=(0.6,0.6,0.6), zorder=4)
        ax.scatter(ks[0,-1], ks[1,-1], ks[2,-1], s=40, edgecolor='w', facecolor='k') 
    plt.xlabel(r'$\kappa_1$')
    plt.ylabel(r'$\kappa_2$')
    ax.set_zlabel(r'$\kappa_3$')
    
    return(fig, ax, Trajs, Times, iTimes, time)

def def_field_net_Inp(k0, M, N, I):    
    sol = -k0 
    dims = len(k0)
    for d in range(dims):
        sol[d] +=   np.mean(N[:,d]*np.tanh(np.dot(M,k0) + I)) + np.mean(M[:,d]*I)

    E = np.sqrt(np.sum(sol**2))
    return(sol, E)

def run_randtraj_fs_all_Inp(M, N, I, T = 80, dt = 0.2, trajs=20):
    dims = np.shape(M)[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', azim=-16, elev=17)
    time = np.arange(0, T, dt)
    Trajs = np.zeros((3, len(time), trajs))
    Times = np.zeros(trajs)
    iTimes = np.zeros(trajs)
    for tr in range(trajs):
        ks = np.zeros((dims, len(time)))
        k_Inp, QI = def_field_net_Inp(np.zeros(3), M, N, I)
        k0 = 0.1*np.random.randn(dims)+k_Inp
        ks[:,0]= k0
        stop=False
        for it, ti in enumerate(time[:-1]):
            ks[:,it+1] = ks[:,it] + dt*(def_field_net_Inp(ks[:,it], M, N, I))[0]
            if stop==False:
                sp = (def_field_net_Inp(ks[:,it], M, N, I))[0]
                if np.sqrt(np.sum(sp**2))<1e-3 and ti>4.:
                    fp = ks[:,it] + dt*sp
                    ks[0,it:]=fp[0]
                    ks[1,it:]=fp[1]
                    ks[2,it:]=fp[2]
                    Times[tr]  = ti 
                    iTimes[tr] = it
                    stop=True
                else:
                    ks[:,it+1] = ks[:,it] + dt*sp
        
        Trajs[:,:,tr] = ks 
        ax.plot(ks[0,:], ks[1,:], ks[2,:], c=0.6*np.ones(3))    
        ax.scatter(ks[0,0], ks[1,0], ks[2,0], s=5, c='k') 
        ax.scatter(ks[0,-1], ks[1,-1], ks[2,-1], s=30, edgecolor='w', facecolor='k') 
    plt.xlabel(r'$\kappa_1$')
    plt.ylabel(r'$\kappa_2$')
    ax.set_zlabel(r'$\kappa_3$')
    
    return(fig, ax, Trajs, Times, iTimes, time)

def run_FP_fs(M, N, T = 180, dt = 0.2, trajs=1):
    dims = np.shape(M)[1]
    time = np.arange(0, T, dt)
    
    for tr in range(trajs):
        ks = np.zeros((dims, len(time)))
        k0 = 0.1*np.random.randn(dims)
        ks[:,0]= k0
        for it, ti in enumerate(time[:-1]):
            ks[:,it+1] = ks[:,it] + dt*(def_field_net(ks[:,it], M, N))[0]
        
    
    return( np.dot(M,ks[:,-1]),ks[:,-1])
    

def plot_field(E_manif_net, En, U_manif_net, V_manif_net, theta, phi, rs, lw=1, 
               cb=True, alpha = 1, s_fp = 70, vmin = -5, vmax = -1, flow=True, density=1., log=True):
    PS, TH = np.meshgrid(phi, theta) 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if log:
        plt.pcolor(TH, PS, np.log10(E_manif_net), alpha=alpha, vmin = vmin, vmax=vmax, shading='auto')
        if cb:
            cbar = plt.colorbar()
            if vmin == -5 and vmax== -1:
                cbar.set_ticks([-5, -3, -1])
            else:
                cbar.set_ticks([vmin, vmax])
    else:
        plt.pcolor(TH, PS, E_manif_net, alpha=alpha, shading='auto')
        if cb:
            cbar = plt.colorbar()
            if vmin == -5 and vmax== -1:
                cbar.set_ticks([-5, -3, -1])
            else:
                cbar.set_ticks([vmin, vmax])
    if flow==True:
        plt.streamplot(theta, phi, U_manif_net.T, V_manif_net.T, color='w', linewidth=lw, density=density)

    plt.xlim(np.min(theta), np.max(theta))
    plt.ylim(np.min(phi), np.max(phi))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')
    
    return(fig, ax)

def initialize_patterns(M,N, input_size, output_size):
    hidden_size = np.shape(M)[0]
    rank = np.shape(M)[1]
    dtype = torch.FloatTensor  
    mrec_i = M.T/np.sqrt(hidden_size)
    nrec_i = N.T/np.sqrt(hidden_size)
    mrec_I = torch.from_numpy(mrec_i.T).type(dtype)
    nrec_I = torch.from_numpy(nrec_i.T).type(dtype)

    #initial inputs and outputs fixed, random, non-orthogonalized
    inp_i = np.random.randn(hidden_size,input_size).T
    randm_coeff = np.random.randn(1,rank)
    inp_i[0,:] = np.dot(randm_coeff, nrec_i)[0,:]
    randm_coeff = np.random.randn(1,rank)
    inp_i[1,:] = np.dot(randm_coeff, nrec_i)[0,:]
    
    inp_i += np.random.randn(hidden_size,2).T
    
    out_i = np.random.randn(hidden_size, output_size)
    randm_coeff = np.random.randn(1,rank)
    out_i[:,0] = np.dot(randm_coeff, mrec_i)[0,:]
    out_i += np.random.randn(hidden_size,output_size)
    
    out_i = out_i/hidden_size
    
    inp_I = torch.from_numpy(inp_i).type(dtype)
    out_I = torch.from_numpy(out_i).type(dtype)
    return(mrec_I, nrec_I, inp_I,out_I)

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

def Phi (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Prime (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def get_weights(net_low):
    M = net_low.m.detach().numpy()
    N = net_low.n.detach().numpy()
    corr = np.dot(M.T, N)
    I = net_low.wi.detach().numpy()
    O = net_low.wo.detach().numpy()
    return(M, N, corr, I, O)

def get_SVDweights(net_low, rank=3):
    M = net_low.m.detach().numpy()
    N = net_low.n.detach().numpy()
    
    J_pre = np.dot(M, N.T)
    u, s, v = np.linalg.svd(J_pre)
    M_pre = u[:,0:rank]
    N_pre = np.diag(s[0:rank]).dot( v[0:rank,:]).T
    corr_pre = np.dot(M_pre.T, N_pre)
    return(M_pre, N_pre, corr_pre, J_pre)

def get_SVD_MN(M, N):
    rank = np.shape(M)[1]
    hidden_size = np.shape(M)[0]
    J_pre = np.dot(M, N.T)/hidden_size
    u, s, v = np.linalg.svd(J_pre)
    M_pre = u[:,0:rank]*np.sqrt(hidden_size)
    N_pre = (1/np.sqrt(hidden_size))*np.diag(s[0:rank]).dot( v[0:rank,:]).T
    return(M_pre, N_pre, J_pre)

def get_SVDweights_CSG(net_low, rank=3):
    M = net_low.m.detach().numpy()
    N = net_low.n.detach().numpy()
    
    J_pre = np.dot(M, N.T)
    u, s, v = np.linalg.svd(J_pre)
    M_pre = u[:,0:rank]
    N_pre = np.diag(s[0:rank]).dot( v[0:rank,:]).T
    corr_pre = np.dot(M_pre.T, N_pre)
    I = net_low.wi.detach().numpy()
    O = net_low.wo.detach().numpy()
    return(M_pre, N_pre, corr_pre, I, O, J_pre)

def create_inp_out_MWG(trials, Nt, tss, R1_on, SR1_on, fact = 1., just=-1,  perc = 0.1, perc1 = 0.1, delayF = 0, 
                            delay_min = 20, delay_max = 250, align_set = False, inp_size=2, inc_mask_pre = 30, inc_mask_post = 30 ):
    '''
    Inputs
    ------
    trials:     Number of trials
    Nt :        Number of time points
    tss :       Intervals between set and go
    R1_on:      Time of ready
    SR1_on:     Possible deviation of the onset of "Ready".
    fact:       Scaling factor for the sampled interval (dividing)
    just:       Not given: all intervals, otherwise, selected interval index
    perc:       Percentage of trials in which no inputs appear
    perc1:      Percentage of trials in which only the ready cue appears
    delayF:     Fixed delay (if not given, variable)
    delay_min:  Minimum delay
    delay_max:  Maximum delay
    noset:      
    noready:
    align_set: “False对齐Ready信号True对齐Set信号”
        
    Outputs
    -------
    inputt:
    outputt:
    maskt:
    ct: Interval index at every trial
    ct2: Trials without inputs
    ct3: Trials without Set inputs
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)  # Produced intervals
    strt = -0.5                     # Initial readout value

    if inp_size==2:
        inputt  = np.zeros(( trials, Nt, 2))
    else:
        inputt  = np.zeros(( trials, Nt, 3))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    interval = np.min(tss_comp)//2 # Minimal interval to be produced
    #inc_mask = 30                  # Minimal numbe of time points 
    
    s_inp_R =  np.zeros((trials, Nt))   #Ready
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
        rnd = np.random.randint(-SR1_on, SR1_on, trials) #random deviation at Ready onset
        
    for itr in range(trials):      
        redset = tss[ct[itr]]           #produced interval
        redset_comp = tss_comp[ct[itr]] #measurement interval
        delay = np.random.randint(delay_min, delay_max)
    
        if not align_set:                           #Align at Ready
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+delay)
            mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]+delay
            
            #Create Ready
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
            # Create End of Measurement
            s_inp_S1[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
            s_inp_S1[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
            #Create Set
            s_inp_S2[itr, time>R1_on+rnd[itr]+redset_comp+delay] = 10.
            s_inp_S2[itr, time>1+R1_on+rnd[itr]+redset_comp+delay] = 0.
            
            # Create output
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask_pre+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask_post+delay)
    
        else:              #Align at Set
            fixT = R1_on + np.max(tss) #Set time (last input)  
            redset = tss[ct[itr]] #produced
            redset_comp = tss_comp[ct[itr]] #measurement

            maskt[itr,:,0] = (time>1+fixT-rnd[itr])*(time<redset+fixT+1-rnd[itr])#(time>1+fixT-inc_mask)*(time<redset+fixT+1+inc_mask)
            mask_aft = time>=redset+1+fixT-rnd[itr]
            
            s_inp_R[itr, time>fixT-redset_comp-rnd[itr]-delayF] = 10.
            s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]-delayF] = 0.
            
            s_inp_S1[itr, time>fixT-delayF-rnd[itr]] = 10.
            s_inp_S1[itr, time>1+fixT-delayF-rnd[itr]] = 0.
            
            
            s_inp_S2[itr, time>fixT-rnd[itr]] = 10.
            s_inp_S2[itr, time>1+fixT-rnd[itr]] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True) 
                
            maskt[itr,:,0] = (time>1+fixT-inc_mask_pre-rnd[itr])*(time<redset+fixT+1+inc_mask_post-rnd[itr])
            
            
        if ct2[itr]==True:
            s_inp_R[itr,:] = 0.
            s_inp_S1[itr,:] = 0.
            s_inp_S2[itr,:] = 0.
            maskt[itr,:,0] = time<Nt
            outputt[itr,:,0] = strt
        elif ct3[itr]==True:
            s_inp_S2[itr,:] = 0.      
            outputt[itr,:,0] = strt
            maskt[itr,:,0] = time<Nt
    
    if inp_size==2:                
        inputt[:,:,0] += s_inp_R
        inputt[:,:,0] += s_inp_S1
        inputt[:,:,1] += s_inp_S2
    else:
        inputt[:,:,0] += s_inp_R
        inputt[:,:,1] += s_inp_S1
        inputt[:,:,2] += s_inp_S2
        
        
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3)

def create_inp_out_MWG2(trials, Nt, tss, tss2, R1_on, SR1_on, fact = 1., just=-1,  perc = 0.1, perc1 = 0.1, delayF = 0, 
                            delay_min = 20, delay_max = 250, align_set = False, inp_size=4, s1 = 0., s2 = 0.1, just2=-1):
    '''
    Inputs
    ------
    trials:     Number of trials
    Nt :        Number of time points
    tss :       Intervals between set and go
    R1_on:      Time of ready
    SR1_on:     Possible deviation of the onset of "Ready".
    fact:       Scaling factor for the sampled interval (dividing)
    just:       Not given: all intervals, otherwise, selected interval index
    perc:       Percentage of trials in which no inputs appear
    perc1:      Percentage of trials in which only the ready cue appears
    delayF:     Fixed delay (if not given, variable)
    delay_min:  Minimum delay
    delay_max:  Maximum delay
    noset:      
    noready:
    align_set:
        
    Outputs
    -------
    inputt:
    outputt:
    maskt:
    ct: Interval index at every trial
    ct2: Trials without inputs
    ct3: Trials without Set inputs
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)  # Produced intervals
    tss_comp2 = np.round(tss2/fact)  # Produced intervals
    
    strt = -0.5                     # Initial readout value


    inputt  = np.zeros(( trials, Nt, inp_size))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    interval = np.min(tss_comp)//2 # Minimal interval to be produced
    inc_mask = 30                  # Minimal numbe of time points 
    
    s_inp_R =  np.zeros((trials, Nt))   #Ready
    s_inp_S1 =  np.zeros((trials, Nt))  
    s_inp_S2 =  np.zeros((trials, Nt))
    s_inp_S3 =  np.zeros((trials, Nt))
    
    
    
    if delayF==0:
        delayF = np.round(np.mean((delay_min, delay_max)))
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
        ct_ctxt  = np.random.randint(2, size = trials)
    else:
        ct = just*np.ones(trials, dtype = np.int8)
        if just2 == 1:
            ct_ctxt = just2*np.ones(trials)
        else:
            ct_ctxt = np.zeros(trials)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials) #random deviation at Ready onset
        
    for itr in range(trials):  
        if ct_ctxt[itr] == 0:
            redset = tss[ct[itr]]           #produced interval
            redset_comp = tss_comp[ct[itr]] #measurement interval
            ss = s1
            
        else:
            redset = tss2[ct[itr]]           #produced interval
            redset_comp = tss_comp2[ct[itr]] #measurement interval
            ss = s2
        delay = np.random.randint(delay_min, delay_max)
           
        s_inp_S3[itr, :] = ss
        if not align_set:                           #Align at Ready
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+delay)
            mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]+delay
            
            #Create Ready
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
            # Create End of Measurement
            s_inp_S1[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
            s_inp_S1[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
            #Create Set
            s_inp_S2[itr, time>R1_on+rnd[itr]+redset_comp+delay] = 10.
            s_inp_S2[itr, time>1+R1_on+rnd[itr]+redset_comp+delay] = 0.
          
            
            # Create output
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask+delay)
    
        else:              #Align at Set
            fixT = R1_on + np.max(tss) #Set time (last input)  
            fixT2 = R1_on + np.max(tss2) #Set time (last input)  
            fixT = np.max((fixT, fixT2))
            if ct_ctxt[itr] == 0:
                redset = tss[ct[itr]] #produced
                redset_comp = tss_comp[ct[itr]] #measurement
            else:
                redset = tss2[ct[itr]]
                redset_comp = tss_comp2[ct[itr]]

            maskt[itr,:,0] = (time>1+fixT-rnd[itr])*(time<redset+fixT+1-rnd[itr])#(time>1+fixT-inc_mask)*(time<redset+fixT+1+inc_mask)
            mask_aft = time>=redset+1+fixT-rnd[itr]
            
            s_inp_R[itr, time>fixT-redset_comp-rnd[itr]-delayF] = 10.
            s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]-delayF] = 0.
            
            s_inp_S1[itr, time>fixT-delayF-rnd[itr]] = 10.
            s_inp_S1[itr, time>1+fixT-delayF-rnd[itr]] = 0.
            
            
            s_inp_S2[itr, time>fixT-rnd[itr]] = 10.
            s_inp_S2[itr, time>1+fixT-rnd[itr]] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True) 
                
            maskt[itr,:,0] = (time>1+fixT-inc_mask-rnd[itr])*(time<redset+fixT+1+inc_mask-rnd[itr])
            
            
    if inp_size==2:                
        inputt[:,:,0] += s_inp_R
        inputt[:,:,0] +=   s_inp_S1
        inputt[:,:,1] +=   s_inp_S2
    else:
        inputt[:,:,0] += s_inp_R
        inputt[:,:,1] +=   s_inp_S1
        inputt[:,:,2] +=   s_inp_S2   
        inputt[:,:,3] +=   s_inp_S3   
            
   
        
        
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3, ct_ctxt)

def create_inp_out_MWGLearn( Nt, tss, sss, R1_on, SR1_on, fact = 1., just=-1,  perc = 0.1, perc1 = 0.1, delayF = 0, 
                            delay_min = 20, delay_max = 250, align_set = False, inp_size=4, s1 = 0., s2 = 0.1, just2=-1):
    '''
    Inputs
    ------
    trials:     Number of trials
    Nt :        Number of time points
    tss :       Intervals between set and go
    R1_on:      Time of ready
    SR1_on:     Possible deviation of the onset of "Ready".
    fact:       Scaling factor for the sampled interval (dividing)
    just:       Not given: all intervals, otherwise, selected interval index
    perc:       Percentage of trials in which no inputs appear
    perc1:      Percentage of trials in which only the ready cue appears
    delayF:     Fixed delay (if not given, variable)
    delay_min:  Minimum delay
    delay_max:  Maximum delay
    noset:      
    noready:
    align_set:
        
    Outputs
    -------
    inputt:
    outputt:
    maskt:
    ct: Interval index at every trial
    ct2: Trials without inputs
    ct3: Trials without Set inputs
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)  # Produced intervals
    
    strt = -0.5                     # Initial readout value

    trials = len(sss)
    inputt  = np.zeros(( trials, Nt, inp_size))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    interval = np.min(tss_comp)//2 # Minimal interval to be produced
    inc_mask = 30                  # Minimal numbe of time points 
    
    s_inp_R =  np.zeros((trials, Nt))   #Ready
    s_inp_S1 =  np.zeros((trials, Nt))  
    s_inp_S2 =  np.zeros((trials, Nt))
    s_inp_S3 =  np.zeros((trials, Nt))
    
    
    
    if delayF==0:
        delayF = np.round(np.mean((delay_min, delay_max)))
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
        ct_ctxt  = np.random.randint(2, size = trials)
    else:
        ct = just*np.ones(trials, dtype = np.int8)
        if just2 == 1:
            ct_ctxt = just2*np.ones(trials)
        else:
            ct_ctxt = np.zeros(trials)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials) #random deviation at Ready onset
        
    for itr in range(trials):  
        
        redset = tss[itr]           #produced interval
        redset_comp = tss_comp[itr] #measurement interval
        ss = sss[itr]
        delay = np.random.randint(delay_min, delay_max)
           
        s_inp_S3[itr, :] = ss
        if not align_set:                           #Align at Ready
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+delay)
            mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]+delay
            
            #Create Ready
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
            # Create End of Measurement
            s_inp_S1[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
            s_inp_S1[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
            #Create Set
            s_inp_S2[itr, time>R1_on+rnd[itr]+redset_comp+delay] = 10.
            s_inp_S2[itr, time>1+R1_on+rnd[itr]+redset_comp+delay] = 0.
          
            
            # Create output
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask+delay)
    
        else:              #Align at Set
            fixT = R1_on + np.max(tss) #Set time (last input)  
            
            
            redset = tss[itr] #produced
            redset_comp = tss_comp[itr] #measurement

            maskt[itr,:,0] = (time>1+fixT-rnd[itr])*(time<redset+fixT+1-rnd[itr])#(time>1+fixT-inc_mask)*(time<redset+fixT+1+inc_mask)
            mask_aft = time>=redset+1+fixT-rnd[itr]
            
            s_inp_R[itr, time>fixT-redset_comp-rnd[itr]-delayF] = 10.
            s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]-delayF] = 0.
            
            s_inp_S1[itr, time>fixT-delayF-rnd[itr]] = 10.
            s_inp_S1[itr, time>1+fixT-delayF-rnd[itr]] = 0.
            
            
            s_inp_S2[itr, time>fixT-rnd[itr]] = 10.
            s_inp_S2[itr, time>1+fixT-rnd[itr]] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True) 
                
            maskt[itr,:,0] = (time>1+fixT-inc_mask-rnd[itr])*(time<redset+fixT+1+inc_mask-rnd[itr])
            
            
    if inp_size==2:                
        inputt[:,:,0] += s_inp_R
        inputt[:,:,0] +=   s_inp_S1
        inputt[:,:,1] +=   s_inp_S2
    else:
        inputt[:,:,0] += s_inp_R
        inputt[:,:,1] +=   s_inp_S1
        inputt[:,:,2] +=   s_inp_S2   
        inputt[:,:,3] +=   s_inp_S3   
            
   
        
        
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3, ct_ctxt)

def create_inp_out_MWG3(trials, Nt, tss, tss2, R1_on, SR1_on, fact = 1., just=-1,  perc = 0.1, perc1 = 0.1, delayF = 0, 
                            delay_min = 20, delay_max = 250, align_set = False, inp_size=4, s1 = 0., s2 = 0.1, just2=-1, rem_prod = False, rem_meas=False):
    '''
    Inputs
    ------
    trials:     Number of trials
    Nt :        Number of time points
    tss :       Intervals between set and go
    R1_on:      Time of ready
    SR1_on:     Possible deviation of the onset of "Ready".
    fact:       Scaling factor for the sampled interval (dividing)
    just:       Not given: all intervals, otherwise, selected interval index
    perc:       Percentage of trials in which no inputs appear
    perc1:      Percentage of trials in which only the ready cue appears
    delayF:     Fixed delay (if not given, variable)
    delay_min:  Minimum delay
    delay_max:  Maximum delay
    noset:      
    noready:
    align_set:
        
    Outputs
    -------
    inputt:
    outputt:
    maskt:
    ct: Interval index at every trial
    ct2: Trials without inputs
    ct3: Trials without Set inputs
    '''
    
    n_ts = len(tss)
    time = np.arange(Nt)
    
    tss_comp = np.round(tss/fact)  # Produced intervals
    tss_comp2 = np.round(tss2/fact)  # Produced intervals
    
    strt = -0.5                     # Initial readout value


    inputt  = np.zeros(( trials, Nt, inp_size))
    outputt = strt*np.ones((trials, Nt, 1))
    maskt   = np.zeros((trials, Nt, 1))
    interval = np.min(tss_comp)//2 # Minimal interval to be produced
    inc_mask = 30                  # Minimal numbe of time points 
    
    s_inp_R =  np.zeros((trials, Nt))   #Ready
    s_inp_S1 =  np.zeros((trials, Nt))  
    s_inp_S2 =  np.zeros((trials, Nt))
    s_inp_S3 =  np.zeros((trials, Nt))
    
    
    
    if delayF==0:
        delayF = np.round(np.mean((delay_min, delay_max)))
    
    if just==-1:   #all types of trials  
        ct = np.random.randint(n_ts, size = trials)
        ct_ctxt  = np.random.randint(2, size = trials)
    else:
        ct = just*np.ones(trials, dtype = np.int8)
        if just2 == 1:
            ct_ctxt = just2*np.ones(trials)
        else:
            ct_ctxt = np.zeros(trials)
    
    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials)<perc
    
    # Don't have a set cue
    ct3 = np.random.rand(trials)<perc1
    
    
    rnd = np.zeros(trials)
    if SR1_on>0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials) #random deviation at Ready onset
        
    for itr in range(trials):  
        if ct_ctxt[itr] == 0:
            redset = tss[ct[itr]]           #produced interval
            redset_comp = tss_comp[ct[itr]] #measurement interval
            ss = s1
            
        else:
            redset = tss2[ct[itr]]           #produced interval
            redset_comp = tss_comp2[ct[itr]] #measurement interval
            ss = s2
        delay = np.random.randint(delay_min, delay_max)
           
        s_inp_S3[itr, :] = ss
        
        if not align_set:                           #Align at Ready
            maskt[itr,:,0] = (time>R1_on+1+rnd[itr]+redset_comp+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+delay)
            mask_aft = time>=redset_comp+redset+R1_on+1+rnd[itr]+delay
            
            #Create Ready
            s_inp_R[itr, time>R1_on+rnd[itr]] = 10.
            s_inp_R[itr, time>R1_on+rnd[itr]+1] = 0.
            # Create End of Measurement
            s_inp_S1[itr, time>R1_on+rnd[itr]+redset_comp] = 10.
            s_inp_S1[itr, time>1+R1_on+rnd[itr]+redset_comp] = 0.
            #Create Set
            s_inp_S2[itr, time>R1_on+rnd[itr]+redset_comp+delay] = 10.
            s_inp_S2[itr, time>1+R1_on+rnd[itr]+redset_comp+delay] = 0.
            
            if rem_prod or rem_meas:
                inter_point = 1+R1_on+rnd[itr]+redset_comp + np.int(0.5*delay)
                if rem_prod:
                    s_inp_S3[itr, time>inter_point] = 0.
                if rem_meas:
                    s_inp_S3[itr, time<inter_point] = 0.
            
            # Create output
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)
                
            maskt[itr,:,0] = (time>redset_comp+R1_on+1+rnd[itr]-inc_mask+delay)*(time<redset_comp+redset+R1_on+1+rnd[itr]+inc_mask+delay)
    
        else:              #Align at Set
            fixT = R1_on + np.max(tss) #Set time (last input)  
            fixT2 = R1_on + np.max(tss2) #Set time (last input)  
            fixT = np.max((fixT, fixT2))
            if ct_ctxt[itr] == 0:
                redset = tss[ct[itr]] #produced
                redset_comp = tss_comp[ct[itr]] #measurement
            else:
                redset = tss2[ct[itr]]
                redset_comp = tss_comp2[ct[itr]]

            maskt[itr,:,0] = (time>1+fixT-rnd[itr])*(time<redset+fixT+1-rnd[itr])#(time>1+fixT-inc_mask)*(time<redset+fixT+1+inc_mask)
            mask_aft = time>=redset+1+fixT-rnd[itr]
            
            s_inp_R[itr, time>fixT-redset_comp-rnd[itr]-delayF] = 10.
            s_inp_R[itr, time>fixT-redset_comp+1-rnd[itr]-delayF] = 0.
            
            s_inp_S1[itr, time>fixT-delayF-rnd[itr]] = 10.
            s_inp_S1[itr, time>1+fixT-delayF-rnd[itr]] = 0.
            
            
            s_inp_S2[itr, time>fixT-rnd[itr]] = 10.
            s_inp_S2[itr, time>1+fixT-rnd[itr]] = 0.
            
            if rem_prod or rem_meas:
                inter_point = fixT-np.int(0.5*delayF)-rnd[itr]
                if rem_prod:
                    s_inp_S3[itr, time>inter_point] = 0.
                if rem_meas:
                    s_inp_S3[itr, time<inter_point] = 0.
            
            if sum(maskt[itr,:,0]):
                outputt[itr, maskt[itr,:,0]==1.,0] = np.linspace(strt, -strt, int(sum(maskt[itr,:,0])), endpoint=True)
                outputt[itr, mask_aft==1,0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True) 
                
            maskt[itr,:,0] = (time>1+fixT-inc_mask-rnd[itr])*(time<redset+fixT+1+inc_mask-rnd[itr])
            
            
    if inp_size==2:                
        inputt[:,:,0] += s_inp_R
        inputt[:,:,0] +=   s_inp_S1
        inputt[:,:,1] +=   s_inp_S2
    else:
        inputt[:,:,0] += s_inp_R
        inputt[:,:,1] +=   s_inp_S1
        inputt[:,:,2] +=   s_inp_S2   
        inputt[:,:,3] +=   s_inp_S3   
            
   
        
        
    dtype = torch.FloatTensor   
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)
    
    return(inputt, outputt, maskt, ct, ct2, ct3, ct_ctxt)

def create_inp_out2(trials, Nt, tss, amps, R_on, SR_on, just=-1,  perc = 0.2):
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
    s_inp = np.zeros((trials, Nt))
    
    
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
            s_inp[itr, time>R_on+rnd[itr]] = 10. #10.
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

#%%
def give_field_CSG(M, N, Iv, amp, k1s, k2s):
    K1s, K2s = np.meshgrid(k1s, k2s)
    U = np.zeros_like(K1s)
    V = np.zeros_like(K2s)
    E = np.zeros_like(K2s)
    hidden_size = np.shape(M)[0]
    
    for ik1, k1 in enumerate(k1s):
        for ik2, k2 in enumerate(k2s):
            x = k1*M[:,0]*np.sqrt(hidden_size) + k2*M[:,1]*np.sqrt(hidden_size) + amp*Iv
            Ph = np.tanh(x)
            Ph = Ph[:, np.newaxis]
            
            Field = -x+M.dot(N.T.dot(Ph))[:,0]+ amp*Iv
            U[ik1, ik2] = np.sum(M[:,0]*Field)/(np.sqrt(hidden_size)*np.sum(M[:,0]**2))
            V[ik1, ik2] = np.sum(M[:,1]*Field)/(np.sqrt(hidden_size)*np.sum(M[:,1]**2))
            K1s[ik1, ik2] = np.sum(M[:,0]*x)/(np.sqrt(hidden_size)*np.sum(M[:,0]**2))
            K2s[ik1, ik2] = np.sum(M[:,1]*x)/(np.sqrt(hidden_size)*np.sum(M[:,1]**2))
            
    E = np.sqrt(U**2+V**2)
    return( E, U, V, K1s, K2s)

#%%
def get_field(net_low_all, k1s, k2s, Amp, rank=2, hidden_size=1500):
    K1s, K2s = np.meshgrid(k1s, k2s)
    M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = get_SVDweights_CSG(net_low_all, rank=rank)

    G0 = np.zeros_like(K1s)
    G1 = np.zeros_like(K2s)
    
    for ik1, k1 in enumerate(k1s):
        for ik2, k2 in enumerate(k2s):
            m1 = M_pre[:,0]*np.sqrt(hidden_size)
            m2 = M_pre[:,1]*np.sqrt(hidden_size)
            x = k1*m1 + k2*m2 + Amp*I_pre[0,:].T
            dx = -x + J_pre.dot(np.tanh(x)) +Amp*I_pre[0,:].T
            G0[ik1, ik2] = np.mean(m1*dx)/np.mean(m1*m1)
            G1[ik1, ik2] = np.mean(m2*dx)/np.mean(m2*m2)
    Q = np.sqrt(G0**2+G1**2)
    return(G0, G1, Q, m1, m2, I_pre, J_pre)

#%%
def get_manifold(thetas, Q, G0, G1, k1s, k2s, m1, m2, Amp, I_pre, J_pre, dim = 0.02):
    array = np.copy(Q)
    trajs1 = []
    trajs2 = []
    st_fp =[]
    u_fp = []
    for rep in range(5):
        x,y = np.unravel_index(np.argmin(array),array.shape)
        k1p = k1s[x]
        k2p = k2s[y]

            
        Ja = np.zeros((2,2))
        Ja[0,0] = G0[x+1,y]-G0[x-1,y]
        Ja[0,1] = G1[x+1,y]-G1[x-1,y]
        Ja[1,1] = G1[x,y+1]-G1[x,y-1]
        Ja[1,0] = G0[x,y+1]-G0[x,y-1]
        lam = np.linalg.eigvals(Ja)
        eh, vh = np.linalg.eig(Ja)
        idx = eh.argsort()[::-1]
        eh = eh[idx]
        vh = vh[:,idx]
        array[x-4:x+4,y-4:y+4] = np.max(Q)
        if np.max(lam)>0:
            u_fp.append(np.array((k2p, k1p)))
        else:
            st_fp.append(np.array((k2p, k1p)))
        if np.max(lam)>0 and np.min(lam)<0:
            
            if np.real(eh[0])>0:
                iXX = 0
            else:
                iXX = 1
                print('hey')
            time = np.arange(0, 40, 0.1)
            dt = time[1]-time[0]
            r1 = np.zeros_like(time)
            r2 = np.zeros_like(time)
            r1_ = np.zeros_like(time)
            r2_ = np.zeros_like(time)
            
            x = (k1p+dim*vh[0,iXX])*m1 + (k2p+dim*vh[1,iXX])*m2 + Amp*I_pre[0,:].T
            r1[0] = np.mean(m1*x)/np.mean(m1*m1)
            r2[0] = np.mean(m2*x)/np.mean(m2*m2)
            for it, ti in enumerate(time[:-1]):
                dx = -x + J_pre.dot(np.tanh(x)) +Amp*I_pre[0,:].T
                x = x+dt*dx
                r1[it+1] = np.mean(m1*x)/np.mean(m1*m1)
                r2[it+1] = np.mean(m2*x)/np.mean(m2*m2)
            x = (k1p-dim*vh[0,iXX])*m1 + (k2p-dim*vh[1,iXX])*m2 + Amp*I_pre[0,:].T
            r1_[0] = np.mean(m1*x)/np.mean(m1*m1)
            r2_[0] = np.mean(m2*x)/np.mean(m2*m2)
            for it, ti in enumerate(time[:-1]):
                dx = -x + J_pre.dot(np.tanh(x)) +Amp*I_pre[0,:].T
                x = x+dt*dx
                r1_[it+1] = np.mean(m1*x)/np.mean(m1*m1)
                r2_[it+1] = np.mean(m2*x)/np.mean(m2*m2)
            
            trajs1.extend(np.hstack((r1, r1_)))
            trajs2.extend(np.hstack((r2, r2_)))
    
    trajs1 = np.array(trajs1)
    trajs2 = np.array(trajs2)
    
    th_M = np.arctan2(trajs2,trajs1)
    R_M  = np.sqrt(trajs2**2+trajs1**2)
    
    Rth = np.zeros_like(thetas)
    Qs = np.zeros_like(thetas)
    GGms = np.zeros((len(thetas), 2))
    
    for it, th in enumerate(thetas):
        iT  = np.argmin(np.abs(th-th_M))
        Rth[it] = R_M[iT]
        x = R_M[iT]*np.cos(th)*m1 + R_M[iT]*np.sin(th)*m2 + Amp*I_pre[0,:].T
        dx = -x + J_pre.dot(np.tanh(x)) +Amp*I_pre[0,:].T
        GGms[it,0] = np.mean(m1*dx)/np.mean(m1*m1)
        GGms[it,1] = np.mean(m2*dx)/np.mean(m2*m2)
        sign = np.sign(np.sin(th)*GGms[it,0]-np.cos(th)*GGms[it,1])
        Qs[it] = sign*np.sqrt(GGms[it,0]**2+GGms[it,1]**2)
    return(Qs, Rth, trajs1, trajs2, st_fp, u_fp)

#%%
def plot_output_MWG2(net_low_all, net_low_fr, tss2, dt, CLL, time, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, dela = 150):
    if fr==False:
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = get_SVDweights_CSG(net_low_all, rank=rank)
    if plot:
        fig_width = 1.5*2.2 # width in inches
        fig_height = 1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        

    factor=1
    R_on = 100
    Nt = len(time)
    trials = 10

    T0=4100
    
    for xx in range(len(tss2)):
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela, inp_size=3)
        outp, traj = net_low_fr.forward(input_train, return_dynamics=True)
        outp = outp.detach().numpy()

        
        avg_outp0 = np.mean(outp[:,:,0],0)#np.mean(outp3[:,:,0],0)
        if xx==0:
            ax.plot(time*dt-T0, avg_outp0, '--', color='k', lw=2, alpha=0.8, label='full rank')
            ax.plot(time*dt-T0, avg_outp0, '--', color=CLL[xx,:],  lw=2)
        else:
            ax.plot(time*dt-T0, avg_outp0, '--', color=CLL[xx,:],  lw=2)
                    
            
    for xx in range(len(tss2)):
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela, inp_size=3)
        outp, traj = net_low_all.forward(input_train, return_dynamics=True)
        outp = outp.detach().numpy()
        
        avg_outp0 = np.mean(outp[:,:,0],0)#np.mean(outp3[:,:,0],0)


        if xx==0:
            ax.plot(time*dt-T0, avg_outp0, color='k', alpha=0.8, lw=2, label='rank three')   
            plt.legend()
            ax.plot(time*dt-T0, avg_outp0, color=CLL[xx,:], lw=2)    
        else:
            ax.plot(time*dt-T0, avg_outp0, color=CLL[xx,:], lw=2) 
            
        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '.', color='k', alpha=0.5) #This is for the mask (in all four)
             

    if plot:                    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plt.xlabel('time after Set (ms)')
        plt.ylabel('read out')
        plt.xlim([-300, 1850])
        plt.yticks([-0.5, 0, 0.5])
        plt.xticks([0, 500, 1000, 1500])
        ax.set_xticklabels(['0', '', '1000', ''])

        
    return(fig, ax)

#%%
def give_traj_MWG2(net_low_all, net_low_fr, tss2, dt, CLL, time, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, dela = 150):
    if fr==False:
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = get_SVDweights_CSG(net_low_all, rank=rank)
        
    factor=1
    R_on = 100
    Nt = len(time)
    trials = 10
    
    for xx in range(len(tss2)):
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela, inp_size=3)
        outp, traj = net_low_fr.forward(input_train, return_dynamics=True)
        if xx==0:
            Traj_fr = traj.detach().numpy()
            #print(np.shape(Traj_fr))
        else:
            Traj_fr = np.vstack((Traj_fr, traj.detach().numpy()))
            #print(np.shape(Traj_fr))
                
            
    for xx in range(len(tss2)):
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela, inp_size=3)
        outp, traj = net_low_all.forward(input_train, return_dynamics=True)
        if xx==0:
            Traj = traj.detach().numpy()
        else:
            Traj = np.vstack((Traj, traj.detach().numpy()))
    return(Traj, Traj_fr)