import torch
from .base import BaseOperator
from scipy.special import hankel1
from scipy.integrate import dblquad
import numpy as np
import os
def A_adjoint(z, f, g, dx, dy):
    """
    Adjoint of A = I - G*diag(f) operator required for forward scattering
    
    Input:
    - z: (Ny x Nx) field
    - f: (Ny x Nx) object
    - g: 2*(Ny x Nx) Green's function
    - dx: Grid spacing in x direction
    - dy: Grid spacing in y direction
    - conv2DAdj: Function for adjoint 2D convolution
    
    Output:
    - u: (Ny x Nx) field
    """
    # Validate the size
    assert g.shape == (2 * z.shape[0], 2 * z.shape[1]), 'g should be twice the size of z'
    assert f.shape == z.shape, 'f should be the same size as z'
    
    # Perform adjoint operation
    u = z - dx * dy * torch.conj(f) * conv2DAdj(z, g)    
    return u

def conv2DAdj(z, g):
    """
    Convolves with the adjoint of the function g using PyTorch.
    
    Input:
    - z: (Ny x Nx) tensor
    - g: 2*(Ny x Nx) tensor
    
    Output:
    - f: (Ny x Nx) output tensor
    """
    assert g.shape == (2 * z.shape[0], 2 * z.shape[1])
    
    # Shift
    g = torch.fft.ifftshift(g)
    
    # Size of computational domain and number of inputs
    Ny, Nx = z.shape
    
    # Zero post-padding
    z_padded = torch.nn.functional.pad(z, (0, Ny, 0, Nx), mode='constant', value=0)
    
    # Convolve with the complex conjugate of g in the frequency domain
    f_padded = torch.fft.ifft2(torch.fft.fft2(z_padded) * torch.conj(torch.fft.fft2(g)))
    
    # Remove zero-padding
    f = f_padded[:Ny, :Nx]
    
    return f

def A_forward(u, f, g, dx, dy):
    '''
    A*u = (I - G*diag(f))*u operator required for forward scattering
    
    Input:
    - u: (Ny x Nx) field
    - f: (Ny x Nx) object
    - g: 2*(Ny x Nx) Green's function
    - dx: scalar
    - dy: scalar
    
    Output:
    - z: (Ny x Nx) field
    '''
    
    assert g.shape == (2 * u.shape[0], 2 * u.shape[1]), 'g should be twice the size of u'
    assert f.shape == u.shape
    
    z = u - dx * dy * conv2D(f * u, g)
    return z

def conv2D(f, g):
    '''
    Convolves f with the function g using PyTorch.
    
    Input:
    - f: (Ny x Nx) tensor
    - g: 2*(Ny x Nx) tensor
    
    Output:
    - z: (Ny x Nx) output tensor
    '''
    
    assert g.shape == (2 * f.shape[0], 2 * f.shape[1])
    
    g = torch.fft.ifftshift(g)
    
    Ny, Nx = f.shape
    
    f_padded = torch.nn.functional.pad(f, (0, Ny, 0, Nx), mode='constant', value=0)

    # Convolve
    z_padded = torch.fft.ifft2(torch.fft.fft2(f_padded) * torch.fft.fft2(g))

    # Remove zero-padding
    z = z_padded[:Ny, :Nx]
    
    return z

def propagate_to_sensor(f, uin, g, dx, dy):
    """
    Computes the scattered field at the sensor locations specified by the
    set of Green's functions using PyTorch.

    Parameters:
    - f: (Ny x Nx) scattering potential tensor
    - uin: (Ny x Nx) input field tensor
    - g: (Ny x Nx x Nr) Green's functions tensor
    - dx, dy: sampling steps (scalars)

    Returns:
    - uscat: (Nr x 1) scattered field tensor
    """
    # Number of transmissions
    Nr = g.shape[2]

    # Contrast-source
    cont_src = f * uin
    cont_src = cont_src.unsqueeze(2).repeat(1, 1, Nr)

    # Compute propagated field
    uscat = dx * dy * torch.sum(torch.sum(g * cont_src, dim=0), dim=0)
    uscat = uscat.flatten()
    
    return uscat


def forward_prop(uinc_dom_set, f, domain_greens_function_set, utot_dom_set0, dx, dy):
    """
    Solves A * utot = uinc for all transmissions and frequencies using PyTorch.
    
    Parameters:
    uinc_dom_set (torch.Tensor): Incident field inside computational domain (Ny x Nx x numTrans x numFreq).
    f (torch.Tensor): Scattering potential (Ny x Nx).
    domain_greens_function_set (torch.Tensor): Green's functions for the domain (2 * Ny x 2 * Nx x numFreq).
    utot_dom_set0 (torch.Tensor): Initial guess for the total field (Ny x Nx x numTrans x numFreq).
    dx, dy (float): Sampling steps.

    Returns:
    torch.Tensor: Computed total field (Ny x Nx x numTrans x numFreq).
    """

    # Get dimensions
    _, _, num_trans, num_freq = uinc_dom_set.shape

    # Initialize the total field
    utot_dom_set = utot_dom_set0.clone()

    # Convergence flags
    conv_flags = torch.zeros((num_freq, num_trans), dtype=torch.int32)

    # Loop through frequencies and transmissions
    for ind_freq in range(num_freq):
        domain_greens_function = domain_greens_function_set[:, :, ind_freq]

        for ind_trans in range(num_trans):
            # Extract incident field
            uinc_dom = uinc_dom_set[:, :, ind_trans, ind_freq]
            utot_dom0 = utot_dom_set0[:, :, ind_trans, ind_freq]
            
            # Scattering operators as functions
            A = lambda u: A_forward(u, f, domain_greens_function, dx, dy)
            AT = lambda z: A_adjoint(z, f, domain_greens_function, dx, dy)
            
            # Compute total field using a conjugate gradient solver
            utot_dom, outs = pcg_wrap(lambda u: AT(A(u)), AT(uinc_dom), xinit=utot_dom0)
            
            # Store convergence flags
            conv_flags[ind_freq, ind_trans] = outs['flag']

            # Store the result
            utot_dom_set[:, :, ind_trans, ind_freq] = utot_dom

    return utot_dom_set, conv_flags

def pcg_wrap(A, b, **kwargs):
    """
    Wrapper function for the PCG solver.
    
    Parameters:
        A : callable
            Function representing the linear operator.
        b : ndarray
            Right-hand side of the linear system.
        kwargs : keyword arguments
            Optional parameters:
                xinit : ndarray
                    Initial guess for the solution.
                numiter : int
                    Maximum number of iterations (default: 1000).
                plotrecon : bool
                    Whether to plot results (default: False).
                tol : float
                    Tolerance for convergence (default: 1e-6).

    Returns:
        utotDom : ndarray
            Solution to the linear system.
        outs : dict
            Contains convergence flag and residuals.
    """
    pass


def generate_em_functions(p):
    # Meshgrid the pixel locations
    XPix, YPix = np.meshgrid(p['x'], p['y'])
    
    # Hankel function
    hank_fun = lambda x: 1j * 0.25 * hankel1(0, x)
    
    # Locations of transmitters and receivers
    transmitter_angles = np.linspace(0, 359, p['numTrans']) * np.pi / 180
    x_transmit = p['sensorRadius'] * np.cos(transmitter_angles)
    y_transmit = p['sensorRadius'] * np.sin(transmitter_angles)

    receiver_angles = np.linspace(0, 359, p['numRec']) * np.pi / 180
    x_receive = p['sensorRadius'] * np.cos(receiver_angles)
    y_receive = p['sensorRadius'] * np.sin(receiver_angles)
    
    # Distance data between sensors and pixels
    
    p['receiverMask'] = np.ones((p['numTrans'], p['numRec']))
    
    diff_x_rp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(x_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_rp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(y_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_rec_to_pix = np.sqrt(diff_x_rp**2 + diff_y_rp**2)

    diff_x_tp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(x_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_tp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(y_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_trans_to_pix = np.sqrt(diff_x_tp**2 + diff_y_tp**2)
    
    # Input fields
    p['uincDom'] = hank_fun(p['kb'] * distance_trans_to_pix)
    
    # Sensor Green's functions
    sensor_greens_function = hank_fun(p['kb'] * distance_rec_to_pix)
    p['sensorGreensFunction'] = (p['kb']**2) * sensor_greens_function
    
    # Domain Green's functions
    x_green = np.arange(-p['Nx'], p['Nx']) * p['dx']                        
    y_green = np.arange(-p['Ny'], p['Ny']) * p['dy']
    
    # Meshgrid the Green's function pixel locations
    XGreen, YGreen = np.meshgrid(x_green, y_green)
    R = np.sqrt(XGreen**2 + YGreen**2)
    
    # Generate Hankel function and remove singularity
    domain_greens_function = hank_fun(p['kb'] * R)
    
    # Replace the singularity at the center
    # def integrand(x, y):
    #     return np.real(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)))
    def integrand_real(x, y):
            if x == 0 and y == 0:
                return 0.0
            return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).real)
        
    def integrand_imag(x, y):
        if x == 0 and y == 0:
            return 0.0
        return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).imag)
    
    # integral_result, _ = dblquad(integrand, -p['dx']/2, p['dx']/2, 
    #                              lambda x: -p['dy']/2, lambda x: p['dy']/2)
    Ny = p['Ny']
    Nx = p['Nx']
    dx = p['dx']
    dy = p['dy']
    domain_greens_function[Ny, Nx] = dblquad(
            integrand_real,
            -dx/2, dx/2, -dy/2, dy/2
        )[0] / (dx * dy)
    domain_greens_function[Ny, Nx] += (dblquad(
        integrand_imag,
        -dx/2, dx/2, -dy/2, dy/2
    )[0] / (dx * dy)) * 1j
    
    # domain_greens_function[p['Ny'], p['Nx']] = integral_result / (p['dx'] * p['dy'])
    p['domainGreensFunction'] = (p['kb']**2) * domain_greens_function
    
    return p

def construct_parameters(Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, numRec=360, numTrans=60, sensorRadius=1.6,
                         device='cuda'):
    # Initialize parameters
    em = {}

    em['Lx'] = Lx  # [m]
    em['Ly'] = Ly  # [m]

    # Number of pixels
    em['Nx'] = Nx
    em['Ny'] = Ny

    # Smallest distance between objects
    em['dx'] = em['Lx'] / em['Nx']  # [m]
    em['dy'] = em['Ly'] / em['Ny']  # [m]

    # Locations of the pixels
    em['x'] = np.linspace(-em['Nx']/2, em['Nx']/2 - 1, em['Nx']) * em['dx']
    em['y'] = np.linspace(-em['Ny']/2, em['Ny']/2 - 1, em['Ny']) * em['dy']

    # Speed of light [m/s]
    em['c'] = 299792458

    # Wavelength [m]
    em['lambda'] = em['dx'] * wave

    # Measured frequency [GHz]
    em['freq'] = em['c'] / em['lambda'] / 1e9

    # Number of receivers and transmitters
    em['numRec'] = numRec
    em['numTrans'] = numTrans

    # Radius of a rig where sensors are located [m]
    em['sensorRadius'] = sensorRadius

    # Wavenumber [1/m]
    em['kb'] = 2 * np.pi / em['lambda']

    # Generate the EM functions
    em = generate_em_functions(em)
    return torch.from_numpy(em['domainGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['sensorGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['uincDom']).to(device).unsqueeze(-1), torch.from_numpy(em['receiverMask']).unsqueeze(-1)


def full_propagate_to_sensor(f, utot_dom_set, sensor_greens_function_set, dx, dy):
    """
    Propagate all the total fields to the sensors.

    Parameters:
    - f: (Ny x Nx) scattering potential
    - utot_dom_set: (Ny x Nx x numTrans) total field inside the computational domain
    - sensor_greens_function_set: (Ny x Nx x numRec) Green's functions
    - receiver_mask_set: (numTrans x numRec) receiver masks
    - dx, dy: sampling steps

    Returns:
    - uscat_pred_set: (numTrans x numRec x numFreq) predicted scattered fields
    """
    num_trans = utot_dom_set.shape[2]
    num_rec = sensor_greens_function_set.shape[2]
    contSrc = f[0, 0].unsqueeze(-1) * utot_dom_set    # (Ny x Nx x numTrans)
    conjSrc = torch.conj(contSrc).reshape(-1, num_trans)    # (Ny x Nx, numTrans)
    sensor_greens_func = sensor_greens_function_set.reshape(-1, num_rec)    # (Ny x Nx, numRec)
    uscat_pred_set = dx * dy * torch.matmul(conjSrc.T, sensor_greens_func)    # (numTrans, numRec)
    return uscat_pred_set


class InverseScatter(BaseOperator):
    
    def __init__(self, Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, 
                 numRec=360, numTrans=60, sensorRadius=1.6, svd=True, **kwargs):
        super(InverseScatter, self).__init__(**kwargs)
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.numRec = numRec
        self.numTrans = numTrans
        self.domain_greens_function_set, self.sensor_greens_function_set, self.uinc_dom_set, self.receiver_mask_set = \
        construct_parameters(Lx, Ly, Nx, Ny, wave, numRec, numTrans, sensorRadius, self.device)
        
        self.sensor_greens_function_set = self.sensor_greens_function_set.to(torch.complex128)   # (Ny x Nx x numRec)
        self.uinc_dom_set = self.uinc_dom_set.to(torch.complex128)   # (Ny x Nx x numTrans)

        if svd:
            self.compute_svd()

        
    def forward(self, f, unnormalize=True):
        '''
        Parameters:
            f - permittivity, (batch_size, 1, Ny, Nx), torch.Tensor, float32
            
        Returns:
            uscat_pred_set - (batch_size, numTrans, numRec) predicted scattered fields, torch.Tensor, complex64
        '''
        f = f.to(torch.float64)
        if unnormalize:
            f = self.unnormalize(f)
        # Linear inverse scattering
        uscat_pred_set = full_propagate_to_sensor(f, self.uinc_dom_set[..., 0], self.sensor_greens_function_set[..., 0], 
                                                  self.dx, self.dy)
        return uscat_pred_set.unsqueeze(0)
    
    def loss(self, pred, observation):
        '''
        Parameters:
            pred - predicted permittivity, (batch_size, 1, Ny, Nx), torch.Tensor, float32
            observation - actual observation, (1, numTrans, numRec), torch.Tensor, complex64
        Returns:
            loss - data consistency loss, (batch_size,), scalar, float32
        '''
        uscat_pred_set = self.forward(pred)
        diff = uscat_pred_set - observation
        squared_diff = diff * diff.conj()
        loss = torch.sum(squared_diff, dim=(1, 2)).to(torch.float64) # Use torch.float64 for numerical stability
        return loss
        
    def compute_svd(self):
        '''
        Compute SVD of the forward operator A.
        The SVD is computed once and cached for future use.
        A = U @ diag(Sigma) @ V_t
        Also compute A_inv as the pseudo-inverse of A.
        '''
        path = 'cache/inv-scatter_numT_{}_numR_{}'.format(self.numTrans, self.numRec)
        if os.path.exists(path + '/matrix.pt'):
            print('Loading SVD from cache.')
            self.U = torch.load(os.path.join(path, 'U.pt'))
            self.Sigma = torch.load(os.path.join(path, 'S.pt'))
            self.V_t = torch.load(os.path.join(path, 'Vt.pt'))
            self.A = torch.load(os.path.join(path, 'matrix.pt'))
            if os.path.exists(path + '/matrix_inv.pt'):
                self.A_inv = torch.load(os.path.join(path, 'matrix_inv.pt'))
            else:
                self.A_inv = torch.linalg.pinv(self.A)
                torch.save(self.A_inv, os.path.join(path, 'matrix_inv.pt'))
        else:
            print('Computing SVD... This may take 10-20 minutes for the first time.')
            T = self.uinc_dom_set[..., 0].flatten(0,1)
            R = self.sensor_greens_function_set[..., 0].reshape(-1, self.numRec)
            A = torch.cat([R.T@torch.conj(torch.diag(T[:,i])) for i in range(T.shape[-1])], dim=0) * self.dx * self.dy
            A = torch.view_as_real(A).permute(0,2,1).flatten(0,1)
            U, Sigma, V = torch.svd(A)
            self.U = U
            self.Sigma = Sigma
            self.V_t = V.T
            self.A = A
            self.A_inv = torch.linalg.pinv(A)
            os.makedirs(path, exist_ok=True)
            torch.save(self.U, os.path.join(path, 'U.pt'))
            torch.save(self.Sigma, os.path.join(path, 'S.pt'))
            torch.save(self.V_t, os.path.join(path, 'Vt.pt'))
            torch.save(self.A, os.path.join(path, 'matrix.pt'))
            torch.save(self.A_inv, os.path.join(path, 'matrix_inv.pt'))

    def Vt(self, x):
        # Return V^Tx
        # [B, 1, H, W] -> [B, num_Trans*num_Rec]
        x = x.to(torch.float64)
        return (self.V_t @ x.flatten(-3)[...,None]).squeeze(-1)
    
    def V(self, x):
        # Return Vx
        # [B, num_Trans*num_Rec] -> [B, 1, H, W]
        return (self.V_t.T @ x[...,None].to(torch.float64)).reshape(-1, 1, self.Ny, self.Nx).to(torch.float32)
    
    def Ut(self, x):
        # Return U^T x
        # [B, num_Trans,num_Rec] (complex) -> [B, num_Trans*num_Rec]
        x = torch.view_as_real(x)
        return (self.U.T @ x.flatten(-3)[...,None]).squeeze(-1)
    
    def pseudo_inverse(self, x):
        # Return A_inv x
        return self.normalize((self.A_inv @ torch.view_as_real(x).flatten()).reshape(-1, self.Ny, self.Nx))

    @property
    def M(self):
        # Return a mask for nonzero singular values, M = (Sigma > 1e-3)
        return (self.Sigma.abs() > 1e-3).float()
    
    @property
    def S(self):
        # Return the singular values Sigma
        return self.Sigma