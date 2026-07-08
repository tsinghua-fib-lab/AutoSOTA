import math
import torch
import torch.fft as fft
from .base import BaseOperator
from typing import Dict


# Solve: w_t = - u . grad(w) + (1/Re)*Lap(w) + f
#       u = (psi_y, -psi_x)
#       -Lap(psi) = w
# Note: Adaptive time-step takes smallest step across the batch
class NavierStokes2d(object):
    def __init__(self, s1, s2, 
                 L1=2*math.pi, L2=2*math.pi,
                 Re=100.0, 
                 device=None, dtype=torch.float64):
        '''
        pseudo-spectral solver for 2D Navier-Stokes equation
        Args:
            - s1, s2: spatial resolution
            - L1, L2: spatial domain
            - Re: Reynolds number
            - device: device to run the solver
            - dtype: data type
        '''
        
        self.s1 = s1
        self.s2 = s2

        self.L1 = L1
        self.L2 = L2

        self.Re = Re

        self.h = 1.0/max(s1, s2)

        #Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.zeros((1,)),\
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)


        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        #Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        #Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        #Dealiasing mask using 2/3 rule
        self.dealias = (k1**2 + k2**2 <= (s1/3)**2 + (s2/3)**2).type(dtype).to(device)
        #Ensure mean zero
        self.dealias[0,0] = 0.0

    #Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h, real_space=False):
        #-Lap(psi) = w
        psi_h = self.inv_lap*w_h

        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    #Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f, real_space=True):
        #Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f

        #Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f

        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    #Compute non-linear term + forcing from given vorticity (Fourier space)
    def nonlinear_term(self, w_h, f_h=None):
        #Dealias vorticity
        dealias_w_h = w_h*self.dealias

        #Physical space vorticity
        w = fft.irfft2(dealias_w_h, s=(self.s1, self.s2))

        #Physical space velocity
        q, v = self.velocity_field(self.stream_function(dealias_w_h, real_space=False), real_space=True)

        #Compute non-linear term in Fourier space
        nonlin = -1j*((2*math.pi/self.L1)*self.k1*fft.rfft2(q*w) + (2*math.pi/self.L1)*self.k2*fft.rfft2(v*w))

        #Add forcing function
        if f_h is not None:
            nonlin += f_h

        return nonlin
    
    def time_step(self, q, v, f, Re):
        #Maxixum speed
        max_speed = torch.max(torch.sqrt(q**2 + v**2)).item()

        #Maximum force amplitude
        if f is not None:
            xi = torch.sqrt(torch.max(torch.abs(f))).item()
        else:
            xi = 1.0
        
        #Viscosity
        mu = (1.0/Re)*xi*((self.L1/(2*math.pi))**(3.0/4.0))*(((self.L2/(2*math.pi))**(3.0/4.0)))

        if max_speed == 0:
            return 0.5*(self.h**2)/mu
        
        #Time step based on CFL condition
        return min(0.5*self.h/max_speed, 0.5*(self.h**2)/mu)

    def solve(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):

        #Rescale Laplacian by Reynolds number
        GG = (1.0/Re)*self.G

        #Move to Fourier space
        w_h = fft.rfft2(w)

        if f is not None:
            f_h = fft.rfft2(f)
        else:
            f_h = None
        
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re)

        time  = 0.0
        #Advance solution in Fourier space
        while time < T:
            if time + delta_t > T:
                current_delta_t = T - time
            else:
                current_delta_t = delta_t

            #Inner-step of Heun's method
            nonlin1 = self.nonlinear_term(w_h, f_h)
            w_h_tilde = (w_h + current_delta_t*(nonlin1 - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #Cranck-Nicholson + Heun update
            nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
            w_h = (w_h + current_delta_t*(0.5*(nonlin1 + nonlin2) - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #Update time
            time += current_delta_t

            #New time step
            if adaptive:
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                delta_t = self.time_step(q, v, f, Re)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.solve(w, f, T, Re, adaptive, delta_t)


'''
Forward operator for 2D Navier-Stokes equation
'''
class ForwardNavierStokes2d(BaseOperator):
    def __init__(self, 
                 resolution=128, L=2 * math.pi,
                 forward_time=1.0,
                 Re=200.0, 
                 downsample_factor=2,
                 dtype=torch.float32,
                 delta_t=1e-2,
                 adaptive=True, 
                 **kwargs):
        super(ForwardNavierStokes2d, self).__init__(**kwargs)
        self.dtype = dtype
        self.solver = NavierStokes2d(resolution, resolution, L, L, device=self.device, dtype=dtype)
        self.force = self.get_forcing(resolution, L)

        self.downsample_factor = downsample_factor
        self.forward_time = forward_time
        self.Re = Re
        self.delta_t = delta_t
        self.adaptive = adaptive

    def get_forcing(self, resolution, L):
        t = torch.linspace(0, L, resolution+1, 
                           device=self.device, dtype=self.dtype)[0:-1]
        _, y = torch.meshgrid(t, t, indexing='ij')
        return - 4 * torch.cos(4.0 * y)

    @torch.no_grad()
    def __call__(self, data: Dict[str, torch.Tensor], 
                 unnormalize=True):
        '''
        Args:
            - data: dictionary containing the initial vorticity field in the key 'target'
            - unnormalize: whether to unnormalize the input data
        Returns:
            - sol: solution velocity field of shape (batch_size, resolution, resolution), torch.float32
        '''
        # Solve for the velocity field
        x = data['target']
        sol = self.forward(x, unnormalize)
        # Add noise
        sol += self.sigma_noise * torch.randn_like(sol)
        return sol
    
    @torch.no_grad()
    def forward(self, 
                x: torch.Tensor, 
                unnormalize=True):
        '''
        Args:
            - u: velocity field of shape (batch_size, 1, resolution, resolution)

        Returns:
            - u: solution velocity field of shape (batch_size, resolution, resolution), torch.float32
        '''
        # Solve for the velocity field
        if unnormalize:
            raw_u = self.unnormalize(x)
        else:
            raw_u = x

        sol = self.solver.solve(raw_u.squeeze(1), self.force, self.forward_time, self.Re, adaptive=self.adaptive, delta_t=self.delta_t)
        # Downsample the velocity field
        sol = sol[..., ::self.downsample_factor, ::self.downsample_factor]
        return sol.unsqueeze(1).to(torch.float32)
