#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from clawpack import riemann 
import numpy as np
from clawpack import pyclaw
from scipy.integrate import solve_ivp
import os
import datetime

LAST_PRINT = -1


def getPrimitive(q,state):
    # Get area and problem data:
    gamma = state.problem_data['gamma']

    rho = q[0,:]
    u = q[1,:]/q[0,:]
    z = q[3,:]/q[0,:]
    P = (gamma-1.)*(q[2,:] - 0.5*(q[1,:]**2/q[0,:]))
    T = P/rho
    
    return rho,u,P,T,z

def step_Euler(solver,state,dt):    
    global LAST_PRINT

    # Get parameters:
    gamma = state.problem_data['gamma']
    Da = state.problem_data['Da']
    Ea = state.problem_data['Ea']
    Tign = state.problem_data['Tign']
    Pref = state.problem_data['Pref']
    rhoref = state.problem_data['rhoref']
    AR = state.problem_data['AR']
    hv = state.problem_data['hv']
    s = state.problem_data['s']

    # ODE solved explicitly with 2-stage, 2nd-order Runge-Kutta method.
    dt2 = dt/2.
    q = state.q
    
    # Get state prior to integrating:
    rho,u,P,T,z = getPrimitive(q,state)
    K = np.exp(-Ea*((1.0/T) - (1.0/Tign)))*np.heaviside(T - 1.01,0.0)
    K = 5.0*(np.heaviside(state.t-0.1,0.5) - np.heaviside(state.t-0.5,0.5))*np.heaviside(1.0-state.grid.x.centers,0.0) + np.heaviside(state.t-10.0,0.5)*K
    Ic = np.sqrt(gamma)*(2.0/(gamma+1.0))**((gamma + 1.0)/(2.0*(gamma - 1.0)))
    r = (1.0+(gamma-1.0)/2.0)**(-gamma/(gamma-1.0))
    uref = np.sqrt(Pref/rhoref)
    H = (1.0 - np.heaviside(P-r,0.0)*(P-r)/(1.0-r))*np.heaviside(1.0-P,0.0)
    s = s*np.heaviside(state.t - 10.0,0.0)
    alpha = Ic*np.sqrt(Pref*rhoref)/(rhoref*uref)
    omega = K*rho*(1.0-z)*Da

    qstar = np.empty(q.shape)
    qstar[0,:] = q[0,:] + dt2*alpha*(H*AR - np.sqrt(P*rho))
    qstar[1,:] = q[1,:]
    qstar[2,:] = q[2,:] + dt2*((alpha/(gamma-1.0)*(H*AR - T*np.sqrt(P*rho))) + omega*hv)
    qstar[3,:] = q[3,:] + dt2*(omega - (rho*s*H*z) + alpha*(H*AR - np.sqrt(P*rho))*z)

    # Update primitive variables:
    rho,u,P,T,z = getPrimitive(qstar,state)
    K = np.exp(-Ea*((1.0/T) - (1.0/Tign)))*np.heaviside(T - 1.01,0.0)
    K = 5.0*(np.heaviside(state.t-0.1,0.5) - np.heaviside(state.t-0.5,0.5))*np.heaviside(1.0-state.grid.x.centers,0.0) + np.heaviside(state.t-10.0,0.5)*K
    H = (1.0 - np.heaviside(P-r,0.0)*(P-r)/(1.0-r))*np.heaviside(1.0-P,0.0)
    omega = K*rho*(1.0-z)*Da
    
    
    if(state.t - LAST_PRINT > 0.01):
        print(f" time: {state.t:.2f} P: ({np.min(P):.2f} , {np.max(P):.2f}), T: ({np.min(T):.2f} , {np.max(T):.2f} ) z:( {np.min(z):.2f} , {np.max(z):.2f}) rho: ( {np.min(rho):.2f} , {np.max(rho):.2f}), H : ({np.min(H):.2f} , {np.max(H):.2f}), omega: ({np.min(omega):.2f} , {np.max(omega):.2f})")
        LAST_PRINT = state.t
        
    q[0,:] = q[0,:] + dt*alpha*(H*AR - np.sqrt(P*rho))
    q[1,:] = q[1,:]
    q[2,:] = q[2,:] + dt*((alpha/(gamma-1.0)*(H*AR - T*np.sqrt(P*rho))) + omega*hv)
    q[3,:] = q[3,:] + dt*(omega - (rho*s*H*z) + alpha*(H*AR - np.sqrt(P*rho))*z)

def init(state):
    # Get area and problem data:
    gamma = state.problem_data['gamma']
    L = state.problem_data['L']
    xc = state.grid.x.centers
    
    P = 1.0 + 0.0*xc
    T = 1.0 + 0.0*xc
    z = 0.5*np.sin((2.0*np.pi/L * 6 )*xc)+0.5
    rho = P/T

    state.q[0,:] = rho
    state.q[1,:] = 0.0 
    state.q[2,:] = P/(gamma-1.)
    state.q[3,:] = rho*z

    
# Specify Riemann Solver and instantiate solver object:
rs_HLL = riemann.euler_1D_py.euler_rq1D
rs_HLLC = riemann.euler_1D_py.euler_hllc_rq1D_counterProp
solver = pyclaw.ClawSolver1D(rs_HLLC)
solver.kernel_language = 'Python'
solver.fwave = True

# Set Boundary Conditions:
solver.step_source = step_Euler
solver.bc_lower[0]=pyclaw.BC.periodic # custom inlet
solver.bc_upper[0]=pyclaw.BC.periodic # custom outlet
solver.aux_bc_lower[0]=pyclaw.BC.extrap
solver.aux_bc_upper[0]=pyclaw.BC.extrap

solver.max_steps = 100000
solver.cfl_desired = 0.1

# Working Fluid:
pref = 1.0
rhoref = 1.0
gamma = 1.29
R = 1.0
Tref = 1.0

AR = 0.2

# beta_target = 0.08
# Injector/mixing:
s = 0.07 # 0.07

beta = 1/s 
print(" beta: ", beta)

# Kinetics:
Da = 289  #128
Tign = 5.8 #3.1
hv = 24.6
Ea = 11.5

# Geometry:
L = 24.0
mx = 100 # 4800

# 1-D domain specification:
x = pyclaw.Dimension(0,L,mx,name='x')
domain = pyclaw.Domain([x])
state = pyclaw.State(domain,num_eqn=4,num_aux=4)

# Working fluid:
state.problem_data['gamma'] = gamma
state.problem_data['gamma1'] = gamma-1.0
state.problem_data['Pref'] = pref
state.problem_data['rhoref'] = rhoref
state.problem_data['R'] = R
state.problem_data['Tref'] = Tref

# Injector/mixing:
state.problem_data['s'] = s
state.problem_data['AR'] = AR
state.problem_data['L'] = L

# Kinetics and heat release:
state.problem_data['Da'] = Da
state.problem_data['Tign'] = Tign
state.problem_data['hv'] = hv
state.problem_data['Ea'] = Ea

# Initialize domains, including the aux cells:
init(state)

# plot the initial condition
import matplotlib.pyplot as plt
rho,u,P,T,z = getPrimitive(state.q,state)
# plt.plot(state.grid.x.centers,P)
# plt.plot(state.grid.x.centers,T)
# plt.plot(state.grid.x.centers,z)
# plt.plot(state.grid.x.centers,rho)
# plt.show()
#initDetTube(state)

# Get coordinates of domain:
xc = state.grid.x.centers
# Initialize arrays. These will be added together to form complete area profile.
state.aux[0,:] = np.ones(xc.shape) #area - needs to be ones always
state.aux[1,:] = np.zeros(xc.shape) #tracker for heat release through time 

# Set up PyClaw Controller:
claw = pyclaw.Controller()
claw.tfinal = 180.0
claw.solution = pyclaw.Solution(state,domain)
claw.solver = solver
claw.num_output_times = 200000

# Create a new directory for the run
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join('./_output', timestamp)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

claw.outdir = run_dir
claw.keep_copy = True
claw.write_aux_always = False
claw.output_format = None

# Save parameters to a text file
params = {
    'gamma': gamma,
    'pref': pref,
    'rhoref': rhoref,
    'R': R,
    'Tref': Tref,
    'AR': AR,
    's': s,
    'beta': beta,
    'Da': Da,
    'Tign': Tign,
    'hv': hv,
    'Ea': Ea,
    'L': L,
    'mx': mx,
    'solver.max_steps': solver.max_steps,
    'solver.cfl_desired': solver.cfl_desired,
    'claw.tfinal': claw.tfinal,
    'claw.num_output_times': claw.num_output_times
}

with open(os.path.join(run_dir, 'parameters.txt'), 'w') as f:
    for key, value in params.items():
        f.write(f'{key}: {value}\n')

#q^T = [rho*A, rho*u*A, E*A, rho*Z]
#2 waves for HLL, 3 waves for HLLC
claw.solver.num_eqn = 4
claw.solver.num_waves = 3 

# Run the simulation:
claw.run()


print("finished running claw")

# plot the final time
import matplotlib.pyplot as plt

# Extract solution data over time
times = []
rho_data = []
u_data = []
P_data = []
T_data = []
z_data = []

for frame in claw.frames:
    times.append(frame.t)
    state = frame.state
    rho, u, P, T, z = getPrimitive(state.q, state)
    
    rho_data.append(rho)
    u_data.append(u)
    P_data.append(P)
    T_data.append(T)
    z_data.append(state.aux[2, :])

# Convert to numpy arrays for plotting
times = np.array(times)
rho_data = np.array(rho_data)
u_data = np.array(u_data)
P_data = np.array(P_data)
T_data = np.array(T_data)
z_data = np.array(z_data)

def plot_all_fields():
    # Create side-by-side heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Spatial grid
    x_grid = state.grid.x.centers
    X, T_grid = np.meshgrid(x_grid, times)

    # Helper function to generate safe contour levels
    def safe_levels(data, num_levels=50):
        # Ignore NaNs when computing contour levels. Use nanmin/nanmax so that
        # entirely-NaN arrays are handled gracefully.
        try:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        except Exception:
            # Fallback: if data is a masked array or something else, try converting
            arr = np.asarray(data)
            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)

        # If all values are NaN, create a tiny symmetric range around zero to avoid
        # contour complaining about NaN bounds.
        if np.isnan(vmin) or np.isnan(vmax):
            vmin, vmax = -1e-12, 1e-12

        if vmax - vmin < 1e-12:  # Essentially constant data
            return np.linspace(vmin - 1e-12, vmax + 1e-12, num_levels)
        else:
            return np.linspace(vmin, vmax, num_levels)

    # Plot heatmaps. Mask invalid (NaN/inf) values so contourf doesn't get NaN
    # bounds for levels. Use np.ma.masked_invalid to create a masked array.
    rho_masked = np.ma.masked_invalid(rho_data)
    im1 = axes[0, 0].contourf(
        X,
        T_grid,
        rho_masked,
        levels=safe_levels(rho_masked),
        cmap="viridis",
    )
    axes[0, 0].set_title('Density (ρ)')
    axes[0, 0].set_ylabel('Time')
    plt.colorbar(im1, ax=axes[0, 0])

    u_masked = np.ma.masked_invalid(u_data)
    im2 = axes[0, 1].contourf(
        X,
        T_grid,
        u_masked,
        levels=safe_levels(u_masked),
        cmap="plasma",
    )
    axes[0, 1].set_title('Velocity (u)')
    plt.colorbar(im2, ax=axes[0, 1])

    P_masked = np.ma.masked_invalid(P_data)
    im3 = axes[0, 2].contourf(
        X,
        T_grid,
        P_masked,
        levels=safe_levels(P_masked),
        cmap="inferno",
    )
    axes[0, 2].set_title('Pressure (P)')
    plt.colorbar(im3, ax=axes[0, 2])

    T_masked = np.ma.masked_invalid(T_data)
    im4 = axes[1, 0].contourf(
        X,
        T_grid,
        T_masked,
        levels=safe_levels(T_masked),
        cmap="hot",
    )
    axes[1, 0].set_title('Temperature (T)')
    axes[1, 0].set_xlabel('Position (x)')
    axes[1, 0].set_ylabel('Time')
    plt.colorbar(im4, ax=axes[1, 0])

    z_masked = np.ma.masked_invalid(z_data)
    im5 = axes[1, 1].contourf(
        X,
        T_grid,
        z_masked,
        levels=safe_levels(z_masked),
        cmap="coolwarm",
    )
    axes[1, 1].set_title('Mixture Fraction (z)')
    axes[1, 1].set_xlabel('Position (x)')
    plt.colorbar(im5, ax=axes[1, 1])

    # Remove the last subplot
    axes[1, 2].remove()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'field_evolution_heatmaps.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # save the value of the temperature field at each time and location:
    print("saving temperature field data. shape of the matrix: ", T_data.shape)
    np.save(os.path.join(run_dir, 'Kochs_model_dataset.npy'), T_data)

    # get the last 4 fields
    final_state = claw.frames[-1].state
    rho, u, P, T, z = getPrimitive(final_state.q, final_state)

    print(" F min, max p: ", np.min(P), np.max(P))
    print(" F min, max T: ", np.min(T), np.max(T))
    print(" F min, max z: ", np.min(z), np.max(z))
    print(" F min, max rho: ", np.min(rho), np.max(rho))


def plot_three_fields():
    """Plot temperature, pressure and velocity fields over time."""
    # plot only T, p and u

    # Spatial grid
    x_grid = state.grid.x.centers
    x_mesh, t_grid = np.meshgrid(x_grid, times)

    # Helper function to generate safe contour levels
    def safe_levels(data, num_levels=50):
        # Ignore NaNs when computing contour levels
        try:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        except Exception:
            # Fallback: if data is masked array or something else,
            # try converting
            arr = np.asarray(data)
            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)

        # If all values are NaN, create a tiny symmetric range
        # around zero to avoid contour complaining about NaN bounds
        if np.isnan(vmin) or np.isnan(vmax):
            vmin, vmax = -1e-12, 1e-12

        if vmax - vmin < 1e-12:  # Essentially constant data
            return np.linspace(vmin - 1e-12, vmax + 1e-12, num_levels)
        return np.linspace(vmin, vmax, num_levels)

    _fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    # Plot heatmaps. Mask invalid (NaN/inf) values so contourf
    # doesn't get NaN bounds for levels

    u_masked = np.ma.masked_invalid(u_data)
    im2 = axes[0].contourf(
        x_mesh,
        t_grid,
        u_masked,
        levels=safe_levels(u_masked),
        cmap="viridis",
    )
    axes[0].set_title('Velocity (u)')
    plt.colorbar(im2, ax=axes[0])

    p_masked = np.ma.masked_invalid(P_data)
    im3 = axes[1].contourf(
        x_mesh,
        t_grid,
        p_masked,
        levels=safe_levels(p_masked),
        cmap="ocean",
    )
    axes[1].set_title('Pressure (P)')
    plt.colorbar(im3, ax=axes[1])

    t_masked = np.ma.masked_invalid(T_data)
    im4 = axes[2].contourf(
        x_mesh,
        t_grid,
        t_masked,
        levels=safe_levels(t_masked),
        cmap="plasma",
    )
    axes[2].set_title('Temperature (T)')
    axes[2].set_xlabel('Position (x)')
    axes[2].set_ylabel('Time')
    plt.colorbar(im4, ax=axes[2])

    plt.tight_layout()
    plt.savefig(
        os.path.join(run_dir,
                     'field_evolution_heatmaps_three_fields.png'),
        dpi=300, bbox_inches='tight')
    plt.show()

    # save the value of the temperature field at each time and location:
    t_data_array = np.array(T_data)
    print("saving temperature field data. shape of the matrix: ",
          t_data_array.shape)
    np.save(os.path.join(run_dir, 'Kochs_model_dataset.npy'), T_data)

    # get the last 4 fields
    final_state = claw.frames[-1].state
    rho, u, P, T, z = getPrimitive(final_state.q, final_state)

    print(" F min, max p: ", np.min(P), np.max(P))
    print(" F min, max T: ", np.min(T), np.max(T))
    print(" F min, max z: ", np.min(z), np.max(z))
    print(" F min, max rho: ", np.min(rho), np.max(rho))
    
# plot_all_fields()
plot_three_fields()
np.save('Kochs_model_dataset.npy', T_data)