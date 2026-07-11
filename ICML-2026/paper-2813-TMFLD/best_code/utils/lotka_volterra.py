from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, Tsit5, PIDController
from diffrax import ControlTerm, Euler, MultiTerm, VirtualBrownianTree, ReversibleHeun, SEA, HalfSolver, ItoMilstein
import jax
import jax.numpy as jnp


def lvfunc(t, y, args):
    u, v = y
    c, d = 0.4, 0.02
    a = jax.nn.sigmoid(args[0])
    b = a * jax.nn.sigmoid(args[1])
    
    du = a * u - b * u * v
    dv = -c * v + d * u * v
    nugget = 1e-10
    return jnp.array([du, dv]) + nugget

def lotka_volterra_ws(y0, params, rng_key, end=60, noise_scale=1.0):
    tobs = 100.0
    ts = jnp.linspace(0.0, tobs, 100)
        
    term = ODETerm(lvfunc)
    solver = Tsit5()
    saveat = SaveAt(ts=ts)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    sol = diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=0.05, y0=y0, saveat=saveat, args=params, stepsize_controller=stepsize_controller)
    noise = noise_scale * jax.random.normal(rng_key, sol.ys.shape)
    return (sol.ys + noise)[:end, :]

def lotka_volterra_ms(y0, params, rng_key, end=60, noise_scale=1.0):
    tobs = 100.0
    ts = jnp.linspace(0.0, tobs, 100)
    rng_key, key = jax.random.split(rng_key)
    diff_coef = jnp.diag(jnp.array([0.2, 0.2]))
    diffusion = lambda t, y, args: diff_coef

    brownian_motion = VirtualBrownianTree(t0=ts[0], t1=ts[-1], tol=1e-3, shape=(2,), key=rng_key)
    terms = MultiTerm(ODETerm(lvfunc), ControlTerm(diffusion, brownian_motion))
    solver = ItoMilstein() #HalfSolver(SEA()) #Euler()
    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(terms, solver, t0=ts[0], t1=ts[-1], dt0=0.1, y0=y0, saveat=saveat, args=params)
    rng_key, key = jax.random.split(rng_key)
    noise = noise_scale * jax.random.normal(rng_key, sol.ys.shape)
    return (sol.ys + noise)[:end, :]
