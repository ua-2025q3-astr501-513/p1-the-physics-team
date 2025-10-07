import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from tqdm.auto import tqdm

def Potential(q, L):
	return -jnp.log(L(q))

dUdq = grad(Potential, argnums=0)

def Kinetic(p, mass):
    minv = jnp.linalg.inv(mass)
    return 0.5 * p @ minv @ p

def Leapfrog(q0, p0, dt, Nsteps, L, Mass):
    q = q0
    p = p0 - 0.5 * dt * dUdq(q, L) # Half-step
    minv = jnp.linalg.inv(Mass)
    
    for _ in range(Nsteps - 1):
        q = q + dt * minv @ p      # Full-step
        p = p - dt * dUdq(q, L)    # Full-step 
    
    q = q + dt * minv @ p          # Final full-step
    p = p - 0.5 * dt * dUdq(q, L)  # Final half-step
    
    return q, -p

def Sampler(q0, dt, Nsteps, L, Mass):
    p0 = jnp.array(np.random.multivariate_normal(np.zeros_like(q0), Mass))

    q, p = Leapfrog(q0, p0, dt, Nsteps, L, Mass)

    Uinit = Potential(q0, L)
    Ufinal = Potential(q, L)
    Kinit = Kinetic(p0, Mass)
    Kfinal = Kinetic(p, Mass)

    if np.random.uniform(0,1) < np.exp(Uinit - Ufinal + Kinit - Kfinal):
        return q
    else:
        return q0
    
def Hmc(q0, Nsamples, dt, Nsteps, L, Mass, burnin=0):
    q_current = q0
    samples = []
    for i in tqdm(range(Nsamples + burnin)):
        q_current = Sampler(q_current, dt, Nsteps, L, Mass)
        samples.append(q_current)
    
    # Remove burn-in samples
    return np.array(samples[burnin:])