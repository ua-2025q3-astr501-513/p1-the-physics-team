import numpy as np
import jax.numpy as jnp
from jax import grad
from tqdm.notebook import tqdm
from joblib import Parallel, delayed


def Potential(q, L):
    """
    Compute the potential energy U(q) = -ln(L(q)) in JAX function form.
    Parameters
    ----------
    q : array-like
        Position, in parameter space.
    L : callable
        Function of a probability distribution P(q) related to Hamiltonian H(q, p).
        This is what we want to sample from. We hoped to use "P" for parameter
        name, but this will cause ambiguity with the Potential name.
    Returns
    -------
    float
        Negative log-potential energy at position q.
    """
    return -jnp.log(L(q))



def Kinetic(p, minv):
    """
    Compute the kinetic energy K(p) = 0.5 * (p^T M^{-1} p) in JAX function form.

    Parameters
    ----------
    p : array-like
        Momentum vector.
    mass : array-like
        Mass matrix. We already considered higher-dimensional momentum 
		vectors. So that the mass is also a matrix, where only the diagonal 
		entries are non-zero.

    Returns
    -------
    float
        Kinetic energy corresponding to momentum p.
    """
    return 0.5 * p @ minv @ p   # = 0.5 * (p^T M^{-1} p)


def Leapfrog(q0, p0, dt, Nsteps, L, Massinv):
    """
    A leapfrog integrator solving for Hamiltonian (H) in the kick-drift-kick scheme.
	This was introduced in Lecture 9 (Mon, Sep 29, 2025):
	https://ua-2025q3-astr501-513.github.io/notes-9/#leapfrog-verlet-integrator

    Parameters
    ----------
    q0 : array-like
        Initial position.
    p0 : array-like
        Initial momentum.
    dt : float
        Time size for every step.
    Nsteps : int
        Number of leapfrog total integration steps.
    L : callable
		Function of a probability distribution P(q) related to Hamiltonian H(q, p).
    Mass : array-like
        Mass matrix.

    Returns
    -------
    (array, array)
        (Position, momentum) tuple (q, p) giving the new position and momentum after integration.
    """
    q = q0
    dUdq = grad(Potential, argnums=0)
	
    # Half-step momentum update
    p = p0 - 0.5 * dt * dUdq(q, L) # Half-step
    
    # Full steps
    for _ in range(Nsteps - 1):
        q = q + dt * Massinv @ p      # Full-step
        p = p - dt * dUdq(q, L)    # Full-step 
    
    # Final position and half momentum update
    q = q + dt * Massinv @ p          # Final full-step
    p = p - 0.5 * dt * dUdq(q, L)  # Final half-step
    
    return q, -p  # TODO: Why negative?


def Sampler(q0, dt, Nsteps, L, Mass, Massinv):
    """
    HMC sampler using leapfrog integrator.

    Parameters
    ----------
    q0 : array-like
        Initial position.
    dt : float
        Time size for every step.
    Nsteps : int
        Number of leapfrog total integration steps.
    L : callable
        Likelihood function.
    Mass : array-like
        Mass matrix.

    Returns
    -------
    array-like
        New sample position after Metropolis acceptance test.
    """
    # Draw a random momentum vector from the Normal distribution: p ~ N(0, Mass)
    p0 = jnp.array(np.random.multivariate_normal(np.zeros_like(q0), Mass))

	# Compute new (q, p) after given N steps from the leapfrog integration
    q, p = Leapfrog(q0, p0, dt, Nsteps, L, Massinv)

    # Compute initial and final energies
	# Reason: In fact, in numerical calculation, we cannot compute the true path of (q, p)
	#         with the constant Hamiltonian/energy. Check what the difference is below. 
    Uinit  = Potential(q0, L)
    Ufinal = Potential(q, L)
    Kinit  = Kinetic(p0, Massinv)
    Kfinal = Kinetic(p, Massinv)

    # Metropolis acceptance criterion
    # Reason: If ideally, our computed (q_new, p_new) has the same energy, we are
    #         very happy to accept this (q_new, p_new). Otherwise (also in most cases), 
    #         we still accept it but with a likelihood of ~ min(1, e^{-ΔH}).
    if np.random.uniform(0,1) < np.exp(Uinit - Ufinal + Kinit - Kfinal):
        return q  # Accept
    else:
        return q0 # Reject


def Hmc(q0, Nsamples, dt, Nsteps, L, Mass, burnin=0):
    """
    Main Hamiltonian Monte Carlo (HMC) sampling.

    Parameters
    ----------
    q0 : array-like
        Initial position (a parameter vector).
    Nsamples : int
        Number of samples.
    dt : float
        Time size for every leapfrog integration step.
    Nsteps : int
        Number of leapfrog steps per sample.
    L : callable
        Likelihood distribution function of position/parameter.
    Mass : array-like
        Mass matrix.
    burnin : int, optional
        Number of initial samples to discard. Default is 0.

    Returns
    -------
    np.ndarray
        Array of accepted samples after burn-in.
    """
    q_current = q0
    samples = []
    minv = jnp.linalg.inv(Mass) # = M^{-1}
    for i in tqdm(range(Nsamples + burnin)):
        q_current = Sampler(q_current, dt, Nsteps, L, Mass, minv)
        samples.append(q_current)
    
    # Remove burn-in samples
    return np.array(samples[burnin:])

def Hmc_parallel(q0, Nsamples, dt, Nsteps, L, Mass, burnin=0, Nwalkers=16):
    """
    Main Hamiltonian Monte Carlo (HMC) sampling.

    Parameters
    ----------
    q0 : array-like
        Initial position (a parameter vector).
    Nsamples : int
        Number of samples.
    dt : float
        Time size for every leapfrog integration step.
    Nsteps : int
        Number of leapfrog steps per sample.
    L : callable
        Likelihood distribution function of position/parameter.
    Mass : array-like
        Mass matrix.
    burnin : int, optional
        Number of initial samples to discard. Default is 0.

    Returns
    -------
    np.ndarray
        Array of accepted samples after burn-in.
    """
        
    minv = jnp.linalg.inv(Mass) # = M^{-1}
    def run_chain(q0, Nsamples, dt, Nsteps, L, Mass, minv, burnin, show_pbar=False, total_samples=None):
        q_current = q0
        samples = []
        
        pbar = tqdm(total=total_samples) if show_pbar else None
        
        for _ in range(Nsamples + burnin):
            q_current = Sampler(q_current, dt, Nsteps, L, Mass, minv)
            samples.append(q_current)
            if pbar is not None:
                pbar.update(Nwalkers)
        
        if pbar is not None:
            pbar.close()
        
        return np.array(samples[burnin:])

    total_samples = (Nsamples//Nwalkers + burnin) * Nwalkers

    results = Parallel(n_jobs=Nwalkers) \
        (delayed(run_chain)(q0, Nsamples//Nwalkers, dt, Nsteps, L, Mass, minv, burnin, 
                            show_pbar=(i==0), total_samples=total_samples) \
        for i in range(Nwalkers))

    samples = np.concatenate(results)

    # Remove burn-in samples
    return samples