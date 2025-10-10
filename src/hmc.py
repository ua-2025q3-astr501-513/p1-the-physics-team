import numpy as np
import jax.numpy as jnp
from jax import grad, random, vmap
from tqdm.notebook import tqdm


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

def Sampler(q0, dt, Nsteps, L, Mass, Massinv, rng_key):
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
    key1, key2 = random.split(rng_key)
    p0 = random.multivariate_normal(key1, jnp.zeros_like(q0), Mass)

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
    accept_prob = jnp.exp(Uinit - Ufinal + Kinit - Kfinal)
    u = random.uniform(key2)

    return jnp.where(u < accept_prob, q, q0)

def Hmc(q0, Nsamples, dt, Nsteps, L, Mass, burnin=0, rng_key=None):
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

    if rng_key is None:
        rng_key = random.PRNGKey(0)

    q_current = q0
    samples = []
    minv = jnp.linalg.inv(Mass) # = M^{-1}
    for _ in tqdm(range(Nsamples + burnin)):
        rng_key, subkey = random.split(rng_key)
        q_current = Sampler(q_current, dt, Nsteps, L, Mass, minv, subkey)
        samples.append(q_current)
    
    # Remove burn-in samples
    return np.array(samples[burnin:])

def Hmc_Vectorized(q0_array, Nsamples, dt, Nsteps, L, Mass, burnin=0, rng_key=None):
    """
    Vectorized Hamiltonian Monte Carlo (HMC) sampling.

    TODO: Add comments later
    """

    if rng_key is None:
        rng_key = random.PRNGKey(0)

    Nchains = q0_array.shape[0]
    minv = jnp.linalg.inv(Mass) # = M^{-1}

    # Vectorize the Sampler function across chains
    sampler_vectorized = vmap(
        lambda q, key: Sampler(q, dt, Nsteps, L, Mass, minv, key),
        in_axes=(0, 0)
    )

    q_current = q0_array
    samples = []

    # Progress bar tracks total samples across all chains
    pbar = tqdm(total=(Nsamples//Nchains) * Nchains, desc="HMC sampling")

    for i in range(Nsamples//Nchains + burnin):
        # Generate one key per chain
        rng_key, *subkeys = random.split(rng_key, Nchains + 1)
        subkeys = jnp.array(subkeys)

        # Update all chains in parallel
        q_current = sampler_vectorized(q_current, subkeys)
        samples.append(q_current)

        if i >= burnin:
            pbar.update(Nchains)
    
    pbar.close()

    # Remove burn-in samples and convert to array
    samples = jnp.array(samples[burnin:])

    # Reshape to (Nsamples, Ndim)
    return samples.reshape(-1, samples.shape[-1])