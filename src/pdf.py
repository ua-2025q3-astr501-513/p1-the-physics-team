import jax.numpy as jnp
from jax.scipy.special import gamma

def Gaussian(q):
    #cov = jnp.array([[1, 0.5], [0.5, 1]])
    covinv = jnp.array([[4./3., -2./3.],[-2./3., 4./3.]])
    return jnp.exp(-0.5 * q @ covinv @ q)

def Banana(q):
    a = 1.0
    b = 100.0
    x, y = q[0], q[1]
    return jnp.exp(-((a - x)**2 + b * (y - x**2)**2) / 200.0)

def sumGaussian(q):
    #covs = jnp.eye(2)
    covinv = jnp.eye(2)
    a = jnp.array([3, 3])
    b = jnp.array([-3, 0])
    c = jnp.array([-5, 6])
    return jnp.exp(-0.5 * (q-a) @ covinv @ (q-a)) + jnp.exp(-0.5 * (q-b) @ covinv @ (q-b)) + jnp.exp(-0.5 * (q-c) @ covinv @ (q-c))

def Donut(q, R=2.0, r=0.5):
    """
    A donut (torus) shaped 2D probability distribution.
    
    Parameters
    ----------
    q : array-like, shape (2,)
        Position [x, y] in 2D space.
    R : float
        Radius from origin to center of donut tube (major radius).
    r : float
        Radius of the donut tube itself (minor radius).
    
    Returns
    -------
    float
        Unnormalized probability density at position q.
    """
    x, y = q[0], q[1]
    
    # Distance from origin
    distance_from_origin = jnp.sqrt(x**2 + y**2)
    
    # Distance from the donut's center ring
    distance_from_ring = jnp.abs(distance_from_origin - R)
    
    # Gaussian falloff from the ring
    # Higher density near the ring (distance_from_ring ≈ 0)
    # Lower density away from the ring
    sigma = r
    prob = jnp.exp(-0.5 * (distance_from_ring / sigma)**2)
    
    return prob

def Lognormal(q, mu=0.0, sigma=1.0, shift=0.0):
    """
    A shifted lognormal probability distribution.
    
    Parameters
    ----------
    q : array-like, shape (1,) or scalar
        Position in 1D space.
    mu : float
        Mean of the underlying normal distribution (log-scale location).
    sigma : float
        Standard deviation of the underlying normal distribution (log-scale).
    shift : float
        Shift parameter. The distribution is defined for q > shift.
        Effectively shifts the support of the lognormal.
    
    Returns
    -------
    float
        Unnormalized probability density at position q.
    """
    # Extract scalar if q is an array
    x = q[0] if hasattr(q, '__len__') else q
    
    # Shifted value
    y = x + shift
    
    # Lognormal is only defined for y > 0
    # Return very small probability if outside support
    prob = jnp.where(
        y > 0,
        (1.0 / y) * jnp.exp(-0.5 * ((jnp.log(y) - mu) / sigma)**2),
        1e-10  # Small value instead of zero to avoid log(0) in Potential
    )
    
    return prob

def Gaussian1D(q, mu=0.0, sigma=1.0):
    """
    A 1D Gaussian (normal) probability distribution.
    
    Parameters
    ----------
    q : array-like, shape (1,) or scalar
        Position in 1D space.
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation of the distribution.
    
    Returns
    -------
    float
        Unnormalized probability density at position q.
    """
    # Extract scalar if q is an array
    x = q[0] if hasattr(q, '__len__') else q
    
    # Gaussian probability density (unnormalized is fine for HMC)
    prob = jnp.exp(-0.5 * ((x - mu) / sigma)**2)
    
    return prob

def Dirichlet(q):
    x = q
    a = jnp.random.uniform(shape=x.shape, minval=0.1, maxval=3.)

    a0 = jnp.sum(a)
    prod = jnp.prod(x**(a-1))
    B = jnp.prod(gamma(a))/gamma(a0)

    return prod/B