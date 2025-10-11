import jax.numpy as jnp

# 2D distribution functions

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

# 1D distribution functions

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

def DoubleGaussian1D(q, mu1=-1.0, mu2=1.0, sigma1=0.5, sigma2=0.5):
    """
    A combination of two 1D Gaussian (normal) probability distributions.
    
    Parameters
    ----------
    q : array-like, shape (1,) or scalar
        Position in 1D space.
    mu1 : float
        Mean of the first distribution.
    mu2 : float
        Mean of the second distribution.
    sigma1 : float
        Standard deviation of the first distribution.
    sigma2 : float
        Standard deviation of the second distribution.
    
    Returns
    -------
    float
        Unnormalized probability density at position q.
    """
    # Extract scalar if q is an array
    x = q[0] if hasattr(q, '__len__') else q
    
    # Gaussian probability density (unnormalized is fine for HMC)
    peak1 = jnp.exp(-0.5 * ((x - mu1) / sigma1)**2)
    peak2 = jnp.exp(-0.5 * ((x - mu2) / sigma2)**2)
    
    return peak1 + peak2

def GammaDist(q):
    x, y = q
    k = 2.0
    theta = 1.0
    if (x <= 0) or (y <= 0):
        return 1e-300  # effectively zero probability
    return (x ** (k - 1)) * jnp.exp(-x / theta) * (y ** (k - 1)) * jnp.exp(-y / theta)

def Chisq(q, k=4.0):
    """
    A chi-squared probability distribution.
    
    Parameters
    ----------
    q : array-like, shape (1,) or scalar
        Position in 1D space.
    k : float
        Degrees of freedom (must be positive).
    
    Returns
    -------
    float
        Unnormalized probability density at position q.
    """
    # Extract scalar if q is an array
    x = q[0] if hasattr(q, '__len__') else q
    
    # Chi-squared is only defined for x > 0
    # Return very small probability if outside support
    prob = jnp.where(
        x > 0,
        jnp.power(x, k/2.0 - 1.0) * jnp.exp(-x / 2.0),
        1e-10  # Small value instead of zero to avoid log(0) in Potential
    )
    
    return prob
