import jax.numpy as jnp

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