import jax.numpy as jnp

def Gaussian(q):
    cov = jnp.array([[1, 0.5], [0.5, 1]])
    covinv = jnp.linalg.inv(cov)
    return jnp.exp(-0.5 * q @ covinv @ q)

def Banana(q):
    a = 1.0
    b = 100.0
    x, y = q[0], q[1]
    return jnp.exp(-((a - x)**2 + b * (y - x**2)**2) / 200.0)