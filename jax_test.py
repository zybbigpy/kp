from jax import grad
import jax.numpy as jnp

def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0


def matrixEigs(a):
    arr = jnp.array([[a, 3*1j], [-3*1j, a]])
    eigs,v = jnp.linalg.eigh(arr)
    return eigs[0]

eigGrad = grad(matrixEigs)
gradientsReal = eigGrad(1.0)

print(gradientsReal)