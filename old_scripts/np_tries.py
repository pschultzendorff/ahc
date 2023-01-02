import numpy as np
import scipy.sparse as sps

exponent: float = 3
power: np.ndarray = np.array([0, 1, 2, 3])
out = np.divide(
    exponent,
    power,
    out=np.zeros_like(power),
    where=power != 0,
    casting="unsafe",
)
print(out)
print(power)
print(type(power))

exponent: float = 3.0
power: np.ndarray = np.power([2, 4], 4)
power = np.array(
    [
        16,
        256,
        3,
        3,
        10,
        5,
        4,
    ],
    dtype=np.float64,
)
out = np.divide(
    exponent,
    power,
    out=np.zeros_like(power),
    where=power != 0,
    casting="unsafe",
    dtype=np.float64,
)
print(out)
print(power)
print(type(power))
print(power != 0)
print(1 / power)

a = np.array([1, 2, 4, 0])
print(1 / a)

print(np.divide(3, 4))

a = np.array([1, 2, 4, 0])
print(1 / a)

print(np.divide(3, a, dtype=np.float64))

print(1 / np.power(3, 3))


A = sps.diags([5, 6])
jac = np.array([[1, 2], [3, 4]])
print(np.array([A * J for J in jac]))
print([J for J in jac])
print(A @ jac)
print(A @ jac.T)
print(A * jac[0])
print(A.todense())
print(jac)
print(A.todense() * jac)


B = np.array([[1, 2], [0, 5]])
C = np.array([[3, 5], [10, 0]])
print(B * C)
print(B @ C)

A2 = np.diag([5, 6])
print(A2 * jac)
print(A2 * jac[0])
print(A @ jac)
print(A2 @ jac)
print(jac * A)
print(np.array([A * J for J in jac]))
print(np.array([J * A for J in jac]))
print(A.A * jac)
print(A2 @ jac)
print(A * jac)

print(np.dot(A.A, jac))
jac2 = sps.diags([1, 2])
print((A * jac2).todense())
jac_sps = sps.csc_matrix(jac)
print((jac * A))
print(A * jac)
