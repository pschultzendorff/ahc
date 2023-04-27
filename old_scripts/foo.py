import tpf_lab.numerics.ad.functions as af
import numpy as np
from porepy.numerics.ad import Ad_array
import scipy.sparse as sps

val = np.array([1, 2, 0])
J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
a = Ad_array(val, sps.csc_matrix(J))
b = af.pow(a, -3)
c = np.diag(np.power(val, 4))
jac = (-3 / c) @ J

pass
