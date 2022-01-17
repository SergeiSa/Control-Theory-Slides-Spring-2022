import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

import matplotlib.pyplot as pp

alpha = np.random.randn() - 0.2
beta  = np.random.randn() + 1

A = np.array([[alpha, -beta], [beta, alpha]])
h, _ = np.linalg.eig(A)

print(h)
print(alpha, beta)

lims = np.array([-1, 1])
Count = 30
qu_X = np.zeros((Count, Count))
qu_Y = np.zeros((Count, Count))
qu_U = np.zeros((Count, Count))
qu_V = np.zeros((Count, Count))
qu_C = np.zeros((Count, Count))
v  = np.zeros((2, ))
for i in range(Count):
    for j in range(Count):
        v[0] = lims[0] + i * (lims[1] - lims[0]) / Count
        v[1] = lims[0] + j * (lims[1] - lims[0]) / Count
        dvdt = A @ v
        qu_X[i, j] = v[0]
        qu_Y[i, j] = v[1]
        qu_U[i, j] = dvdt[0]
        qu_V[i, j] = dvdt[1]
        qu_C[i, j] = np.linalg.norm(v)

v0  = np.array((0.8, 0.8))
def dvdt_fnc(t, v):
    return A @ v
tf = 10
times = np.linspace(0, tf, num=150)    
vv = solve_ivp(dvdt_fnc, (0, tf), v0, t_eval=times)

v = vv["y"]
t = vv["t"]

pp.quiver(qu_X, qu_Y, qu_U, qu_V, qu_C)
pp.plot(v[0, :], v[1, :])

pp.show()