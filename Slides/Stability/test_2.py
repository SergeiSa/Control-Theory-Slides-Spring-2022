import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

import matplotlib.pyplot as pp


A1 = np.array([[-0.1, -2], [2, -0.1]])
h1, _ = np.linalg.eig(A1)
print(h1)

A2 = np.array([[-0.1, -2], [-2, -0.1]])
h2, _ = np.linalg.eig(A2)
print(h2)

A3 = np.array([[-2.1, -1], [-1, -2.1]])
h3, _ = np.linalg.eig(A3)
print(h3)

A4 = np.array([[-2.0, -1], [-1, 0]])
h4, _ = np.linalg.eig(A4)
print(h4)



fig, ((ax1, ax2), (ax3, ax4)) = pp.subplots(nrows=2, ncols=2)

def solve_and_plot(A, ax):
    # h, _ = np.linalg.eig(A)
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

    ax.quiver(qu_X, qu_Y, qu_U, qu_V, qu_C)

    def dvdt_fnc(t, v):
        return A @ v
    v0  = np.array((0.8, 0.7))
    tf = 30
    Count2 = 500
    times = np.linspace(0, tf, num=Count2)    
    vv = solve_ivp(dvdt_fnc, (0, tf), v0, t_eval=times)

    v = vv["y"]
    t = vv["t"]
    # print(v.shape)
    for i in range(Count2):
        if (abs(v[0, i]) > 1) or (abs(v[1, i]) > 1):
            v[0, i] = np.NaN
            v[1, i] = np.NaN
    print(v)
    ax.plot(v[0, :], v[1, :])

solve_and_plot(A1, ax1)
solve_and_plot(A2, ax2)
solve_and_plot(A3, ax3)
solve_and_plot(A4, ax4)
pp.show()