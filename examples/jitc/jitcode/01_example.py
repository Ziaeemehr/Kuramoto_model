import os
import numpy as np
import pylab as plt
from numpy import pi
from time import time
from jitcode import jitcode, y
from symengine import sin, cos, Symbol
from numpy.random import uniform, normal
from mpl_toolkits.axes_grid1 import make_axes_locatable


if not os.path.exists("data"):
    os.makedirs("data")


def kuramotos_f():
    for i in range(N):
        coupling_sum = 0.0
        if kind == 2:
            coupling_sum = sum(sin(y(j)-y(i))
                               for j in range(N)
                               if adj[i, j])
        else:
            coupling_sum = 0.5 * sum(1-cos(y(j)-y(i))
                                     for j in range(N)
                                     if adj[i, j])
        yield omega[i] + g / N * coupling_sum


def order_parameter(phases):
    # calculate the order parameter

    n = phases.shape[1]
    r = np.zeros(len(phases))
    for i in range(len(phases)):
        r[i] = abs(sum(np.exp(1j * phases[i]))) / n
    return r


def plot_data(times, order):

    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.plot(times, order, lw=2, color='k')
    ax.set_xlabel("times")
    ax.set_ylabel("r(r)")
    ax.set_xlim(0, times[-1])
    ax.set_ylim(0, 1.1)
    fig.savefig("data/kuramoto.png")
    plt.close()


N = 3
g = 1.0
omega = normal(0, 0.1, N)

FB = np.array([[0, 0, 1],
               [1, 0, 0],
               [0, 1, 0]])
adj = FB

if __name__ == "__main__":

    np.random.seed(1)

    start_time = time()
    kind = 2
    initial_state = uniform(-pi, pi, N)
    times = range(0, 201)

    I = jitcode(kuramotos_f, n=N)
    I.set_integrator("dopri5", atol=1e-6, rtol=1e-5)
    I.set_initial_value(initial_state, time=0.0)
    phases = np.empty((len(times), N))

    for i in range(len(times)):
        phases[i, :] = I.integrate(times[i]) % (2*np.pi)

    order = order_parameter(phases)
    plot_data(times, order)
