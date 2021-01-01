import os
import numpy as np
import pylab as plt
from time import time
import networkx as nx
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


def make_compiled_file(OMP=True):

    I = jitcode(kuramotos_f, n=N, control_pars=[g])
    I.generate_f_C()
    I.compile_C(omp=OMP)
    I.save_compiled(overwrite=True, destination="data/jitced.so")


def simulate(simulation_time,
             transition_time,
             coupling,
             step=1):

    I = jitcode(n=N, module_location="data/jitced.so")
    I.set_parameters(coupling)
    I.set_integrator("dopri5")
    initial_state = uniform(-np.pi, np.pi, N)
    I.set_initial_value(initial_state, 0.0)
    times = np.arange(transition_time, simulation_time, step)
    n_steps = len(times)

    phases = np.empty((n_steps, N))
    for i in range(n_steps):
        phases[i, :] = I.integrate(times[i])

    return times, phases


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
    fig.savefig("data/kuramoto.png")
    plt.close()


N = 15
kind = 2
g = Symbol("g")
omega = normal(0, 0.1, N)
couplings = np.linspace(0, 0.5, 50)
adj = nx.to_numpy_array(nx.complete_graph(N), dtype=int)
simulation_time = 2000.0
transition_time = 500.0
OMP = True

if __name__ == "__main__":

    np.random.seed(1)
    start_time = time()

    R = np.empty(len(couplings))
    make_compiled_file(OMP)

    for i in range(len(couplings)):
        times, phases = simulate(simulation_time,
                                 transition_time,
                                 couplings[i])
        order = order_parameter(phases)
        R[i] = np.mean(order)

    plt.plot(couplings, R, lw=2)
    plt.show()
