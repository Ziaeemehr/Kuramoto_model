import os
import warnings
import numpy as np
import pylab as plt
from time import time
import networkx as nx
from os.path import join
from scipy.stats import sem
from scipy.stats.morestats import Std_dev
from symengine import sin, cos, Symbol
from jitcode import jitcode_lyap, y
from joblib import Parallel, delayed
from numpy.random import normal, uniform
from jitcxde_common import DEFAULT_COMPILE_ARGS

if not os.path.exists("data"):
    os.makedirs("data")
# -------------------------------------------------------------------

os.environ["CC"] = "clang"


def display_time(time, message):
    ''' 
    show real time elapsed
    '''
    hour = int(time//3600)
    minute = int(time % 3600) // 60
    second = time - (3600.0 * hour + 60.0 * minute)
    print("{:s} done in {:d} hours {:d} minutes {:.4f} seconds".format(
          message, hour, minute, second))
# -------------------------------------------------------------------


def plot_data(label, coupling, alpha=1, verbose=False):

    filename = "{:s}-{:.4f}".format(label, coupling)
    data = np.load(join("data", filename + ".npz"))
    t = data['t']
    le = data['LE']

    # num_p = len(np.where(le[-1, :] > 0)[0])
    # num_n = len(np.where(le[-1, :] < 0)[0])
    color_p = plt.cm.Reds(np.linspace(0, 1, 2*n_lyap+1))
    color_n = plt.cm.Blues(np.linspace(0, 1, 2*n_lyap+1))
    counter_p = n_lyap//2
    counter_n = n_lyap//2

    fig, ax = plt.subplots(1, figsize=(6, 4))

    for j in range(n_lyap):
        if (le[-1, j] > 0):
            ax.loglog(t,
                      np.abs(le[:, j]),
                      c=color_p[counter_p],
                      alpha=alpha,
                      lw=2,
                      )
            counter_p += 1
        else:
            ax.loglog(t,
                      np.abs(le[:, j]),
                      c=color_n[counter_n],
                      alpha=alpha,
                      lw=2,
                      )
            counter_n += 1

    ax.set_xlabel("time")
    ax.set_ylabel(r"$\lambda_i$")
    plt.tight_layout()
    fig.savefig('data/{:s}.png'.format(filename))

    if verbose:
        for i in range(n_lyap):
            print("{:18.9f}".format(le[-1, i]))
# -------------------------------------------------------------------


def compiling(N, n_lyap, label, adj, USE_OMP=False):

    g = Symbol("g")
    cfilename = "data/jitced_{:s}.so".format(label)

    def kuramotos_f_II():
        for i in range(N):
            coupling_sum = sum(sin(y(j)-y(i))
                               for j in range(N)
                               if adj[i, j])
            yield omega[i] + g * coupling_sum

    start_time = time()
    I = jitcode_lyap(kuramotos_f_II,
                     n=N,
                     n_lyap=n_lyap,
                     simplify=False,
                     control_pars=[g])

    I.generate_f_C(chunk_size=chunk_size)
    I.compile_C(extra_compile_args=DEFAULT_COMPILE_ARGS+["-O0"],
                omp=USE_OMP,
                verbose=False)
    I.save_compiled(overwrite=True, destination=cfilename)
    display_time(time() - start_time, "(compile)")


def simulate_lyap(coupling, label):

    # ---------------------------------------------------------------
    initial_state = uniform(-np.pi, np.pi, N)
    start_time = time()
    cfilename = "data/jitced_{:s}.so".format(label)

    if not os.path.exists(cfilename):
        print("no compiled file found!")
        exit(0)

    else:
        I = jitcode_lyap(n=N,
                         n_lyap=n_lyap,
                         module_location=cfilename)

    I.set_parameters(coupling)
    I.set_integrator("RK45")
    I.set_initial_value(initial_state, 0.0)

    times = np.arange(0, simulation_time, step)
    nstep = len(times)
    lyaps = np.zeros((nstep, n_lyap))

    # run -----------------------------------------------------------
    start_time = time()
    for iter in range(len(times)):
        lyaps[iter, :] = I.integrate(times[iter])[1]
    display_time(time() - start_time, "(run)    ")
    # ---------------------------------------------------------------

    np.savez(join("data", "{:s}-{:.4f}".format(label, coupling)),
             LE=lyaps,
             t=times,
             le=np.mean(lyaps[-500:, :], axis=0))

    # printing the Lyapunov exponents
    for i in range(n_lyap):
        lyap = np.average(lyaps[100:, i])
        stderr = sem(lyaps[100:, i])
        print("%4d. Lyapunov exponent: % .4f +- %.4f" % (i+1, lyap, stderr))
# -------------------------------------------------------------------


def run_command(args):
    coupling, label = args
    simulate_lyap(coupling, label)
# -------------------------------------------------------------------


def batch_run():
    args = []
    for label in net.keys():
        for coupling in couplings:
            args.append([coupling, label])
    Parallel(n_jobs=n_jobs)(map(delayed(run_command), args))


if __name__ == "__main__":

    np.random.seed(1)

    # PARAMETERS ----------------------------------------------------
    chunk_size = 100
    USE_OMP = False
    n_jobs = 2

    N = 3
    n_lyap = 2
    simulation_time = 1000.0
    transition_time = 200.0
    step = 0.5
    couplings = np.arange(0.01, 0.05, 0.01)

    path_networks = f"networks/"
    FF = np.asarray([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0]])
    FB = np.asarray([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])
    omega = np.random.normal(0, 0.1, size=N)

    g = Symbol("g")
    net = {"FF": FF,
           "FB": FB}

    # compile serial
    for label in net.keys():
        adj = net[label]
        cfilename = "data/jitced_{:s}.so".format(label)
        if not os.path.exists(cfilename):
            compiling(N, n_lyap, label, adj, USE_OMP)

    batch_run()

    # plot serial
    for label in net.keys():
        for coupling in couplings:
            plot_data(label, coupling)
