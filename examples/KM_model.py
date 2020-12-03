import os
import numpy as np
from math import pi
import pylab as plt
from time import time
from os.path import join
from scipy.integrate import odeint


"""
A simple script to calculate order parameter and correlation between 3 nodes
directed edges are : [(0, 1),(1, 2)] 

return:
    order parameter and correlations between pair of nodes in text file and 
    also plot the results.
"""

np.random.seed(1)

data_path = 'data'

if not os.path.exists(data_path):
    os.makedirs(data_path)


def KM_model(theta, t, p):

    K, N, omega, adj = p
    dtheta = np.zeros(len(theta))
    n = len(theta)

    for i in range(n):
        dtheta[i] = omega[i] + K * \
            np.sum(adj[i, :] * np.sin(theta - theta[i]))

    return dtheta


def order_parameter(theta):
    # calculate the order parameter

    r = np.zeros(len(theta))
    for i in range(len(theta)):
        r[i] = abs(sum(np.exp(1j * theta[i]))) / N
    return r
#-------------------------------------------------------------------#


def correlation_matrix(arr):

    nstep, n = arr.shape
    cor = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            cor[i, j] = np.abs(np.sum(np.exp(1j*(arr[:, i]-arr[:, j]))))
            cor[i, j] /= nstep

    return (cor[0, 1], cor[0, 2], cor[1, 2])
#-------------------------------------------------------------------#


def main():
    ind_transition = int(T_trans/dt)
    theta_0 = np.random.uniform(-pi, pi, size=N)

    tspan = np.arange(0.0, T, 0.01)
    p = [K, N, omega_0, adj]

    start = time()

    R = np.zeros(len(K))
    cor = np.zeros((len(K), 3))
    for k in range(len(K)):
        p[0] = K[k]
        theta = odeint(KM_model, theta_0, tspan, args=(p,))
        theta = theta[ind_transition:, :]   # drop transition time
        r = order_parameter(theta)
        R[k] = np.average(r)
        cor[k, :] = correlation_matrix(theta)

        print("K = {:.3f}, R = {:.3f}".format(K[k], R[k]))

    np.savez(join(data_path, "data"), R=R, K=K, cor=cor)
    print("Done in {} seconds".format(time() - start))
#-------------------------------------------------------------------#


def plot_data(filename, plot_cor=True):

    data = np.load(filename)
    R = data['R']
    K = data['K']
    cor = data['cor']
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.plot(K, R, marker="o", lw=1, c="k", label="R")
    if plot_cor:
        ax.plot(K, cor[:, 0], marker="o", label="1->2", color='g', alpha=0.6)
        ax.plot(K, cor[:, 1], marker="o", label="1->3",
                color='purple', alpha=0.6)
        ax.plot(K, cor[:, 2], marker="o", label="2->3",
                color='orange', alpha=0.6)
    ax.legend(loc='lower right')
    ax.set_xlabel("K")
    ax.set_ylabel("Order, Cor")
    fig.savefig(join(data_path, "fig.png"))
    plt.close()


# PARAMETERS --------------------------------------------------

K = np.arange(0.0, 0.4, 0.01)   # coupling
N = 3                           # number of nodes
dt = 0.01                       # time step
T = 200.0                       # simulation time
T_trans = 100.0                 # transition time
mu = 0.0                        # mean value of initial frequencies
omega_0 = [0.3, 0.4, 0.5]
adj = np.asarray([[0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0]])  # suppose input nodes are on the rows


if __name__ == "__main__":

    main()
    plot_data("data/data.npz", plot_cor=True)
