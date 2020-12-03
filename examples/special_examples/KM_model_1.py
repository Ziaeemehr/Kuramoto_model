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


def main(trial=0, filename="data"):
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

        print("trial = {:03d}, K = {:.3f}, R = {:.3f}".format(
            trial, K[k], R[k]))

    np.savez(join(data_path, filename), R=R, K=K, cor=cor)
    print("Done in {} seconds".format(time() - start))
#-------------------------------------------------------------------#


def plot_data(filename, plot_cor=True, labels=['1-2', '1-3', '2-3'], fig_name="fig"):

    data = np.load(filename)
    R = data['R']
    K = data['K']
    cor = data['cor']
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.plot(K, R, marker="o", lw=1, c="k", label="R")
    if plot_cor:
        colors = ['g', 'purple', 'orange']
        for i in range(3):
            ax.plot(K, cor[:, i], marker="o", label="cor-" +
                    labels[i], color=colors[i], alpha=0.5)

    ax.set_xlabel("K")
    ax.legend(loc='lower right')
    ax.set_ylabel("Order, Cor")
    fig.savefig(join(data_path, fig_name), dpi=150)
    plt.close()


if __name__ == "__main__":

    # np.random.seed(1)

    # PARAMETERS --------------------------------------------------
    N = 3                           # number of nodes
    dt = 0.01                       # time step
    T = 200.0                       # simulation time
    T_trans = 100.0                 # transition time
    K = np.arange(0.0, 0.4, 0.01)   # coupling

    # case 1 :
    omega_i = 0.1
    omega_0 = [0, omega_i, omega_i]
    adj = np.asarray([[0, 0, 0],
                      [1, 0, 0],
                      [1, 1, 0]])  # suppose input nodes are on the rows

    main(filename="data_1")
    plot_data(join(data_path, "data_1.npz"),
              plot_cor=True, fig_name="case_1.png")

    # case 2 :
    omega_i = 0.1
    omega_0 = [omega_i, 0, omega_i]
    adj = np.asarray([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])  # suppose input nodes are on the rows

    main(filename="data_2")
    plot_data(join(data_path, "data_2.npz"),
              plot_cor=True, fig_name="case_2.png")
