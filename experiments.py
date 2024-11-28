# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 Radu I. Bot (radu.bot@univie.ac.at)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#                       Jalal M. Fadili (jalal.fadili@ensicaen.fr)
#
#    This file is part of the example code repository for the paper:
#
#      R. I. Bot, E. Chenchene, J. M. Fadili.
#      Generalized Fast Krasnoselskii-Mann Method with Preconditioners,
#      2024. DOI: 10.48550/arXiv.2411.18574.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains our numerical experiments in Section 5 of:

R. I. Bot, E. Chenchene, J. M. Fadili.
Generalized Fast Krasnoselskii-Mann Method with Preconditioners,
2024. DOI: 10.48550/arXiv.2411.18574.

"""

import numpy as np
import optimize as opt
import structures as st
import experiment_ot as ot
import experiment_median as md


def experiment_1(maxit, sig, init, J, num_experiment=1):
    '''
    Implements the experiment in Section 5.2.1.
    '''

    # testing the following mus
    Etas = np.linspace(0.01, 0.99, 20)

    # testing the following alphas
    Choices = np.array([[2, 4],
                        [16, 32]])

    # storage
    Storage = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

    for i in range(2):
        for j in range(2):
            print(f'Case: {(i, j)}')

            alpha = Choices[i, j]

            # competitors
            for case_num, eta in enumerate(Etas):

                Rs = opt.fkm(J, alpha, eta, sig, init, maxit)
                if case_num == 0:
                    Storage[(i, j)] = Rs
                else:
                    Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

            # baseline
            eta = 0.5
            Rs = opt.fkm(J, alpha, eta, sig, init, maxit)
            Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

    return Storage


def experiment_2(maxit, eta, init, J, num_experiment):
    '''
    Implements the experiment in Section 5.2.2.
    '''

    # testing the following mus
    Sigs = np.linspace(1, 100, 20)

    # testing the following alphas
    Choices = np.array([[2, 4],
                        [16, 32]])

    # storage
    Storage = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

    for i in range(2):
        for j in range(2):

            alpha = Choices[i, j]
            alpha_eta = eta + (1 - eta) * (alpha - 1)

            # competitor
            for case_num, sig in enumerate(Sigs):

                Rs = opt.fkm(J, alpha, eta, sig, init, maxit)

                if case_num == 0:
                    Storage[(i, j)] = Rs
                else:
                    Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

            # baseline
            Rs = opt.fkm(J, alpha, eta, alpha_eta, init, maxit)
            Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

    return Storage


def experiment_3(maxit, init, J, num_experiment):
    '''
    Implements the experiment in Section 5.3.
    '''

    # choosing parameters
    eta = 1 / 2
    sig = 2

    # testing the following alphas
    Choices = np.array([[2, 4],
                        [16, 32]])

    # storage
    Storage = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

    # comparison
    for i in range(2):
        for j in range(2):

            print(f'Case: {(i, j)}')
            alpha = Choices[i, j]

            # km
            Rs = opt.km(J, init, lambda k: 1, maxit)
            Storage[(i, j)] = Rs

            # fast km no cooling
            Rs = opt.fkm_plus(J, alpha, eta, sig, init, maxit,
                              Cooling=(False, None), Verbose=False)
            Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

            # fast km with cooling
            Rs = opt.fkm_plus(J, 2, eta, sig, init, maxit,
                              Cooling=(True, 'linear'), Verbose=False)
            Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

            # fast km with cooling
            Rs = opt.fkm_plus(J, 2, eta, sig, init, maxit,
                              Cooling=(True, 'log'), Verbose=False)
            Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

            # axs[i, j].grid(which='both')
            Storage[(i, j)] = np.vstack((Storage[(i, j)], Rs))

    return Storage


def experiment_ot(maxit):
    '''
    Implements the experiment in Section 5.4.1.
    '''

    p = 100

    [mu, nu] = ot.read_measures("data/mu.png", "data/nu.png", p)

    n = len(mu)
    p = int(np.sqrt(n))

    tau = 1e-1

    Dy = ot.create_sparse_gradx_mat(p)
    Dx = ot.create_sparse_grady_mat(p)

    M1 = -Dx.T
    M2 = -Dy.T
    Lap = -(Dx.T @ Dx + Dy.T @ Dy)

    Storage = ot.optimization_ot(tau, p, n, mu, nu, M1, M2, Dx, Dy, Lap, maxit)

    return Storage


def experiment_median(maxit):
    '''
    Implements the experiment in Section 5.4.2.
    '''

    # generating sample
    dim = 2
    N = 65
    tau = 0.1
    np.random.seed(1)
    S = np.random.normal(0, 100, size=(dim, N))

    # defining DRS operator
    # Lap = N * np.eye(N) - np.ones(N)
    Z = np.random.rand(N, N - 1)
    mean = np.mean(Z, axis=0)
    Z = Z - mean[np.newaxis, :]
    Lap = Z @ Z.T

    prox_list = [lambda tau, w: md.prox_abs_tilted(tau, w, s) for s in S.T]
    J = st.Operator_N(tau, prox_list, dim, Lap)

    # choosing parameters
    eta = 0.1
    sig = 16

    init = np.zeros((N, dim))

    Storage = md.optimize_median(maxit, init, J, sig, eta)

    return Storage
