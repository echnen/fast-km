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
This file contains useful functions to plot our numerical experiments in
Section 5 of:

R. I. Bot, E. Chenchene, J. M. Fadili.
Generalized Fast Krasnoselskii-Mann Method with Preconditioners,
2024. DOI: 10.48550/arXiv.2411.18574.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap


fonts = 30
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
              'size': fonts})
rc('text', usetex=True)


def plot_experiment_1(Storage, maxit, sig, num_experiment=1):
    '''
    Testing various choices of eta
    '''

    # testing the following mus
    Etas = np.linspace(0.01, 0.99, 20)

    # testing the following alphas
    Choices = np.array([[2, 4],
                        [16, 32]])

    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)

    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
                                                           (0, 0, 1),
                                                           (0.5, 0.9, 1)])
    norm = Normalize(0.01, 0.99)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in range(2):
        for j in range(2):

            alpha = Choices[i, j]
            num_plot = 0

            axs[i, j].loglog([1 / (k + 1) for k in range(maxit)],
                             '--', color='k', alpha=0.3)
            axs[i, j].loglog([1 / (k + 1) ** 2 for k in range(maxit)],
                             '--', color='k', alpha=0.3)

            # competitors
            for eta in Etas:
                # retrieving information
                Rs = Storage[(i, j)][num_plot, :]
                color = cmap(norm(eta))
                axs[i, j].loglog(Rs, color=color, alpha=1)
                num_plot += 1

            # baseline
            eta = 0.5
            Rs = Storage[(i, j)][num_plot, :]
            axs[i, j].loglog(Rs, linewidth=3, color='r')
            axs[i, j].set_xlim(1, maxit)

            if (i, j) == (1, 1):
                axs[i, j].set_title(r'$\alpha$ = ' + f'{alpha}')
            else:
                axs[i, j].set_title(r'$\alpha$ = ' + f'{int(alpha)}')

            if i == 1:
                axs[i, j].set_xlabel('Number of iteration')
            if j == 0:
                axs[i, j].set_ylabel('Residual')

    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.01])
    cbar = plt.colorbar(sm, ticks=np.round(Etas[0: -1: 3], 2), anchor=(2, 0),
                        cax=cbar_ax, orientation="horizontal")
    plt.savefig('results/experiment_1_' + f'{num_experiment}.pdf',
                bbox_inches='tight')
    # cbar.ax.set_title(r'$ \eta $', fontsize=fonts)
    # cbar.set_label(r'$ \eta $')
    plt.show()


def plot_experiment_2(Storage, maxit, eta, num_experiment):

    # testing the following mus
    Sigs = np.linspace(1, 100, 20)

    # testing the following alphas
    Choices = np.array([[2, 4],
                        [16, 32]])

    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)

    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
                                                           (0, 0, 1),
                                                           (0.5, 0.9, 1)])
    norm = Normalize(1, 100)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for ScalarMappable

    for i in range(2):
        for j in range(2):

            # counting plots
            num_plot = 0

            alpha = Choices[i, j]

            axs[i, j].loglog([1 / (k + 1) for k in range(maxit)],
                             '--', color='k', alpha=0.3)
            axs[i, j].loglog([1 / (k + 1) ** 2 for k in range(maxit)],
                             '--', color='k', alpha=0.3)

            # competitor
            for case_num, sig in enumerate(Sigs):

                Rs = Storage[(i, j)][num_plot, :]

                color = cmap(norm(sig))
                axs[i, j].loglog(Rs, color=color, alpha=1)
                num_plot += 1

            # baseline
            Rs = Storage[(i, j)][num_plot, :]
            axs[i, j].loglog(Rs, linewidth=3, color='r')

            # prettifying plot
            axs[i, j].set_xlim(1, maxit)

            if (i, j) == (1, 1):
                axs[i, j].set_title(r'$\alpha$ = ' + f'{alpha}')
            else:
                axs[i, j].set_title(r'$\alpha$ = ' + f'{int(alpha)}')

            if i == 1:
                axs[i, j].set_xlabel('Number of iteration')
            if j == 0:
                axs[i, j].set_ylabel('Residual')

    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.01])
    cbar = plt.colorbar(sm, ticks=np.round(Sigs[0:-1:3], 2), anchor=(2.0, 0.0),
                        cax=cbar_ax, orientation="horizontal")
    plt.savefig('results/experiment_2_' + f'{num_experiment}.pdf',
                bbox_inches='tight')
    cbar.ax.set_title(r'$ \sigma $', fontsize=fonts)
    plt.show()


def plot_experiment_3(Storage, maxit, num_experiment):

    # initialize plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                  'size': 20})

    # testing the following alphas
    Choices = np.array([2, 4, 16, 32])

    # initialize plot
    fig, axs = plt.subplots(1, 4, figsize=(22, 5), sharey=True)

    # comparison
    for i in range(4):

        print(f'Case: {(i)}')
        alpha = Choices[i]

        axs[i].loglog([1 / (k + 1) for k in range(maxit)], '--',
                      color='k', alpha=0.3)
        axs[i].loglog([1 / (k + 1) ** 2 for k in range(maxit)], '--',
                      color='k', alpha=0.3)

        # km
        Rs = Storage[(i % 2, i // 2)][0, :]
        axs[i].loglog(Rs, linewidth=3, color='k', label='KM')

        # fast km no cooling
        Rs = Storage[(i % 2, i // 2)][1, :]
        axs[i].loglog(Rs, linewidth=3, color='b', label='fKM')

        # fast km with cooling
        Rs = Storage[(i % 2, i // 2)][2, :]
        axs[i].loglog(Rs, linewidth=3, color='r', label='fKM-LNC')

        # fast km with cooling
        Rs = Storage[(i % 2, i // 2)][3, :]
        axs[i].loglog(Rs, linewidth=3, color='orange', label='fKM-LGC')

        # axs[i, j].grid(which='both')
        axs[i].set_xlim(1, maxit)
        axs[i].set_ylim(1e-70, 1e14)
        axs[i].set_xlabel('Number of iteration')

        if i == 0:
            axs[i].set_title(r'$\alpha$ = ' + f'{alpha}')
            axs[i].legend(bbox_to_anchor=(2.25, -0.2), ncol=4,
                          loc='upper center')

        else:
            axs[i].set_title(r'$\alpha$ = ' + f'{int(alpha)}')

        if i == 0:
            axs[i].set_ylabel('Residual')

    plt.savefig('results/experiment_3_' + f'{num_experiment}.pdf',
                bbox_inches='tight')
    plt.show()

    return


def plot_ot_and_median_experiment(Storage_ot, Storage_md, maxit_ot, maxit_md):

    # initialize plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                  'size': 20})

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    Storages = [Storage_ot, Storage_md]
    maxits = [maxit_ot, maxit_md]

    for i in range(2):

        maxit = maxits[i]
        axs[i].loglog([1 / (k + 1) for k in range(maxit)], '--',
                      color='k', alpha=0.3)
        axs[i].loglog([1 / (k + 1) ** 2 for k in range(maxit)], '--',
                      color='k', alpha=0.3)

        # km
        Rs = Storages[i][0, :]
        axs[i].loglog(Rs, linewidth=3, color='k', alpha=1, label='KM')

        # fast km no cooling
        Rs = Storages[i][1, :]
        axs[i].loglog(Rs, linewidth=3, color='orange', alpha=1,
                      label='fKM, ' + r'$\eta = 0.1$')

        # fast km no cooling
        Rs = Storages[i][2, :]
        axs[i].loglog(Rs, linewidth=3, color='r', alpha=1,
                      label='fKM, ' + r'$\eta = 0.5$')

        # fast km with cooling
        Rs = Storages[i][3, :]
        axs[i].loglog(Rs, linewidth=3, color='b', alpha=1, label='fKM-LNC')

        # prettifying plot
        axs[i].set_xlim(1, maxit)
        axs[i].set_xlabel('Number of iteration')

        if i == 0:
            axs[i].set_title('Optimal Transport Example')
            axs[i].set_ylabel('Residual')

        if i == 1:
            axs[i].set_title('Geometric Median Example')
            axs[i].set_ylim(1e-6, 1e0)
            # axs[i].legend(bbox_to_anchor=(1, 0.75))
            axs[i].legend(bbox_to_anchor=(0.42, -0.19), ncol=2)

    plt.savefig('results/experiment_app.pdf', bbox_inches='tight')
    plt.show()
