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
#      2024. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.
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
This file contains an implementation of the graph DRS method to find the
geometric median of N points in Rd

R. I. Bot, E. Chenchene, J. M. Fadili.
Generalized Fast Krasnoselskii-Mann Method with Preconditioners,
2024. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import numpy as np
import structures as st
import matplotlib.pyplot as plt
import optimize as opt


def proj_ball(radius, w):

    norm = np.linalg.norm(w)

    if norm > radius:
        return w / norm * radius
    else:
        return w


def soft(tau, w):
    '''
    proximity operator of | w |
    '''

    return w - proj_ball(tau, w)


def prox_abs_tilted(tau, w, x):
    '''
    proximity operator of | w - x |
    '''

    return x + soft(tau, w - x)


def optimize_median(maxit, init, J, sig, eta):

    # initialize plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)
    plt.rcParams.update({'font.size': 22})

    # storing objects
    Storage = np.zeros((4, maxit))

    # comparison
    alpha = 16

    plt.loglog([1 / (k + 1) for k in range(maxit)],
                     '--', color='k', alpha=0.3)
    plt.loglog([1 / (k + 1) ** 2 for k in range(maxit)],
                     '--', color='k', alpha=0.3)

    # km
    Rs = opt.km(J, init, lambda k : 1, maxit)
    Storage[0, :] = Rs

    # fast km no cooling
    Rs = opt.fkm_plus(J, alpha, eta, sig, init, maxit, Cooling=(False, None),
                      Verbose=False)
    Storage[1, :] = Rs

    # fast km no cooling
    Rs = opt.fkm_plus(J, alpha, 0.5, sig, init, maxit, Cooling=(False, None),
                      Verbose=False)
    Storage[2, :] = Rs

    # fast km with cooling
    Rs = opt.fkm_plus(J, 2, eta, sig, init, maxit, Cooling=(True, 'linear'),
                      Verbose=False)
    Storage[3, :] = Rs

    return Storage
