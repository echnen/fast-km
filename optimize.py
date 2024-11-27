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
from tabulate import tabulate

def chm(J, mu_f, eps_f, theta_f, maxit, init, v):

    # initialize variables
    wt = np.copy(init)

    # placeholders
    Rs = np.zeros(maxit)

    for k in range(maxit):

        mu = mu_f(k) # (k + 1) ** 2 # 1 * np.log(k + 2)
        eps = eps_f(k) # alpha / (k + beta)
        theta = theta_f(k)

        # actual iteration
        w = J.apply(wt)
        wt = w - (mu - theta) / mu * (w - wt) - theta * eps * (w - v)

        # computing and storing residual
        res = np.sum((w - wt) ** 2)
        Rs[k] = res

    return Rs


def km(J, init, theta_f, maxit):

    # initialize variables
    w = np.copy(init)

    # placeholders
    Rs = np.zeros(maxit)

    for k in range(maxit):

        w_old = np.copy(w)
        w = w + theta_f(k) * (J.apply(w) - w)

        # computing and storing residual
        res = np.sum((w - w_old) ** 2)
        Rs[k] = res


    return Rs


def fkm(J, alpha, eta, sig, init, maxit):

    # initialize variables
    wt = np.copy(init)
    w = np.copy(init)

    # placeholders
    Rs = np.zeros(maxit)

    # scaling parameter
    scale = eta + (1 - eta) * (alpha - 1)

    for k in range(maxit):

        w_old = np.copy(w)

        # actual iteration
        w = J.apply(wt)
        wt = (1 - scale / (k + sig)) * wt + scale / (k + sig) * w\
            + (1 - alpha / (k + sig)) * (w - w_old)

        # computing and storing residual
        res = np.sum((w - wt) ** 2)
        Rs[k] = res

    return Rs


def fkm_plus(J, alpha, eta, sig, init, maxit,
             Cooling=(False, 'linear'),
             Verbose=False):

    # initialize variables
    wt = np.copy(init)
    w = np.copy(init)

    # placeholders
    Rs = np.zeros(maxit)

    # scaling parameter
    alpha_eta = eta + (1 - eta) * (alpha - 1)

    if Cooling[0]:
        alpha_max = alpha * 100
        if Cooling[1] == 'linear':
            alphas = np.linspace(alpha, alpha_max, int(maxit / 2))
        else:
            alphas = np.logspace(np.log10(alpha), np.log10(alpha_max), int(maxit / 2))

    if Verbose:
        print('|| {:>6} | {:>15} ||'.format('Iter.', 'Resd.'))

    for k in range(maxit):

        w_old = np.copy(w)

        # actual iteration
        w = J.apply(wt)
        wt = (1 - alpha_eta / (k + sig)) * wt + alpha_eta / (k + sig) * w\
            + (1 - alpha / (k + sig)) * (w - w_old)

        # computing residual
        res = np.sum((w - wt) ** 2)

        if Cooling[0] and k < int(maxit / 2):
            alpha = alphas[k]
            alpha_eta = eta + (1 - eta) * (alpha - 1)

        if Verbose and int(maxit % (k + 1)) == 10:
            print('|| {:>6} | {:>15} ||'.format(k, np.round(res, 10)))

        # storing residual
        Rs[k] = res

    return Rs
