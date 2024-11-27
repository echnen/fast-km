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


class Operator:

    def __init__(self, prox1, prox2):

        self.prox1 = prox1
        self.prox2 = prox2

    def apply(self, w):

        x1 = self.prox1(w)
        x2 = self.prox2(2 * x1 - w)

        return w + x2 - x1


class Operator_N:

    def __init__(self, tau, prox_list, dim, Lap):

        self.prox_list = prox_list
        self.N = len(prox_list)
        self.dim = dim
        self.tau = tau
        self.Lap = Lap
        self.L = - 2 * np.tril(Lap, k=-1) / np.diag(Lap)[:, np.newaxis]

    def apply(self, wt):

        x = np.zeros((self.N, self.dim))

        for i in range(self.N):
            in_prox = self.L[i, :] @ x + wt[i, :] / self.Lap[i, i]
            x[i, :] = self.prox_list[i](self.tau / self.Lap[i, i], in_prox)

        return wt - self.Lap @ x


def proj_ball(x, center):

    norm = np.linalg.norm(x - center)

    if norm > 1:
        return center + (x - center) / norm

    return x


def soft(w, lm):

    return np.sign(w) * np.maximum(np.abs(w) - lm, 0)


def create_operator(op_type):

    if op_type == 'primal_dual':

        d = 2
        prox_1 = lambda w : soft(w, 0.001)
        center2 = np.ones(2)
        prox_2 = lambda w : (w + proj_ball(w, center2)) / 2
        J = Operator(prox_1, prox_2)
        init = np.zeros(2)

    elif op_type == 'matrix':

        d = 10
        M = np.zeros((d, d))
        M[int(d / 2):, :int(d / 2)] = -np.eye(int(d / 2))
        M[:int(d / 2), int(d / 2):] = np.eye(int(d / 2))
        resolvent = np.linalg.inv(np.eye(d) + 1e-2 * M)
        prox_2 = lambda w: resolvent @ w
        prox_1 = lambda w : w
        J = Operator(prox_1, prox_2)
        init = 10 * np.ones(d)

    return J, init, d
