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
This file contains an implementation of the fast Graph-DRS method to solve the
geometric median problem in Section 5 of:

R. I. Bot, E. Chenchene, J. M. Fadili.
Generalized Fast Krasnoselskii-Mann Method with Preconditioners,
2024. DOI: 10.48550/arXiv.2411.18574.

"""

import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse as sp
from PIL import Image
import structures as st
import optimize as opt


def create_sparse_gradx_mat(p):
    '''
    Creates a sparse matrix that computes the partial derivative w.r.t. x.
    '''

    diag = np.ones(p)
    diag[-1] = 0
    diag = np.tile(diag, p)

    Dx = sp.spdiags([-diag, [0] + list(diag[:-1])], [0, 1], p ** 2, p ** 2)

    return Dx


def create_sparse_grady_mat(p):
    '''
    Creates a sparse matrix that computes the partial derivative w.r.t. y.
    '''

    diag = np.ones(p ** 2)
    diag[-p:] = 0 * diag[-p:]

    up_diag = np.ones(p ** 2)
    up_diag[:p] = 0 * up_diag[:p]

    Dy = sp.spdiags([-diag, up_diag], [0, p], p ** 2, p ** 2)

    return Dy


def grad(psi, Dx, Dy, n):
    '''
    Computes the gradient of an image psi.
    '''

    sig_out = np.zeros((n, 2))

    sig_out[:, 0] = Dx @ psi
    sig_out[:, 1] = Dy @ psi

    return sig_out


def div(sig, M1, M2):
    '''
    Computes the divergence of a vector field sig.
    '''

    return M1 @ sig[:, 0] + M2 @ sig[:, 1]


def proj_inf_2(sig, n):
    '''
    Computes the projection onto the ell 2 norm for vector fields.
    '''

    sig_out = np.copy(sig)
    Norm = np.linalg.norm(sig_out, axis=1)
    Greater = Norm > 1
    sig_out[Greater] = np.divide(sig_out[Greater],
                                 np.transpose(np.asmatrix(Norm[Greater])))

    return sig_out


def prox_l1(tau, sig, n):
    '''
    Computes the proximity operator of the group Lasso penalty.
    '''

    return sig - tau * proj_inf_2(sig / tau, n)


def proj_div(tau, sig, mu, nu, M1, M2, Dx, Dy, Lap, n):
    '''
    Computes the projection onto zero divergence contraints.
    '''

    return sig - grad(spl.spsolve(Lap, div(sig, M1, M2) + mu - nu), Dx, Dy, n)


def read_image(img1, p):
    '''
    Reads an image.
    '''

    img_brg = Image.open(img1).convert('L')
    Img = 255 - np.array(img_brg.resize((p, p)))
    Img = np.reshape(Img, p ** 2)

    return Img


def read_measures(img1, img2, p):
    '''
    Turns images into probability measures.
    '''

    mu = read_image(img1, p)
    mu = mu / np.sum(mu)
    nu = read_image(img2, p)
    nu = nu / np.sum(nu)

    return mu, nu


def optimization_ot(tau, p, n, mu, nu, M1, M2, Dx, Dy, Lap, maxit):

    # choosing parameters
    alpha = 16
    eta = 0.1
    t0 = 16

    # setting up storage
    Storage = np.zeros((4, maxit))

    # define DRS operator
    prox_1 = lambda w: proj_div(tau, w, mu, nu, M1, M2, Dx, Dy, Lap, n)
    prox_2 = lambda w: prox_l1(tau, w, n)
    J = st.Operator(prox_1, prox_2)

    # initialization
    w = np.zeros((n, 2))

    alpha = 16

    # km
    Rs = opt.km(J, w, lambda k: 1, maxit)
    Storage[0, :] = Rs

    # fast km no cooling
    Rs = opt.fkm_plus(J, alpha, eta, t0, w, maxit, Cooling=(False, None),
                      Verbose=False)
    Storage[1, :] = Rs

    # fast km no cooling
    Rs = opt.fkm_plus(J, alpha, 0.5, t0, w, maxit, Cooling=(False, None),
                      Verbose=False)
    Storage[2, :] = Rs

    # fast km with cooling
    Rs = opt.fkm_plus(J, 2, eta, t0, w, maxit, Cooling=(True, 'linear'),
                      Verbose=False)
    Storage[3, :] = Rs

    return Storage
