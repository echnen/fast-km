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
Run this script to reproduce all numerical experiments in Section 5 of:

R. I. Bot, E. Chenchene, J. M. Fadili.
Generalized Fast Krasnoselskii-Mann Method with Preconditioners,
2024. DOI: 10.48550/arXiv.2411.18574.

Note: This takes from 5 to 10 minutes to complete.
"""

import experiments as exp
import plots as show
import structures as st

if __name__ == "__main__":

    # creating two instances
    J_pd, init_pd, d_pd = st.create_operator('primal_dual')
    J_mt, init_mt, d_mt = st.create_operator('matrix')

    # experiment 1
    Storage_11 = exp.experiment_1(10000, 2, init_pd, J_pd, num_experiment=1)
    Storage_12 = exp.experiment_1(10000, 16, init_pd, J_pd, num_experiment=2)
    Storage_13 = exp.experiment_1(10000, 2, init_mt, J_mt, num_experiment=3)
    Storage_14 = exp.experiment_1(10000, 16, init_mt, J_mt, num_experiment=4)

    show.plot_experiment_1(Storage_11, 10000, 2, num_experiment=1)
    show.plot_experiment_1(Storage_12, 10000, 16, num_experiment=2)
    show.plot_experiment_1(Storage_13, 10000, 2, num_experiment=3)
    show.plot_experiment_1(Storage_14, 10000, 16, num_experiment=4)

    # experiment 2
    Storage_21 = exp.experiment_2(10000, 0.5, init_pd, J_pd, num_experiment=1)
    Storage_22 = exp.experiment_2(10000, 0.1, init_pd, J_pd, num_experiment=2)
    Storage_23 = exp.experiment_2(10000, 0.5, init_mt, J_mt, num_experiment=3)
    Storage_24 = exp.experiment_2(10000, 0.1, init_mt, J_mt, num_experiment=4)

    show.plot_experiment_2(Storage_21, 10000, 0.5, num_experiment=1)
    show.plot_experiment_2(Storage_22, 10000, 0.1, num_experiment=2)
    show.plot_experiment_2(Storage_23, 10000, 0.5, num_experiment=3)
    show.plot_experiment_2(Storage_24, 10000, 0.1, num_experiment=4)

    # experiment 3
    Storage_3 = exp.experiment_3(100000, init_mt, J_mt, num_experiment=1)
    show.plot_experiment_3(Storage_3, 100000, num_experiment=1)

    # experiment optimal transport and median
    Storage_ot = exp.experiment_ot(1000)
    Storage_md = exp.experiment_median(5000)
    show.plot_ot_and_median_experiment(Storage_ot, Storage_md, 1000, 5000)
