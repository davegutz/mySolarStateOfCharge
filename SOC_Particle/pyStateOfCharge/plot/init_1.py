# GP_batteryEKF - general purpose battery class for EKF use
# Copyright (C) 2023 Dave Gutz
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# See http://www.fsf.org/licensing/licenses/lgpl.txt for full license text.

""" init_1 does..TODO
Dependencies:
    - numpy      (everything)
    - plq       GP plotter function
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq


def init_1(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())  # init 1
    plt.subplot(221)
    plt.title(plot_title + ' init 1')
    print('init 1', end=':  ')
    plq(plt, sr, 'time', sr, 'reset_s', color='black', linestyle='-')
    plq(plt, smv, 'time', smv, 'reset_s', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'reset', color='magenta', linestyle='-')
    plq(plt, mv, 'time', mv, 'reset', color='cyan', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, sr, 'time', sr, 'Tb_s', color='black', linestyle='-')
    plq(plt, sv, 'time', sv, 'Tb', color='red', linestyle='--')
    plq(plt, mr, 'time_t', mr, 'Tb', color='blue', linestyle='-.', stairs=True)
    plq(plt, mv, 'time', mv, 'Tb', color='green', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, sr, 'time', sr, 'soc_s', color='black', linestyle='-')
    plq(plt, sv, 'time', sv, 'soc', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'soc', color='green', linestyle=':')
    plq(plt, mr, 'time', mr, 'soc_ekf', color='orange', linestyle='None', marker='^', markersize='5', markevery=32)
    plq(plt, mv, 'time', mv, 'soc_ekf', color='cyan', linestyle='None', marker='+', markersize='5', markevery=32)
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
