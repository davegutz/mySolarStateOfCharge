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

""" dom_3 does..TODO
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq


def dom_3(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(221)
    plt.title(plot_title + ' DOM 3')
    print('DOM 3', end=':  ')
    plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='red', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, mr, 'time', mr, 'soc_ekf', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc_ekf', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, mr, 'time', mr, 'soc_s', color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'soc_s', color='red', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'soc_s', color='green', linestyle='-.')
    plq(plt, mv, 'time', mv, 'soc_s', color='orange', linestyle=':')
    plq(plt, mr, 'time', mr, 'soc_ekf', color='cyan', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc_ekf', color='black', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
