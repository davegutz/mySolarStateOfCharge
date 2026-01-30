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

""" dom_2 does..TODO
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq


def dom_2(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(321)
    plt.title(plot_title + ' DOM 2')
    print('DOM 2', end=':  ')
    plq(plt, mr, 'time', mr, 'dv_dyn', color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='green', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'dv_dyn', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(322)
    plq(plt, mr, 'time', mr, 'voc_stat', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_stat', color='orange', linestyle='--')
    if not strict_overplot:
        plq(plt, mr, 'time', mr, 'voc_stat_f', color='green', linestyle='-.', warn=False)
        plq(plt, mv, 'time', mv, 'voc_stat_f', color='red', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(323)
    plq(plt, mr, 'time', mr, 'voc', color='green', linestyle='-', stairs=True)
    plq(plt, mr, 'time', mr, 'voc_d', color='green', linestyle='-', stairs=True, warn=False)
    plq(plt, mv, 'time', mv, 'voc', color='orange', linestyle='--', stairs=True)
    plq(plt, mr, 'time', mr, 'voc_ekf', color='blue', linestyle='-.', stairs=True)
    plq(plt, mv, 'time', mv, 'voc_ekf', color='red', linestyle=':', stairs=True)
    plt.legend(loc=1)
    plt.subplot(324)
    plq(plt, mr, 'time', mr, 'y_ekf', color='green', linestyle='-', stairs=True)
    plq(plt, mv, 'time', mv, 'y_ekf', color='orange', linestyle='--', stairs=True)
    plq(plt, mr, 'time', mr, 'y_ekf_f', color='blue', linestyle='-.', stairs=True)
    plq(plt, mv, 'time', mv, 'y_filt', color='red', linestyle=':', stairs=True)
    if not strict_overplot:
        plq(plt, mv, 'time', mv, 'y_filt2', color='cyan', linestyle=':', stairs=True)
    plt.legend(loc=1)
    plt.subplot(325)
    plq(plt, mr, 'time', mr, 'dv_hys', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys', color='cyan', linestyle='--')
    if not strict_overplot:
        plq(plt, smv, 'time', smv, 'dv_hys_s', add=0.1, color='red', linestyle='-', warn=False)
        plq(plt, sr, 'time', sr, 'dv_hys_s', add=-0.1, color='magenta', linestyle='-', warn=False)
    plt.legend(loc=1)
    plt.subplot(326)
    plq(plt, mr, 'time_t', mr, 'Tb', color='green', linestyle='-', stairs=True)
    plq(plt, mv, 'time_t', mv, 'Tb', color='orange', linestyle='--', stairs=True, warn=False)
    plq(plt, mv, 'time', mv, 'Tb', color='red', linestyle='-.')
    plq(plt, mr, 'time_t', mr, 'Tb_f', color='green', linestyle='-', stairs=True)
    plq(plt, mv, 'time', mv, 'Tb_f', color='orange', linestyle='--', stairs=True)
    plq(plt, mr, 'time', mr, 'chm', color='black', linestyle='-')
    plq(plt, sr, 'time', sr, 'chm_s', color='cyan', linestyle='--')
    plt.ylim(0., 50.)
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
