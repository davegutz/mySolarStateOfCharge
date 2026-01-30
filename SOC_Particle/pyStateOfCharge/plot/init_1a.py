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

""" init_1a does..TODO
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq


def init_1a(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(221)
    plt.title(plot_title + ' 1a')
    print('1a', end=':  ')
    if hasattr(mr, 'mod_data') and mr.mod_data[0] != 0 and strict_overplot:
        plq(plt, mr, 'time', mr, 'ib_amp_model', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib_amp_model', color='red', linestyle='-.')
        plq(plt, mr, 'time', mr, 'ib_noa_model', color='green', linestyle='--')
        plq(plt, mv, 'time', mv, 'ib_noa_model', color='blue', linestyle=':')
    else:
        plq(plt, mr, 'time', mr, 'ib_amp_hdwe', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib_amp_hdwe', color='red', linestyle='-.')
        plq(plt, mr, 'time', mr, 'ib_noa_hdwe', color='green', linestyle='--')
        plq(plt, mv, 'time', mv, 'ib_noa_hdwe', color='blue', linestyle=':')
    if not strict_overplot:
        plq(plt, mr, 'time', mr, 'ib_sel', add=+0, color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib_sel', add=+0, color='black', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_charge', add=+0, color='orange', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ib_charge', add=+0, color='blue', linestyle=':')
    plt.legend(loc=1)
    if not strict_overplot and hasattr(mr, 'ib_sel_stat'):
        plt.subplot(222)
        plq(plt, mr, 'time', mr, 'ib_sel_stat', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib_sel_stat', color='red', linestyle='--', warn=False)
        plq(plt, mr, 'time', mr, 'ib_dec', add=2, color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib_dec', add=2, color='red', linestyle='--', warn=False)
        plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-', stairs=True)
    plq(plt, mv, 'time', mv, 'e_wrap', color='red', linestyle='--', stairs=True)
    plq(plt, mr, 'time', mr, 'e_wrap_filt', color='black', linestyle='-.', stairs=True)
    plq(plt, mv, 'time', mv, 'e_wrap_filt', color='red', linestyle=':', stairs=True)
    plq(plt, mr, 'time', mr, 'y_ekf', slr=-1, color='green', linestyle='-.', stairs=True)
    plq(plt, mv, 'time', mv, 'y_ekf', slr=-1, color='orange', linestyle=':', stairs=True)
    plq(plt, mr, 'time', mr, 'y_ekf_f', slr=-1, color='black', linestyle='-.', stairs=True)
    plq(plt, mv, 'time', mv, 'y_filt', slr=-1, color='red', linestyle=':', stairs=True)
    plt.legend(loc=1)
    if not strict_overplot:
        plt.subplot(224)
        plq(plt, mr, 'time', mr, 'cc_dif', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'cc_dif', color='red', linestyle='--', warn=False)
        plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")
