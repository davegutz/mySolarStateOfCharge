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

""" General data-over-model
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq

def init_1(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())
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

def init_1a(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []

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

    return fig_list, fig_files

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

def dom_4(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(131)
    plt.title(plot_title + ' DOM 4')
    print('DOM 4', end=':  ')
    plq(plt, mr, 'time', mr, 'soc', color='orange', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='green', linestyle='--')
    plq(plt, sr, 'time', sr, 'soc_s', color='red', linestyle='-.')
    plq(plt, smv, 'time', smv, 'soc_s', color='black', linestyle=':')
    plq(plt, mr, 'time', mr, 'soc_ekf', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc_ekf', color='cyan', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(132)
    plq(plt, mr, 'time', mr, 'vb', color='orange', linestyle='-')
    plq(plt, mr, 'time', mr, 'vb_f', color='orange', linestyle='-', warn=False)
    plq(plt, mr, 'time', mr, 'vb_h', color='cyan', linestyle='--')
    plq(plt, mv, 'time', mv, 'vb', color='green', linestyle='-.')
    plq(plt, mr, 'time', mr, 'vb_s', color='red', linestyle='-.')
    plq(plt, smv, 'time', smv, 'vb_s', color='black', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(133)
    plq(plt, mr, 'soc', mr, 'vb', color='orange', linestyle='-')
    plq(plt, mr, 'soc', mr, 'vb_f', color='orange', linestyle='-', warn=False)
    plq(plt, mr, 'soc', mr, 'vb_h', color='cyan', linestyle='--')
    plq(plt, mv, 'soc', mv, 'vb', color='green', linestyle='-.')
    plq(plt, mr, 'soc_s', mr, 'vb_s', color='red', linestyle='-.')
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='black', linestyle=':')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files

def dom_4a(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(311)
    plt.title(plot_title + ' DOM 4a')
    print('DOM 4a', end=':  ')
    plq(plt, mr, 'time', mr, 'ib', color='orange', linestyle='-')
    plq(plt, mr, 'time', mr, 'ib_f', color='orange', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'ib', color='green', linestyle='--')
    plq(plt, mr, 'time', mr, 'ib_charge', color='red', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_charge_f', color='red', linestyle='-.', warn=False)
    plq(plt, mv, 'time', mv, 'ib_charge', color='black', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(312)
    plq(plt, sr, 'time', sr, 'soc_s', color='orange', linestyle='-')
    plq(plt, smv, 'time', smv, 'soc_s', color='green', linestyle='--')
    plq(plt, mr, 'time', mr, 'soc', color='red', linestyle='-.')
    plq(plt, mv, 'time', mv, 'soc', color='black', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(313)
    plq(plt, mr, 'time', mr, 'Tb_rap', color='red', linestyle='-')
    plq(plt, mv, 'time', mv, 'Tb_rap', color='cyan', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
