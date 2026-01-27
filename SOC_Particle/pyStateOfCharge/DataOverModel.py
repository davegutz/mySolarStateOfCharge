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

"""Define a general purpose battery model including Randles' model and SoC-VOV model as well as Kalman filtering
support for simplified Mathworks' tracker. See Huria, Ceraolo, Gazzarri, & Jackey, 2013 Simplified Extended Kalman
Filter Observer for SOC Estimation of Commercial Power-Oriented LFP Lithium Battery Cells.
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Battery import Battery, overall_batt
from myFilters import LagExp
# below suppresses runtime error display******************
# import os
# os.environ["KIVY_NO_CONSOLELOG"] = "1"
# from kivy.utils import platform  # failed experiment to run BLE data plotting realtime on android
# if platform != 'linux':
#     from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files
from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files
import Chemistry_BMS
from Colors import Colors
import re
from local_paths import version_from_data_file, local_paths
import os
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})

def plq(plt_, sx, st, sy, yt, slr=1, add=0., color='black', linestyle='-', label=None, marker=None,
        markersize=None, markevery=None, stairs=False, warn=True, linewidth=None):
    if not label:
        slr_str = ''
        if slr != 1.:
            slr_str = str(slr) + '*'
        add_str = ''
        if add > 0:
            add_str = '+' + str(add)
        elif add < 0:
            add_str = str(add)
        if sy.str:
            label = slr_str + yt + '_' + sy.str + add_str
        else:
            label = slr_str + yt + add_str
    if (sx is not None and sy is not None and hasattr(sx, st) and hasattr(sy, yt) and getattr(sy, yt) is not None and
            len(getattr(sy, yt)) > 0 and getattr(sy, yt)[0] is not None):
        try:
            yscld = getattr(sy, yt) * slr + add
        except TypeError:
            yscld = np.array(getattr(sy, yt)) * slr + add
        try:
            if stairs:
                try:
                    dt = getattr(sx, st)[-1] - getattr(sx, st)[-2]
                except IndexError:
                    if warn:
                        print(f"plq: skipping     {yt}({st})     labeled  '{label}'  Dimensions of time different")
                    return
                x_in = np.append(getattr(sx, st), getattr(sx, st)[-1]+dt)
                plt_.stairs(yscld, x_in, color=color, linestyle=linestyle, label=label)
            else:
                plt_.plot(getattr(sx, st), yscld, color=color, linestyle=linestyle, label=label, marker=marker,
                          markersize=markersize, markevery=markevery, linewidth=linewidth)
        except ValueError:
            if warn:
                print(f"plq: skipping     {yt}({st})     labeled  '{label}'")
    else:
        if warn:
            print(f"plq: skipping     {yt}({st})     labeled  '{label}'")

def dom_plot(mr, mv, sr, sv, smv, filename, fig_files=None, plot_title=None, fig_list=None, plot_init_in=False,
             run_str='_run', ver_str='_ver', strict_overplot=False):
    print('dom_plot', end=':  ')
    if fig_files is None:
        fig_files = []

    if plot_init_in and hasattr(smv, 'time') and hasattr(sr, 'time'):
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

    fig_list.append(plt.figure())  # 1a
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

    fig_list.append(plt.figure())  # DOM 2
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

    fig_list.append(plt.figure())  # DOM 3
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

    fig_list.append(plt.figure())  # DOM 4
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

    fig_list.append(plt.figure())  # DOM 4a
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

    # fig_list.append(plt.figure())  # DOM 5
    # plt.subplot(231)
    # plt.title(plot_title + ' DOM 5')
    # print('DOM 5', end=':  ')
    # plq(plt, mr, 'time', mr,'ib_charge, color='black', linestyle='-', label='ib_charge' + run_str)
    # plq(plt, mv, 'time', mv,'ib_charge, linestyle='--', color='blue', label='ib_charge' + ver_str)
    # plq(plt, mr, 'time', mr,'ib_diff_flt + 2, color='green', linestyle='-', label='ib_diff_flt' + run_str + '+2')
    # plq(plt, mv, 'time', mv, 'ib_diff_flt', add=+2, color='red', linestyle='--', label='ib_diff_flt' + ver_str + '+2')
    # plq(plt, mr, 'time', mr,'ib_diff_fa + 2, color='magenta', linestyle='-', label='ib_diff_fa' + run_str + '+2')
    # plq(plt, mv, 'time', mv, 'ib_diff_fa', add=+2, color='cyan', linestyle='--', label='ib_diff_fa' + ver_str + '+2')
    # plt.legend(loc=1)
    # plt.subplot(232)
    # plq(plt, mr, 'time', mr, 'wrap_hi_flt', add=+12, color='green', linestyle='-', label='wrap_hi_flt' + run_str + '+12')
    # plq(plt, mv, 'time', mv, 'wrap_hi_flt', add=+12, color='red', linestyle='--', label='wrap_hi_flt' + ver_str + '+12')
    # plq(plt, mr, 'time', mr, 'wrap_hi_m_flt', add=+10, color='green', linestyle='-', label='wrap_hi_m_flt' + run_str + '+10')
    # plq(plt, mv, 'time', mv, 'wrap_hi_m_flt', add=+10, color='red', linestyle='--', label='wrap_hi_m_flt' + ver_str + '+10')
    # plq(plt, mr, 'time', mr, 'wrap_hi_n_flt', add=+8, color='green', linestyle='-', label='wrap_hi_n_flt' + run_str + '+8')
    # plq(plt, mv, 'time', mv, 'wrap_hi_n_flt', add=+8, color='red', linestyle='--', label='wrap_hi_n_flt' + ver_str + '+8')
    # plq(plt, mr, 'time', mr, 'wrap_lo_flt', add=+6, color='green', linestyle='-', label='wrap_lo_flt' + run_str + '+6')
    # plq(plt, mv, 'time', mv, 'wrap_lo_flt', add=+6, color='red', linestyle='--', label='wrap_lo_flt' + ver_str + '+6')
    # plq(plt, mr, 'time', mr, 'wrap_lo_m_flt', add=+4, color='green', linestyle='-', label='wrap_lo_m_flt' + run_str + '+4')
    # plq(plt, mv, 'time', mv, 'wrap_lo_m_flt', add=+4, color='red', linestyle='--', label='wrap_lo_m_flt' + ver_str + '+4')
    # plq(plt, mr, 'time', mr, 'wrap_lo_n_flt', add=+2, color='green', linestyle='-', label='wrap_lo_n_flt' + run_str + '+2')
    # plq(plt, mv, 'time', mv, 'wrap_lo_n_flt', add=+2, color='red', linestyle='--', label='wrap_lo_n_flt' + ver_str + '+2')
    # plt.legend(loc=1)
    # plt.subplot(233)
    # plq(plt, mr, 'time', mr, 'e_wrap', color='magenta', linestyle='--', label='e_wrap' + run_str)
    # plq(plt, mr, 'time', mr, 'e_wrap_filt', color='black', linestyle='-', label='e_wrap_filt' + run_str)
    # plq(plt, mv, 'time', mv, 'e_wrap', color='red', linestyle='-.', label='e_wrap' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_filt', color='blue', linestyle='--', label='e_wrap_filt' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_m', color='cyan', linestyle='-.', label='e_wrap_m' + ver_str)
    # plq(plt, mr, 'time', mv, 'e_wrap_m_filt', color='blue', linestyle='-', label='e_wrap_m_filt' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_n', color='cyan', linestyle='-.', label='e_wrap_n' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_n_filt', color='green', linestyle='--', label='e_wrap_n_filt' + ver_str)
    # plq(plt, mr, 'time', mr, 'cc_dif', color='green', linestyle='-', label='cc_diff'+run_str)
    # plq(plt, mv, 'time', mv, 'cc_dif', color='red', linestyle='--', label='cc_diff'+ver_str)
    # plt.ylim(-1, 1)
    # plt.legend(loc=1)
    # plt.subplot(234)
    # plq(plt, mr, 'time', mr, 'ib_sel_stat', add=-2, color='black', linestyle='-', label='ib_sel_stat' + run_str + '-2')
    # plq(plt, mv, 'time', mv, 'ib_sel_stat', add=-2, color='blue', linestyle='--', label='ib_sel_stat' + ver_str + '-2')
    # plq(plt, mr, 'time', mr, 'tb_flt', color='green', linestyle='-', label='tb_flt' + run_str)
    # plq(plt, mv, 'time', mv, 'tb_flt', color='red', linestyle='--', label='tb_flt' + ver_str)
    # plq(plt, mr, 'time', mr, 'tb_fa', color='magenta', linestyle='-.', label='tb_fa' + run_str)
    # plq(plt, mv, 'time', mv, 'tb_fa', color='cyan', linestyle=':', label='tb_fa' + ver_str)
    # plq(plt, mr, 'time', mr, 'vb_sel', add=+2, color='magenta', linestyle='-', label='vb_sel_stat' + run_str + '+2')
    # plq(plt, mv, 'time', mv, 'vb_sel', add=+2, color='cyan', linestyle='--', label='vb_sel_stat' + ver_str + '+2')
    # plq(plt, mr, 'time', mr, 'tb_sel', add=+6, color='green', linestyle='-', label='tb_sel_stat' + run_str + '+6')
    # plq(plt, mv, 'time', mv, 'tb_sel', add=+6, color='red', linestyle='--', label='tb_sel_stat' + ver_str + '+6')
    # plt.legend(loc=1)
    # plt.subplot(235)
    # plq(plt, mr, 'time', mr, 'wrap_hi_fa', add=+12, color='green', linestyle='-', label='wrap_hi_fa' + run_str + '+12')
    # plq(plt, mv, 'time', mv, 'wrap_hi_fa', add=+12, color='red', linestyle='--', label='wrap_hi_fa' + ver_str + '+12')
    # plq(plt, mr, 'time', mr, 'wrap_hi_m_fa', add=+10, color='green', linestyle='-', label='wrap_hi_m_fa' + run_str + '+10')
    # plq(plt, mv, 'time', mv, 'wrap_hi_m_fa', add=+10, color='red', linestyle='--', label='wrap_hi_m_fa' + ver_str + '+10')
    # plq(plt, mr, 'time', mr, 'wrap_hi_n_fa', add=+8, color='green', linestyle='-', label='wrap_hi_n_fa' + run_str + '+8')
    # plq(plt, mv, 'time', mv, 'wrap_hi_n_fa', add=+8, color='red', linestyle='--', label='wrap_hi_n_fa' + ver_str + '+8')
    # plq(plt, mr, 'time', mr, 'wrap_lo_fa', add=+6, color='green', linestyle='-', label='wrap_lo_fa' + run_str + '+6')
    # plq(plt, mv, 'time', mv, 'wrap_lo_fa', add=+6, color='red', linestyle='--', label='wrap_lo_fa' + ver_str + '+6')
    # plq(plt, mr, 'time', mr, 'wrap_lo_m_fa', add=+4, color='green', linestyle='-', label='wrap_lo_fa' + run_str + '+4')
    # plq(plt, mv, 'time', mv, 'wrap_lo_m_fa', add=+4, color='red', linestyle='--', label='wrap_lo_m_fa' + ver_str + '+4')
    # plq(plt, mv, 'time', mr, 'wrap_lo_n_fa', add=+2, color='green', linestyle='-.', label='wrap_lo_fa' + run_str + '+2')
    # plq(plt, mv, 'time', mv, 'wrap_lo_n_fa', add=+2, color='red', linestyle='-.', label='wrap_lo_n_fa' + ver_str + '+2')
    # plt.legend(loc=1)
    # plt.subplot(236)
    # plq(plt, mr, 'time', mr,'red_loss, color='green', linestyle='-', label='red_loss' + run_str)
    # plq(plt, mv, 'time', mv, 'red_loss', color='red', linestyle='--', label='red_loss' + ver_str)
    # plq(plt, mr, 'time', mr,'wv_fa - 2, color='green', linestyle='-', label='wrap_vb_fa' + run_str + '-2')
    # plq(plt, mv, 'time', mv, 'wv_fa', add=-2, color='red', linestyle='--', label='wrap_vb_fa' + ver_str + '-2')
    # plq(plt, mr, 'time', mr,'ccd_fa - 4, color='green', linestyle='-', label='cc_diff_fa' + run_str + '-4')
    # plq(plt, mv, 'time', mv, 'ccd_fa', add=-4, color='red', linestyle='--', label='cc_diff_fa' + ver_str + '-4')
    # plt.legend(loc=1)

    # fig_list.append(plt.figure())  # DOM 6
    # plt.subplot(221)
    # plt.title(plot_title + ' DOM 6')
    # print('DOM 6', end=':  ')
    # plq(plt, mr, 'time', mr, 'ib_amp_hdwe', color='blue', linestyle='-', label='ib_amp_hdwe' + run_str)
    # plq(plt, mr, 'time', mr, 'ib_noa_hdwe', color='green', linestyle='-', label='ib_noa_hdwe' + run_str)
    # plq(plt, mr, 'time', mr, 'ib_sel', color='red', linestyle='--', label='ib_sel' + run_str)
    # plq(plt, mr, 'time', mr, 'ib_charge', linestyle=':', color='blue', label='ib_charge' + run_str)
    # plq(plt, mr, 'time', mr, 'ib_diff', color='orange', linestyle='-', label='ib_diff' + run_str)
    # plq(plt, mr, 'time', mr, 'ib_diff_f', color='magenta', linestyle='--', label='ib_diff_f' + run_str)
    # plq(plt, mr, 'time', mr, 'ibd_thr', color='red', linestyle=':', label='ib_diff_thr' + run_str)
    # plq(plt, mr, 'time', mr, 'ibd_thr', slr=-1, color='red', linestyle=':')
    # plq(plt, mr, 'time', mr, 'ib_diff_flt', add=+2, color='green', linestyle='-', label='ib_diff_flt' + run_str + '+2')
    # plq(plt, mr, 'time', mr, 'ib_diff_fa', add=+2, color='magenta', linestyle='-', label='ib_diff_fa' + run_str + '+2')
    # plt.legend(loc=1)
    # plt.subplot(222)
    # plq(plt, mr, 'time', mr, 'wrap_hi_m_flt', add=+10, color='green', linestyle='-', label='wrap_hi_m_flt' + run_str + '+10')
    # plq(plt, mv, 'time', mv, 'wrap_hi_m_flt', add=+10, color='blue', linestyle='--', label='wrap_hi_m_flt' + ver_str + '+10')
    # plq(plt, mr, 'time', mr, 'wrap_hi_m_fa', add=+10, color='red', linestyle='-.', label='wrap_hi_m_fa' + run_str + '+10')
    # plq(plt, mv, 'time', mv, 'wrap_hi_m_fa', add=+10, color='cyan', linestyle=':', label='wrap_hi_m_fa' + ver_str + '+10')
    # plq(plt, mr, 'time', mr, 'wrap_hi_n_flt', add=+8, color='green', linestyle='-', label='wrap_hi_n_flt' + run_str + '+8')
    # plq(plt, mv, 'time', mv, 'wrap_hi_n_flt', add=+8, color='blue', linestyle='--', label='wrap_hi_n_flt' + ver_str + '+8')
    # plq(plt, mr, 'time', mr, 'wrap_hi_n_fa', add=+8, color='red', linestyle='-.', label='wrap_hi_n_fa' + run_str + '+8')
    # plq(plt, mv, 'time', mv, 'wrap_hi_n_fa', add=+8, color='cyan', linestyle=':', label='wrap_hi_n_fa' + ver_str + '+8')
    # plq(plt, mr, 'time', mr, 'wrap_lo_m_flt', add=+4, color='green', linestyle='-', label='wrap_lo_m_flt' + run_str + '+4')
    # plq(plt, mv, 'time', mv, 'wrap_lo_m_flt', add=+4, color='blue', linestyle='--', label='wrap_lo_m_flt' + ver_str + '+4')
    # plq(plt, mr, 'time', mr, 'wrap_lo_m_fa', add=+4, color='red', linestyle='-.', label='wrap_lo_fa' + run_str + '+4')
    # plq(plt, mv, 'time', mv, 'wrap_lo_m_fa', add=+4, color='cyan', linestyle=':', label='wrap_lo_m_fa' + ver_str + '+4')
    # plq(plt, mr, 'time', mr, 'wrap_lo_n_flt', add=+2, color='green', linestyle='-', label='wrap_lo_n_flt' + run_str + '+2')
    # plq(plt, mv, 'time', mv, 'wrap_lo_n_flt', add=+2, color='blue', linestyle='--', label='wrap_lo_n_flt' + ver_str + '+2')
    # plq(plt, mv, 'time', mr, 'wrap_lo_n_fa', add=+2, color='red', linestyle='-.', label='wrap_lo_fa' + run_str + '+2')
    # plq(plt, mv, 'time', mv, 'wrap_lo_n_fa', add=+2, color='cyan', linestyle=':', label='wrap_lo_n_fa' + ver_str + '+2')
    # plt.legend(loc=1)
    # plt.subplot(223)
    # plq(plt, mr, 'time', mr, 'e_wm_f', color='blue', linestyle='-', label='e_wrap_m_filt' + run_str)
    # plq(plt, mr, 'time', mr, 'e_wrap_m_filt', color='blue', linestyle='-', label='e_wrap_m_filt' + run_str)
    # plq(plt, mv, 'time', mv, 'e_wm_f', color='blue', linestyle='--', label='e_wrap_m_filt' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_m', color='cyan', linestyle='-.', label='e_wrap_m' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_m_trim', color='magenta', linestyle='-', label='e_wrap_m_trim' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_m_filt', color='blue', linestyle='--', label='e_wrap_m_filt' + ver_str)
    # # plq(plt, mr, 'time', mr, 'e_wrap_n_filt', color='green', linestyle='-', label='e_wrap_n_filt' + run_str)
    # plq(plt, mr, 'time', mr, 'ewhi_thr', color='red', linestyle='-.', label='ewhi_thr' + run_str)
    # plq(plt, mr, 'time', mr, 'ewlo_thr', color='red', linestyle='-.', label='ewlo_thr' + run_str)
    # plq(plt, mv, 'time', mv, 'ewmhi_thr', color='orange', linestyle=':', label='ewmhi_thr' + ver_str)
    # plq(plt, mv, 'time', mv, 'ewmlo_thr', color='orange', linestyle=':', label='ewmlo_thr' + ver_str)
    # plt.ylim(-0.2, 0.2)
    # plt.legend(loc=1)
    # plt.subplot(224)
    # plq(plt, mr, 'time', mr, 'e_wn_f', color='green', linestyle='-', label='e_wrap_n_filt' + run_str)
    # plq(plt, mr, 'time', mr, 'e_wrap_n_filt', color='green', linestyle='-', label='e_wrap_n_filt' + run_str)
    # plq(plt, mv, 'time', mv, 'e_wn_f', color='green', linestyle='--', label='e_wrap_n_filt' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_n', color='cyan', linestyle='-.', label='e_wrap_n' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_n_trim', color='magenta', linestyle='-', label='e_wrap_n_trim' + ver_str)
    # plq(plt, mv, 'time', mv, 'e_wrap_n_filt', color='green', linestyle='--', label='e_wrap_n_filt' + ver_str)
    # # plq(plt, mr, 'time', mr, 'e_wrap_m_filt', color='blue', linestyle='-', label='e_wrap_m_filt' + run_str)
    # plq(plt, mr, 'time', mr, 'ewhi_thr', color='red', linestyle=':', label='ewhi_thr' + run_str)
    # plq(plt, mr, 'time', mr, 'ewlo_thr', color='red', linestyle=':', label='ewlo_thr' + run_str)
    # plq(plt, mv, 'time', mv, 'ewnhi_thr', color='orange', linestyle=':', label='ewnhi_thr' + ver_str)
    # plq(plt, mv, 'time', mv, 'ewnlo_thr', color='orange', linestyle=':', label='ewnlo_thr' + ver_str)
    # plt.ylim(-1, 1)
    # plt.legend(loc=1)
    fig_list, fig_files = ult_plot(mr, mv, sr, smv, filename,
                                   fig_files, plot_title=plot_title, fig_list=fig_list,
                                   run_str='', ver_str='_ver', strict_overplot=strict_overplot)

    return fig_list, fig_files



def ult_plot(mr, mv, sr, smv, filename, fig_files=None, plot_title=None, fig_list=None, run_str='_run', ver_str='_ver',
             strict_overplot=False):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())  # Ult 1
    plt.subplot(331)
    plt.title(plot_title + ' Ult 1')
    print('Ult 1', end=':  ')
    plq(plt, mr, 'time', mr, 'ib_amp_hdwe', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_amp_hdwe', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_noa_hdwe', color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ib_noa_hdwe', color='orange', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'ib_amp_model', add=1., color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_amp_model', add=1., color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_noa_model', add=1., color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ib_noa_model', add=1., color='orange', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'ib_diff_f', color='magenta', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_diff_f', color='cyan', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ibd_thr', color='red', linestyle=':')
    plq(plt, mr, 'time', mr, 'ibd_thr', slr=-1, color='red', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(334)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'e_wrap', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'e_wrap_filt', color='black', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_filt', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'e_w_f', color='black', linestyle='-.', warn=False)
    plq(plt, mr, 'time', mr, 'e_wrap_n', color='green', linestyle='-.')
    plq(plt, mv, 'time', mv, 'e_wrap_n', color='pink', linestyle=':')
    plq(plt, mr, 'time', mr, 'e_wrap_n_filt', color='cyan', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'e_wn_f', color='cyan', linestyle='--', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_n_filt', color='green', linestyle='-.')
    plq(plt, mr, 'time', mr, 'cc_dif', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'cc_dif', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ewhi_thr', color='red', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ewlo_thr', color='red', linestyle='-.')
    plt.ylim(-1, 1)
    plt.legend(loc=1)
    plt.subplot(332)
    plq(plt, mr, 'time', mr, 'tb_sel', add=+6, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'tb_sel', add=+6, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'vb_sel', add=+2, color='magenta', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb_sel', add=+2, color='cyan', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'tb_flt', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'tb_flt', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'tb_fa', color='magenta', linestyle='-.')
    plq(plt, mv, 'time', mv, 'tb_fa', color='cyan', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'ib_sel_stat', add=-2, color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_sel_stat', add=-2, color='blue', linestyle='--', warn=False)
    plt.legend(loc=1)
    plt.subplot(337)
    plq(plt, mr, 'time', mr, 'e_wrap_m_filt', color='green', linestyle='-', warn=False)
    plq(plt, mr, 'time', mr, 'e_wm_f', color='green', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_m_filt', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'e_wrap_m_trim', color='magenta', linestyle='-.')
    plq(plt, mr, 'time', mr, 'e_wm_t', color='magenta', linestyle='-.', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_m_trim', color='cyan', linestyle=':', warn=False)
    plq(plt, mv, 'time', mv, 'ewmhi_thr', color='red', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ewmlo_thr', color='red', linestyle='-.')
    plt.ylim(-0.2, 0.2)
    plt.legend(loc=1)
    plt.subplot(338)
    plq(plt, mr, 'time', mr, 'cc_dif', color='black', linestyle='-')
    plq(plt, mr, 'time', mr, 'ccd_thr', color='red', linestyle='--')
    # plq(plt, mr.time, -mr.ccd_thr, color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'ccd_thr', slr=-1, color='red', linestyle='--')
    plt.ylim(-.01, .01)
    plt.legend(loc=3)
    plt.subplot(133)
    plq(plt, mr, 'time', mr, 'wrap_hi_fa', add=+24, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_fa', add=+24, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_hi_flt', add=+22, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_flt', add=+22, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_fa', add=+20, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_fa', add=+20, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_flt', add=+18, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_flt', add=+18, color='red', linestyle='--', warn=False)

    plq(plt, mr, 'time', mr, 'wrap_hi_m_fa', add=+16, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_m_fa', add=+16, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_hi_m_flt', add=+14, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_m_flt', add=+14, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_m_fa', add=+12, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_m_fa', add=+12, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_m_flt', add=+10, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_m_flt', add=+10, color='red', linestyle='--', warn=False)

    plq(plt, mr, 'time', mr, 'wrap_hi_n_fa', add=+8, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_n_fa', add=+8, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_hi_n_flt', add=+6, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_n_flt', add=+6, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_n_fa', add=+4, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_n_fa', add=+4, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_n_flt', add=+2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_n_flt', add=+2, color='red', linestyle='--', warn=False)

    plq(plt, mr, 'time', mr, 'red_loss', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'red_loss', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wv_fa', add=-2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wv_fa', add=-2, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ccd_fa', add=-4, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ccd_fa', add=-4, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_diff_fa', add=-6, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_diff_fa', add=-6, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_diff_flt', add=-8, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_diff_flt', add=-8, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_dec', color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ib_dec', color='orange', linestyle=':', warn=False)
    plt.legend(loc=1)
    plt.subplot(335)
    plq(plt, mr, 'time', mr, 'bms_off', add=+4, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'bms_off', add=+4, color='red', linestyle='--')
    if sr is not None:
        plq(plt, sr, 'time', sr, 'bms_off_s', add=+4, color='blue', linestyle='-.')
    if hasattr(mr, 'mod_data'):
        mod_min = min(min(mr.mod_data), min(mv.mod_data))
    else:
        mod_min = min(mv.mod_data)
    plq(plt, mr, 'time', mr, 'mod_data', add=-mod_min, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'mod_data', add=-mod_min, color='red', linestyle='--')
    if smv is not None:
        if hasattr(smv, 'bmso_s'):
            plq(plt, smv, 'time', smv, 'bmso_s', add=+4, color='orange', linestyle=':')
        elif hasattr(smv, 'bms_off_s'):
            plq(plt, smv, 'time', smv, 'bms_off_s', add=+4, color='orange', linestyle=':')
    plq(plt, mr, 'time', mr, 'sat', add=+2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'sat', add=+2, color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'sel', color='black', linestyle='-.')
    plq(plt, mv, 'time', mv, 'sel', color='blue', linestyle=':')
    plq(plt, mr, 'time', mr, 'ib_sel_stat', add=-2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_sel_stat', add=-2, color='red', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'vb_sel', add=-2, color='black', linestyle='--')
    plq(plt, mv, 'time', mv, 'vb_sel', add=-2, color='orange', linestyle='-.', warn=False)
    plq(plt, mr, 'time', mr, 'preserving', add=-2, color='blue', linestyle='-.')
    plt.legend(loc=1)

    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def count_text_fields(line):
    count = 0
    tokens = line.split(',')
    for token in tokens:
        try:
            float(token)
        except ValueError:
            # print(f"bad {token}")
            count += 1
    return count


def filter_f15_sequence(data_stream):
    # The escape sequence is typically ^[[28~, where ^[ is the ESC character (ASCII 27)
    # In a Python string, you can represent ESC as '\x1b' or '\033'
    f15_sequence = re.escape('\x1b[28~')
    # Use re.sub to replace the sequence with an empty string
    filtered_data = re.sub(f15_sequence, '', data_stream)
    return filtered_data


def write_clean_file(path_to_data, type_=None, hdr_key=None, unit_key=None, skip=1, comment_str='#'):
    """First line with hdr_key defines the number of fields to be imported cleanly"""
    import os
    (path, basename) = os.path.split(path_to_data)
    version = version_from_data_file(path_to_data)
    (path_to_temp, save_pdf_path, _) = local_paths(version)
    csv_file = path_to_temp+'/'+basename.replace('.csv', type_ + '.csv', 1)
    # Header
    have_header_str = None
    num_fields = 0
    with open(path_to_data, "r", encoding='cp437') as input_file:
        with open(csv_file, "w") as output:
            try:
                for line in input_file:
                    line = filter_f15_sequence(line)  # ESC[28~ injected by f15 keypress GUI_TestSOC to keep term awake
                    if line.__contains__('FRAG'):
                        print(Colors.fg.red, "\n\n\nDataOverModel(write_clean_file): Heap fragmentation error\
                         detected in Particle.  Decrease NSUM constant and re-run\n\n", Colors.reset)
                        return None
                    if line.__contains__(hdr_key):
                        if have_header_str is None:
                            have_header_str = True  # write one title only
                            output.write('skip,' + line)
                            num_fields = line.count(',')  # first line with hdr_key defines number of fields
            except IOError:
                print("DataOverModel381:", line)  # last line
    # Data
    num_lines = 0
    num_text_run = 0
    num_lines_in = 0
    num_skips = 0
    unit_key_found = False
    skipped_last = False
    with (open(path_to_data, "r", encoding='cp437') as input_file):  # reads all characters even bad ones
        with open(csv_file, "a") as output:
            for line in input_file:
                line = filter_f15_sequence(line)  # ESC[28~ injected by f15 keypress GUI_TestSOC to keep term awake
                if line.__contains__(unit_key) and not line.__contains__('Config:'):
                    unit_key_found = True
                    # if line.__contains__('946s868214.902'):
                    #     print("line_run:", line_run)
                    #     print("bad line:", line)
                    #     exit(1)
                    num_text = count_text_fields(line)
                    if num_lines == 0:
                        num_text_run = num_text
                    if line.count(",") == num_fields and line.count(";") == 0 and \
                            num_text == num_text_run and \
                            re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None and \
                            (num_lines == 0 or ((num_lines_in+1) % skip) == 0) and line.count(comment_str) == 0:
                        output.write("{:2d},".format(skipped_last) + line)
                        num_lines += 1
                        skipped_last = False
                    else:
                        print(f"discarding: ", line, end='')
                        print(f"  line.count(',') == num_fields  {line.count(",") == num_fields}   \
\nAND num_text == num_text_run {num_text == num_text_run} \
\nAND re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None {re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None} \
\nAND (num_lines == 0 or ((num_lines_in+1) % skip) == 0) {(num_lines == 0 or ((num_lines_in+1) % skip) == 0)} \
\nAND line.count(comment_str) == 0 {line.count(comment_str) == 0}")
                        print(f"{line.count(',')=} {num_fields=}")
                        print(f"{line[-1]=}")
                        print(f"{num_text=} {num_text_run=}")
                        num_skips += 1
                        skipped_last = True
                    num_lines_in += 1
    if not num_lines:
        csv_file = None
        print("I(write_clean_file): no data to write")
        if not unit_key_found:
            print("W(write_clean_file):  unit_key not found in ", basename, ".  Looking with '{:s}'".format(unit_key))
    else:
        print("Wrote(write_clean_file):", csv_file, num_lines, "lines", num_skips, "skips", num_fields, "fields")
    return csv_file


class SavedData:
    def __init__(self, battery=None, rap=None, sel=None, ekf=None, temp=None, shunt=None,
                 time_end=None, zero_zero=False, zero_thr=0.02, sync_cTime=None, init_time_in=None, time_shift_in=None,
                 str_=None):
        self.str = str_
        i_end = 0
        n = None
        ib_lag = None
        self.time_shift = time_shift_in

        # Load off-nominal Battery values
        if battery is not None:
            # Scroll through all off-nominals make dictionary
            self.Battery_off_dict = {}
            for field_name in battery.dtype.names:
                print(f"field_name {field_name}", end='')
                try:
                    self.Battery_off_dict[field_name] = battery[field_name][-1]
                except IndexError:
                    self.Battery_off_dict[field_name] = battery[field_name]
                    print(f"Battery_off field_name {field_name}   valeu {battery[field_name]}")
            # print(self.Battery_off_dict)
            # Print affected values
            print(f"dictionary to apply to Battery class")
            if self.Battery_off_dict:
                for key in dir(Battery):
                    if key in self.Battery_off_dict and key.isupper() and not key.startswith('__'):
                        print(f"Battery.{key} {getattr(Battery, key)} --> ", end='')
                        print("Battery.{:s} = {:8.6g}".format(key, self.Battery_off_dict[key]))


        if rap is None:
            IbLag = None
            self.skip_rap = None
            self.i = 0
            self.time = None
            self.reset = None
            self.reset_all_faults = None
            self.soft_reset = None
            self.reset_temp = None
            self.soft_reset_sim = None
            self.init_mon = None
            self.init_sim = None
            self.time_min = None
            self.time_day = None
            self.dt = None  # Update time, s
            self.unit = None  # text title
            self.hm = None  # hours, minutes
            self.cTime = None  # Control time, s
            self.ib = None  # Bank current, A
            self.ib_f = None  # Bank current filtered, A
            self.ioc = None  # Hys indicator current, A
            self.voc = None
            self.voc_soc = None
            # self.ib_past = None  # Past bank current, A
            self.ib_charge = None  # BMS switched current, A
            self.vb = None  # Bank voltage, V
            self.chm = None  # Battery chemistry code
            self.qcrs = None  # Unit capacity rated scaled, Coulombs
            self.d_delta_q = None  # Change in the charge for update, Coulombs
            self.delta_q = None  # Change in the charge for update, Coulombs
            self.q_capacity = None  # Charge capacity at instant, Coulombs
            self.sat = None  # Indication that battery is saturated, T=saturated
            self.ib_lag = None  # Lagged indication that battery is saturated, 1=saturated
            self.sel = None  # Current source selection, 0=amp, 1=no amp
            self.mod = None  # Configuration control code, 0=all hardware, 7=all simulated, +8 tweak test
            self.bms_off = None  # Battery management system off, T=off
            self.Tb_rap = None  # Battery bank temperature, deg C
            self.Tb_f_rap = None  # Battery bank filtered temperature, deg C
            self.Tb_f_rate_rap = None  # Battery bank filtered temperature, deg C
            self.vsat = None  # Monitor Bank saturation threshold at temperature, deg C
            self.dv_dyn = None  # Monitor Bank current induced back emf, V
            self.dv_hys = None  # Drop across hysteresis, V
            self.voc_stat = None  # Monitor Static bank open circuit voltage, V
            self.voc = None  # Bank VOC estimated from vb and RC model, V
            self.voc_ekf = None  # Monitor bank solved static open circuit voltage, V
            self.y_ekf = None  # Monitor single battery solver error, V
            self.y_ekf_f = None  # Monitor single battery solver filtered error, V
            self.soc_s = None  # Simulated state of charge, fraction
            self.soc_ekf = None  # Solved state of charge, fraction
            self.soc = None  # Coulomb Counter fraction of saturation charge (q_capacity_) available (0-1)
            self.time_run = 0.  # Adjust time for start of ib input
            self.voc_soc_new = None  # For studies
            self.init_time = None
            self.ib_dyn_lstate = None
            self.ib_dyn_rstate = None
        else:
            self.skip_rap = np.bool(np.array(rap.skip))
            self.i = 0
            self.cTime = np.array(rap.cTime)
            self.time = np.array(rap.cTime)
            self.reset = np.array(rap.reset)
            self.reset_all_faults = np.array(rap.reset_all_faults)
            self.soft_reset = np.array(rap.soft_reset)
            self.soft_reset_sim = np.array(rap.soft_reset_sim)
            self.init_mon = np.array(rap.init_mon)
            self.init_sim = np.array(rap.init_sim)
            self.reset_temp = np.array(rap.reset_temp)
            self.ib = np.array(rap.ib)
            # manage data shape
            # Find first non-zero ib and use to adjust time
            # Ignore initial run of non-zero ib because resetting from previous run
            if zero_zero:
                self.zero_end = 0
            elif sync_cTime is not None:
                self.zero_end = np.where(self.cTime < sync_cTime[0])[0][-1] + 2
            else:
                try:
                    self.zero_end = 0
                    # stop after first non-zero
                    while self.zero_end < len(self.ib) and abs(self.ib[self.zero_end]) < zero_thr:
                        self.zero_end += 1
                    self.zero_end -= 1  # backup one
                    if self.zero_end == len(self.ib) - 1:
                        print(Colors.fg.red, f"\n\nLikely ib is zero throughout the data.  Check setup and retry\n\n",
                              Colors.reset)
                        self.zero_end = 0
                    elif self.zero_end == -1:
                        print(Colors.fg.red, f"\n\nLikely ib is noisy throughout the data.  Check setup and retry\n\n",
                              Colors.reset)
                        self.zero_end = 0
                except IOError:
                    self.zero_end = 0
            self.time_run = self.time[self.zero_end]
            self.time -= self.time_run
            self.time_min = self.time / 60.
            self.time_day = self.time / 3600. / 24.

            # Truncate
            i_end = None
            i_end_sel =  None
            i_end_shunt = None
            if time_end is None:
                if temp is not None:
                    time_t = np.atleast_1d(np.array(np.array(temp.c_time) - self.time_run))
                    Tt = np.atleast_1d(np.array(temp.T_t))
                    time_end = time_t[-1] + Tt[-1]
                    i_end = np.where(self.time <= time_end)[0][-1] + 1
                else:
                    i_end = len(self.time)
                if sel is not None:
                    self.c_time_s = np.array(sel.c_time) - self.time_run
                    i_end = min(i_end, len(self.c_time_s))
                if ekf is not None:
                    self.time_e = np.array(np.atleast_1d(ekf.c_time) - self.time_run)
                if shunt is not None:
                    self.c_time_shunt = np.array(np.atleast_1d(shunt.c_time) - self.time_run)
                    i_end = min(i_end, len(self.c_time_shunt))
            else:
                if temp is not None:
                    time_t = np.atleast_1d(np.array(np.array(temp.c_time) - self.time_run))
                    Tt = np.atleast_1d(np.array(temp.T_t))
                    i_end = np.where(self.time <= time_end)[0][-1] + 1
                else:
                    i_end = len(self.time)
                if sel is not None:
                    self.c_time_s = np.array(sel.c_time) - self.time_run
                    i_end_sel = np.where(self.c_time_s <= time_end)[0][-1] + 1
                    i_end = np.minimum(i_end, i_end_sel)
                    self.zero_end = np.minimum(self.zero_end, i_end-1)
                if ekf is not None:
                    self.time_e = np.array(np.atleast_1d(ekf.c_time) - self.time_run)
                if shunt is not None:
                    self.c_time_shunt = np.array(shunt.c_time) - self.time_run
                    i_end_shunt = np.where(self.c_time_shunt <= time_end)[0][-1] + 1
                    i_end = np.minimum(i_end, i_end_shunt)
                    self.zero_end = np.minimum(self.zero_end, i_end-1)
            self.cTime = self.cTime[:i_end]
            self.dt = np.array(rap.dt[:i_end])
            self.time = np.array(self.time[:i_end])
            if self.time_shift:
                self.time += self.time_shift
            self.reset = np.array(rap.reset[:i_end])
            self.reset_all_faults = np.array(rap.reset_all_faults[:i_end])
            self.reset_temp = np.array(rap.reset_temp[:i_end])
            self.soft_reset = np.array(rap.soft_reset[:i_end])
            self.soft_reset_sim = np.array(rap.soft_reset_sim[:i_end])
            self.init_mon = np.array(rap.init_mon[:i_end])
            self.init_sim = np.array(rap.init_sim[:i_end])
            self.ib = np.array(rap.ib[:i_end])
            self.ioc = np.array(rap.ib[:i_end])
            self.voc_soc = np.array(rap.voc_soc[:i_end])
            self.vb = np.array(rap.vb[:i_end])
            self.chm = np.array(rap.chm[:i_end])
            if hasattr(rap, 'qcrs'):
                self.qcrs = rap.qcrs[:i_end]
            if hasattr(rap, 'd_delta_q'):
                self.d_delta_q = rap.d_delta_q[:i_end]
            if hasattr(rap, 'delta_q'):
                self.delta_q = rap.delta_q[:i_end]
            if hasattr(rap, 'qcap'):
                self.q_capacity = rap.qcap[:i_end]
            self.sat = np.array(rap.sat[:i_end])
            # Lag for saturation
            n = len(self.cTime)
            ib_lag = Chemistry_BMS.ib_lag(self.chm[0])
            IbLag = LagExp(1., ib_lag, -100., 100.)
            self.ib_lag = np.zeros(n)
            self.sel = np.array(rap.sel[:i_end])
            self.mod_data = np.array(rap.mod[:i_end])
            self.bms_off = np.array(rap.bmso[:i_end])
            # not_bms_off = self.bms_off < 1
            # bms_off_and_not_charging = self.bms_off * not_bms_off
            # self.ib_charge = self.ib * (bms_off_and_not_charging < 1)
            self.ib_charge = np.array(rap.ib_charge[:i_end])
            self.Tb_rap = np.array(rap.Tb_rap[:i_end])
            self.Tb_f_rap = np.array(rap.Tb_f_rap[:i_end])
            self.Tb_f_rate_rap = np.array(rap.Tb_f_rate_rap[:i_end])
            self.vsat = np.array(rap.vsat[:i_end])
            self.dv_dyn = np.array(rap.dv_dyn[:i_end])
            if hasattr(rap, 'ib_dyn_lstate'):
                self.ib_dyn_lstate = np.array(rap.ib_dyn_lstate[:i_end])
            else:
                self.ib_dyn_lstate = self.vsat*0.
            if hasattr(rap, 'ib_dyn_rstate'):
                self.ib_dyn_rstate = np.array(rap.ib_dyn_rstate[:i_end])
            else:
                self.ib_dyn_rstate = self.vsat*0.
            self.ib_dyn = np.array(rap.ib_dyn[:i_end])
            self.voc_stat = np.array(rap.voc_stat[:i_end])
            self.voc = self.vb - self.dv_dyn
            self.dv_hys = np.array(rap.dv_hys[:i_end])
            self.voc_ekf = np.array(rap.voc_ekf[:i_end])
            self.y_ekf = np.array(rap.y_ekf[:i_end])
            self.soc_s = np.array(rap.soc_s[:i_end])
            self.soc_ekf = np.array(rap.soc_ekf[:i_end])
            self.soc = np.array(rap.soc[:i_end])
            self.voc_soc_new = None
        if sel is None:
            self.skip_sel = None
            self.c_time_s = None
            self.user_sel = None
            self.cc_dif = None
            self.ccd_fa = None
            self.ib_amp_hdwe = None
            self.ib_noa_hdwe = None
            self.ib_amp_model = None
            self.ib_noa_model = None
            self.ibm = None
            self.ib_amp_hdwe_kf = None
            self.ibnkf = None
            self.vovcn = None
            self.ib_noa_lo = None
            self.ib_noa_hi = None
            self.ib_noa_kf = None
            self.kfres = None
            self.x1n = None
            self.ib_amp_lo = None
            self.ib_amp_hi = None
            self.ib_diff = None
            self.ib_diff_f = None
            self.ib_diff_flt = None
            self.ib_diff_fa = None
            self.e_wrap = None
            self.e_wrap_filt = None
            self.e_wrap_trim = None
            self.ib_dyn_m = None
            self.dv_dyn_m = None
            self.ib_dyn_n = None
            self.dv_dyn_n = None
            self.e_wrap_m = None
            self.e_wrap_m_filt = None
            self.e_wrap_m_trim = None
            self.e_wrap_m_trimmed = None
            self.e_wrap_m_reset = None
            self.ib_amp = None
            self.e_wrap_n = None
            self.e_wrap_n_filt = None
            self.e_wrap_n_trim = None
            self.wrap_hi_flt = None
            self.wrap_hi_m_flt = None
            self.wrap_hi_n_flt = None
            self.wrap_lo_flt = None
            self.wrap_lo_m_flt = None
            self.wrap_lo_n_flt = None
            self.red_loss = None
            self.wrap_hi_fa = None
            self.wrap_hi_m_fa = None
            self.wrap_hi_n_fa = None
            self.wrap_lo_fa = None
            self.wrap_lo_m_fa = None
            self.wrap_lo_n_fa = None
            self.wv_fa = None
            self.ib_sel_stat = None
            self.ib_h = None
            self.ib_s = None
            self.mib = None
            self.ib_sel = None
            self.vb_h = None
            self.vb_s = None
            self.mvb = None
            self.vb = self.vb
            self.Tb_h = None
            self.Tb_s = None
            self.mtb = None
            self.Tb_fa = None
            self.vb_sel = None
            self.ib_rate = None
            self.ib_quiet = None
            self.ib_really_quiet = None
            self.dscn_flt = None
            self.dscn_fa = None
            self.vb_flt = None
            self.vb_fa = None
            self.tb_sel = None
            self.tb_flt = None
            self.tb_fa = None
            self.ccd_thr = None
            self.ewhi_thr = None
            self.ewlo_thr = None
            self.ewhm_thr = None
            self.ewlm_thr = None
            self.ibd_thr = None
            self.ibq_thr = None
            self.preserving = None
            self.y_ekf_f = None
            self.ib_dec = None
            self.ib_dyn_T_m = None
            self.ib_dyn_tau_m = None
            self.ib_dyn_rstate_m = None
            self.ib_dyn_lstate_m = None
            self.ib_dyn_T_n = None
            self.ib_dyn_tau_n = None
            self.ib_dyn_rstate_n = None
            self.ib_dyn_lstate_n = None
            self.ib_wrp_T_m = None
            self.ib_wrp_tau_m = None
            self.ib_wrp_T_n = None
            self.ib_wrp_tau_n = None
            self.disable_amp_fault = None
            self.disable_amp_fault_per = None
            self.vr = None
            self.ib_wrp_rate_n = None
            self.ib_wrp_state_n = None
            self.ib_wrp_reset_m = None
            self.ib_wrp_rate_m = None
            self.ib_wrp_state_m = None
            self.vb_functional_flt = None
            self.vb_functional_fa = None
            self.wrap_m_and_n_fa = None
            self.ib_is_functional = None
            self.voltage_low = None
            self.vb_model = None
            self.vb_hdwe = None
            self.vb_hdwe_f = None
        else:
            falw = np.array(sel.falw[:i_end], dtype=np.uint32)
            fltw = np.array(sel.fltw[:i_end], dtype=np.uint32)
            self.skip_sel = np.array(np.bool(sel.skip[:i_end]))
            self.c_time_s = np.array(sel.c_time[:i_end]) - self.time_run
            self.user_sel = np.array(sel.user_sel[:i_end])
            self.cc_dif = np.array(sel.cc_dif[:i_end])
            self.ccd_fa = np.bool_(np.array(falw) & 2**4)
            self.ib_amp_hdwe = np.array(sel.ibmh[:i_end])
            self.ib_noa_hdwe = np.array(sel.ibnh[:i_end])
            self.ib_amp_model = np.array(sel.ibmm[:i_end])
            self.ib_noa_model = np.array(sel.ibnm[:i_end])
            self.ib_model = np.array(sel.ibm[:i_end])
            self.ib_amp_hdwe_kf = np.array(sel.ibmkf[:i_end])
            self.ib_noa_hdwe_kf = np.array(sel.ibnkf[:i_end])
            self.vovcn = np.array(sel.vovcn[:i_end])
            self.ib_noa_kf = np.array(sel.ib_noa_kf[:i_end])
            self.kfres = np.array(sel.kfres[:i_end])
            self.x1n = np.array(sel.x1n[:i_end])
            self.ib_noa_lo = np.array(sel.ib_noa_lo[:i_end])
            self.ib_noa_hi = np.array(sel.ib_noa_hi[:i_end])
            self.ib_amp_lo = np.array(sel.ib_amp_lo[:i_end])
            self.ib_amp_hi = np.array(sel.ib_amp_hi[:i_end])
            self.ib_diff = np.array(sel.ib_diff[:i_end])
            self.ib_diff_f = np.array(sel.ib_diff_f[:i_end])
            self.ib_diff_flt = np.bool_((np.array(fltw) & 2**8) | (np.array(fltw) & 2**9))
            self.ib_diff_fa = np.bool_((np.array(falw) & 2**8) | (np.array(falw) & 2**9))
            self.e_wrap = np.array(sel.e_w[:i_end])
            self.e_wrap_filt = np.array(sel.e_w_f[:i_end])
            self.ib_dyn_m = np.array(sel.ib_dm[:i_end])
            self.dv_dyn_m = np.array(sel.dv_dm[:i_end])
            self.ib_dyn_n = np.array(sel.ib_dn[:i_end])
            self.dv_dyn_n = np.array(sel.dv_dn[:i_end])
            # self.e_wrap_trim = np.array(sel.e_w_t[:i_end])
            if hasattr(sel, 'e_wm'):
                self.e_wrap_m = np.array(sel.e_wm[:i_end])
            if hasattr(sel, 'e_wm_f'):
                self.e_wrap_m_filt = np.array(sel.e_wm_f[:i_end])
            if hasattr(sel, 'e_wm_r'):
                self.e_wrap_m_reset = np.array(sel.e_wm_r[:i_end])
            if hasattr(sel, 'e_wn'):
                self.e_wrap_n = np.array(sel.e_wn[:i_end])
            if hasattr(sel, 'e_wn_f'):
                self.e_wrap_n_filt = np.array(sel.e_wn_f[:i_end])
            if hasattr(sel, 'e_wm_t'):
                self.e_wrap_m_trim = np.array(sel.e_wm_t[:i_end])
            self.e_wrap_m_trimmed = np.array(sel.ib_wrp_tr_m[:i_end])
            self.vb_model = np.array(sel.vb_m[:i_end])
            if hasattr(sel, 'vb_h'):
                self.vb_hdwe = np.array(sel.vb_h[:i_end])
            else:
                self.vb_hdwe = np.array(sel.vb[:i_end])
            if hasattr(sel, 'vb_h_f'):
                self.vb_hdwe_f = np.array(sel.vb_h_f[:i_end])
            else:
                self.vb_hdwe_f = np.array(sel.vb_h[:i_end])
            self.voc_m = np.array(sel.voc_m[:i_end])
            self.voc_soc_m = np.array(sel.voc_soc_m[:i_end])
            if hasattr(sel, 'ib_amp'):
                self.ib_amp = np.array(sel.ib_amp[:i_end])
            self.wrap_hi_flt = np.bool_(np.array(fltw) & 2**5)
            self.wrap_lo_flt = np.bool_(np.array(fltw) & 2**6)
            self.wrap_hi_m_flt = np.bool_(np.array(fltw) & 2**14)
            self.wrap_lo_m_flt = np.bool_(np.array(fltw) & 2**15)
            self.wrap_hi_n_flt = np.bool_(np.array(fltw) & 2**16)
            self.wrap_lo_n_flt = np.bool_(np.array(fltw) & 2**17)
            self.wrap_m_and_n_flt = (self.wrap_lo_n_flt & self.wrap_lo_m_flt) | (self.wrap_hi_n_flt & self.wrap_hi_m_flt)
            self.red_loss = np.bool_(np.array(fltw) & 2**7)
            self.wrap_hi_fa = np.bool_(np.array(falw) & 2**5)
            self.wrap_lo_fa = np.bool_(np.array(falw) & 2**6)
            self.wv_fa = np.bool_(np.array(falw) & 2**7)
            self.wrap_hi_m_fa = np.bool_(np.array(falw) & 2**14)
            self.wrap_lo_m_fa = np.bool_(np.array(falw) & 2**15)
            self.wrap_hi_n_fa = np.bool_(np.array(falw) & 2**16)
            self.wrap_lo_n_fa = np.bool_(np.array(falw) & 2**17)
            self.wrap_m_and_n_fa = (self.wrap_lo_n_fa & self.wrap_lo_m_fa) | (self.wrap_hi_n_fa & self.wrap_hi_m_fa)
            self.ib_sel_stat = np.array(sel.ib_sel_stat[:i_end])
            self.ib_h = np.array(sel.ib_h[:i_end])
            self.ib_s = np.array(sel.ib_s[:i_end])
            self.mib = np.array(sel.mib[:i_end])
            self.ib_sel = np.array(sel.ib[:i_end])
            self.vb_h = np.array(sel.vb_h[:i_end])
            self.vb_s = np.array(sel.vb_s[:i_end])
            self.mvb = np.array(sel.mvb[:i_end])
            self.vb = np.array(sel.vb[:i_end])
            self.mtb = np.array(sel.mtb[:i_end])
            self.Tb_fa = np.array(sel.Tb_fa[:i_end])
            self.vb_sel = np.array(sel.vb_sel[:i_end])
            self.ib_rate = np.array(sel.ib_rate[:i_end])
            self.ib_quiet = np.array(sel.ib_quiet[:i_end])
            self.ib_really_quiet = np.array(sel.ib_really_quiet[:i_end])

            """    String::format("1 wnl     %d  %d 'Fo ^'\n", wrap_lo_n_flt(), wrap_lo_n_fa()) +
                String::format("0 wnh     %d  %d 'Fi ^'\n", wrap_hi_n_flt(), wrap_hi_n_fa()) +
                String::format("F wml     %d  %d 'Fo ^'\n", wrap_lo_m_flt(), wrap_lo_m_fa()) +
                String::format("E wmh     %d  %d 'Fi ^'\n", wrap_hi_m_flt(), wrap_hi_m_fa()) +
                String::format("D vc      %d  %d 'FI 1'\n", vc_flt(), vc_fa()) +
                String::format("C bare n  %d  x \n", ib_noa_bare()) +
                String::format("B bare m  %d  x \n", ib_amp_bare()) +
                String::format("A ib_dsc  %d  %d 'Fq v'\n", ib_dscn_flt(), ib_dscn_fa()) +
                String::format("9 ibd_lo  %d  %d 'Fd ^  *SA/*SB'\n", ib_diff_lo_flt(), ib_diff_lo_fa()) +
                String::format("8 ibd_hi  %d  %d 'Fd ^  *SA/*SB'\n", ib_diff_hi_flt(), ib_diff_hi_fa()) +
                String::format("7 red wv  %d  %d   'Fd, Fi/Fo ^'\n",  red_loss(), wrap_vb_fa()) +
                String::format("6 wl      %d  %d 'Fo ^'\n", wrap_lo_flt(), wrap_lo_fa()) +
                String::format("5 wh      %d  %d 'Fi ^'\n", wrap_hi_flt(), wrap_hi_fa()) +
                String::format("4 vc | cc_dif %d  %d 'x Fc ^'\n", vc_fa(), cc_diff_fa()) +
                String::format("3 ib n    %d  %d 'FI 1'\n", ib_noa_flt(), ib_noa_fa()) +
                String::format("2 ib m    %d  %d 'FI 1'\n", ib_amp_flt(), ib_amp_fa()) +
                String::format("1 vb      %d  %d 'Fv 1  *SV, *Dc/*Dv'.", vb_flt(), vb_fa()) +  String::format("  bms_off %d\n", Mon->bms_off()) +
                String::format("0 tb      %d  %d 'Ft 1'\n  ", tb_flt(), tb_fa()) +
                String::format("    Fault  Fail'\n");
            """
            self.dscn_flt = np.bool_(np.array(fltw) & 2**10)
            self.dscn_fa = np.bool_(np.array(falw) & 2**10)
            self.vb_flt = np.bool_(np.array(fltw) & 2**1)
            self.vb_fa = np.bool_(np.array(falw) & 2**1)
            self.tb_sel = np.array(sel.tb_sel[:i_end])
            self.tb_flt = np.bool_(np.array(fltw) & 2**0)
            self.tb_fa = np.bool_(np.array(falw) & 2**0)
            self.ccd_thr = np.array(sel.ccd_thr[:i_end])
            self.ewhi_thr = np.array(sel.ewh_thr[:i_end])
            self.ewlo_thr = np.array(sel.ewl_thr[:i_end])
            self.ewhm_thr = self.ewhi_thr / 10.  # WRAP_HI_NOA / WRAP_HI_AMP = SHUNT_AMP_R2 / SHUNT_NOA_R2
            self.ewlm_thr = self.ewlo_thr / 10.  # WRAP_LO_NOA / WRAP_LO_AMP = SHUNT_AMP_R2 / SHUNT_NOA_R2
            self.ibd_thr = np.array(sel.ibd_thr[:i_end])
            self.ibq_thr = np.array(sel.ibq_thr[:i_end])
            self.preserving = np.array(sel.preserving[:i_end])
            if hasattr(sel, 'y_ekf_f'):
                self.y_ekf_f = np.array(sel.y_ekf_f[:i_end])
            if hasattr(sel, 'ib_dec'):
                self.ib_dec = np.array(sel.ib_dec[:i_end])
            self.ib_dyn_T_m = np.array(sel.ib_dyn_T_m[:i_end])
            self.ib_dyn_rstate_m = np.array(sel.ib_dyn_rstate_m[:i_end])
            self.ib_dyn_lstate_m = np.array(sel.ib_dyn_lstate_m[:i_end])
            self.ib_dyn_tau_m = np.array(sel.ib_dyn_tau_m[:i_end])
            self.ib_dyn_T_n = np.array(sel.ib_dyn_T_n[:i_end])
            self.ib_dyn_rstate_n = np.array(sel.ib_dyn_rstate_n[:i_end])
            self.ib_dyn_lstate_n = np.array(sel.ib_dyn_lstate_n[:i_end])
            self.ib_dyn_tau_n = np.array(sel.ib_dyn_tau_n[:i_end])
            self.ib_wrp_T_m = np.array(sel.ib_wrp_T_m[:i_end])
            self.ib_wrp_rate_m = np.array(sel.ib_wrp_rate_m[:i_end])
            self.ib_wrp_reset_m = np.array(sel.ib_wrp_reset_m[:i_end])
            self.ib_wrp_state_m = np.array(sel.ib_wrp_state_m[:i_end])
            self.ib_wrp_tau_m = np.array(sel.ib_wrp_tau_m[:i_end])
            self.ib_wrp_T_n = np.array(sel.ib_wrp_T_n[:i_end])
            self.ib_wrp_rate_n = np.array(sel.ib_wrp_rate_n[:i_end])
            self.ib_wrp_state_n = np.array(sel.ib_wrp_state_n[:i_end])
            self.ib_wrp_tau_n = np.array(sel.ib_wrp_tau_n[:i_end])
            self.disable_amp_fault = np.array(sel.disable_amp_fault[:i_end])
            self.disable_amp_fault_per = np.array(sel.disable_amp_fault_per[:i_end])
            if hasattr(sel, 'vr'):
                self.vr = np.array(sel.vr[:i_end])
            self.wrap_m_and_n_fa = np.bool_(np.array(sel.wrap_m_and_n_fa[:i_end]))
            self.ib_is_functional = np.bool_(np.array(sel.ib_is_functional[:i_end]))
            self.voltage_low = np.bool_(np.array(sel.v_low[:i_end]))
        if shunt is None:
            self.i = 0
            self.Vcm = None
            self.Vom = None
            self.VoVcm = None
            self.Vcn = None
            self.Von = None
            self.VoVcn = None
            self.Tbv = None
            self.Vbv = None
        else:
            self.assign_all_from(shunt, i_end)
            # Special handling
            self.c_time_shunt = np.array(shunt.c_time[:i_end]) - self.time_run

        if ekf is None:
            self.skip_e = None
            self.time_e = None
            self.dt_ekf = None
            self.Fx = None
            self.Bu = None
            self.Q = None
            self.R = None
            self.P = None
            self.S = None
            self.K = None
            self.u = None
            self.x = None
            self.y = None
            self.z = None
            self.x_prior = None
            self.frz = None
            self.P_prior = None
            self.x_post = None
            self.P_post = None
            self.hx = None
            self.H = None
            self.tb_f_for_hx = None
            self.x_for_hx = None
            self.voc_stat_f_a = None
            self.voc_stat_f_b = None
            self.voc_stat_f_b = None
            self.voc_stat_f_T = None
            self.voc_stat_f_tau = None
            self.voc_stat_f_rstate = None
            self.voc_stat_f_lstate = None
        else:
            self.skip_e = np.bool(np.atleast_1d(ekf.skip)[:i_end])
            self.time_e = np.array(np.atleast_1d(ekf.c_time)[:i_end] - self.time_run)
            self.dt_ekf = np.array(np.atleast_1d(ekf.dt)[:i_end])
            self.Fx = np.array(np.atleast_1d(ekf.Fx_)[:i_end])
            self.Bu = np.array(np.atleast_1d(ekf.Bu_)[:i_end])
            self.Q = np.array(np.atleast_1d(ekf.Q_)[:i_end])
            self.R = np.array(np.atleast_1d(ekf.R_)[:i_end])
            self.P = np.array(np.atleast_1d(ekf.P_)[:i_end])
            self.S = np.array(np.atleast_1d(ekf.S_)[:i_end])
            self.K = np.array(np.atleast_1d(ekf.K_)[:i_end])
            self.u = np.array(np.atleast_1d(ekf.u_)[:i_end])
            self.x = np.array(np.atleast_1d(ekf.x_)[:i_end])
            self.y = np.array(np.atleast_1d(ekf.y_)[:i_end])
            self.z = np.array(np.atleast_1d(ekf.z_)[:i_end])
            self.x_prior = np.array(np.atleast_1d(ekf.x_prior_)[:i_end])
            self.frz = np.array(np.bool(np.atleast_1d(ekf.frz_)[:i_end]))
            self.P_prior = np.array(np.atleast_1d(ekf.P_prior_)[:i_end])
            self.x_post = np.array(np.atleast_1d(ekf.x_post_)[:i_end])
            self.P_post = np.array(np.atleast_1d(ekf.P_post_)[:i_end])
            self.hx = np.array(np.atleast_1d(ekf.hx_)[:i_end])
            self.H = np.array(np.atleast_1d(ekf.H_)[:i_end])
            self.tb_f_for_hx = np.array(np.atleast_1d(ekf.tb_f_hx_)[:i_end])
            self.x_for_hx = np.array(np.atleast_1d(ekf.x_for_hx_)[:i_end])
            self.voc_stat_f_rstate = np.array(np.atleast_1d(ekf.voc_stat_rstate)[:i_end])
            self.voc_stat_f_lstate = np.array(np.atleast_1d(ekf.voc_stat_lstate)[:i_end])
            self.voc_stat_f_T = np.array(np.atleast_1d(ekf.voc_stat_T)[:i_end])
            self.voc_stat_f_tau = np.array(np.atleast_1d(ekf.voc_stat_tau)[:i_end])
        if temp is None:
            self.skip_t = None
            self.time_t = None
            self.T_t = None
            self.Tb_hdwe = None
            self.Tb_model = None
            self.Tb = None
            self.Tb_f = None
            self.Tb_f_rate = None
            self.Tb_hdwe_filt = None
            self.Tb_model_filt = None
            self.Tb_hdwe_filt_rate = None
            self.Tb_model_filt_rate = None
        else:
            self.skip_t = np.array(np.bool(np.atleast_1d(temp.skip)[:i_end]))
            self.time_t = np.array(np.atleast_1d(temp.c_time)[:i_end]) - self.time_run
            self.Tt = np.array(np.atleast_1d(temp.T_t)[:i_end])
            self.Tb_hdwe = np.array(np.atleast_1d(temp.Tb_hdw)[:i_end])
            self.Tb = np.array(np.atleast_1d(temp.Tb)[:i_end])
            self.Tb_f = np.array(np.atleast_1d(temp.Tb_f)[:i_end])
            self.Tb_f_rate = np.array(np.atleast_1d(temp.Tb_f_rate)[:i_end])
            self.Tb_model = np.array(np.atleast_1d(temp.Tb_mod)[:i_end])
            self.Tb_hdwe_filt = np.array(np.atleast_1d(temp.Tb_hdwe_filt)[:i_end])
            self.Tb_model_filt = np.array(np.atleast_1d(temp.Tb_model_filt)[:i_end])
            self.Tb_hdwe_filt_rate = np.array(np.atleast_1d(temp.Tb_hdwe_filt_rate)[:i_end])
            self.Tb_model_filt_rate = np.array(np.atleast_1d(temp.Tb_model_filt_rate)[:i_end])

        # Workarounds for incomplete data sets e.g. vv1, vv2, vv3
        if self.dv_dyn_m is None:
            self.dv_dyn_m = np.copy(self.dv_dyn)
        if self.dv_dyn_n is None:
            self.dv_dyn_n = np.copy(self.dv_dyn)
        if self.ib_amp_hdwe is None:
            self.ib_amp_hdwe = np.copy(self.ib)
        if self.ib_noa_hdwe is None:
            self.ib_noa_hdwe = np.copy(self.ib)
        if self.ib_amp_model is None:
            self.ib_amp_model = np.copy(self.ib)
        if self.ib_noa_model is None:
            self.ib_noa_model = np.copy(self.ib)
        if self.ib_dyn_m is None:
            self.ib_dyn_m = np.copy(self.ib_dyn)
        if self.ib_dyn_lstate_m is None:
            self.ib_dyn_lstate_m = np.copy(self.ib_dyn)
        if self.ib_dyn_lstate_n is None:
            self.ib_dyn_lstate_n = np.copy(self.ib_dyn)
        if self.ib_dyn_rstate_m is None:
            self.ib_dyn_rstate_m = np.copy(self.ib)
        if self.ib_dyn_rstate_n is None:
            self.ib_dyn_rstate_n = np.copy(self.ib)
        if self.ib_dyn_T_m is None:
            self.ib_dyn_T_m = np.copy(self.dt)
        if self.ib_dyn_T_n is None:
            self.ib_dyn_T_n = np.copy(self.dt)
        if self.ib_dyn_tau_m is None:
            self.ib_dyn_tau_m = np.copy(self.dt) * 0. + 10.
        if self.ib_dyn_tau_n is None:
            self.ib_dyn_tau_n = np.copy(self.dt) * 0. + 10.
        if self.ib_dyn_n is None:
            self.ib_dyn_n = np.copy(self.ib_dyn)
        if self.ib_dec is None:
            self.ib_dec = np.copy(self.ib) * 0
        if self.ib_sel is None:
            self.ib_sel = np.copy(self.ib)
        if self.ib_sel_stat is None:
            self.ib_sel_stat = np.copy(self.ib) * 0
        if self.ib_h is None:
            self.ib_h = np.copy(self.ib)
        if self.ib_s is None:
            self.ib_s = np.copy(self.ib)
        if self.ib_wrp_reset_m is None:
            self.ib_wrp_reset_m = np.copy(self.dt) * 0
        if self.ib_wrp_rate_m is None:
            self.ib_wrp_rate_m = np.copy(self.dt) * 0.
        if self.ib_wrp_state_m is None:
            self.ib_wrp_state_m = np.copy(self.dt) * 0.
        if self.ib_wrp_T_m is None:
            self.ib_wrp_T_m = np.copy(self.dt)
        if self.ib_wrp_tau_m is None:
            self.ib_wrp_tau_m = np.copy(self.dt) * 0. + 10.
        if self.ib_wrp_rate_n is None:
            self.ib_wrp_rate_n = np.copy(self.dt) * 0.
        if self.ib_wrp_state_n is None:
            self.ib_wrp_state_n = np.copy(self.dt) * 0.
        if self.ib_wrp_T_n is None:
            self.ib_wrp_T_n = np.copy(self.dt)
        if self.ib_wrp_tau_n is None:
            self.ib_wrp_tau_n = np.copy(self.dt) * 0. + 10.
        if self.e_wrap_m is None:
            self.e_wrap_m = np.copy(self.ib) * 0.
        if self.e_wrap_m_filt is None:
            self.e_wrap_m_filt = np.copy(self.ib) * 0.
        if self.e_wrap_m_reset is None:
            self.e_wrap_m_reset = np.copy(self.ib) * 0
        if self.e_wrap_m_trim is None:
            self.e_wrap_m_trim = np.copy(self.ib) * 0.
        if self.ib_amp is None:
            self.ib_amp = np.copy(self.ib) * 0.
        if self.e_wrap_n is None:
            self.e_wrap_n = np.copy(self.ib) * 0.
        if self.e_wrap_n_filt is None:
            self.e_wrap_n_filt = np.copy(self.ib) * 0.
        if self.e_wrap is None:
            self.e_wrap = np.copy(self.ib) * 0.
        if self.e_wrap_filt is None:
            self.e_wrap_filt = np.copy(self.ib) * 0.
        if self.mvb is None:
            self.mvb = np.bool(np.copy(self.mod_data))
        if self.Tb is None:
            self.Tb = np.copy(self.Tb_rap)
        if self.Tb_f is None:
            self.Tb_f = np.copy(self.Tb_f_rap)
        if self.Tb_f_rate is None:
            self.Tb_f_rate = np.copy(self.Tb_f_rate_rap)
        if self.Tb_hdwe is None:
            self.Tb_hdwe = np.copy(self.Tb_rap)
        if self.Tb_hdwe_filt_rate is None:
            self.Tb_hdwe_filt_rate = np.copy(self.Tb_f_rate_rap)
        if self.Tb_model_filt_rate is None:
            self.Tb_model_filt_rate = np.copy(self.Tb_f_rate_rap)
        if self.Tb_hdwe_filt is None:
            print(f"Using Tb_f_rap to initialize Tb_hdwe_filt")
            self.Tb_hdwe_filt = np.copy(self.Tb_f_rap)
        if self.Tb_model is None:
            self.Tb_model = np.copy(self.Tb_rap)
        if self.Tb_model_filt is None:
            print(f"Using Tb_f_rap to initialize Tb_model_filt")
            self.Tb_model_filt = np.copy(self.Tb_f_rap)
        if self.dt_ekf is None:
            self.dt_ekf = np.copy(self.dt)
        if self.vb_h is None:
            self.vb_h = np.copy(self.vb)
        if self.x is None:
            self.x = np.copy(self.soc_ekf)
        if self.x_prior is None:
            self.x_prior = np.copy(self.soc_ekf)
        if self.x_post is None:
            self.x_post = np.copy(self.soc_ekf)
        if self.y_ekf is None:
            self.y_ekf = np.copy(self.voc_stat) * 0.
        if self.y_ekf_f is None:
            self.y_ekf_f = np.copy(self.voc_stat) * 0.
        if self.z is None:
            self.z = np.copy(self.voc_stat)
        if self.H is None:
            self.H = np.copy(self.voc_stat)
        if self.hx is None:
            self.hx = np.copy(self.voc_stat)
        if self.K is None:
            self.K = np.copy(self.x) * 0.
        if self.P is None:
            self.P = np.copy(self.x) * 0.
        if self.P_post is None:
            self.P_post = np.copy(self.x) * 0.
        if self.P_prior is None:
            self.P_prior = np.copy(self.x) * 0.
        if self.Q is None:
            self.Q = np.copy(self.x) * 0.
        if self.S is None:
            self.S = np.copy(self.x) * 0.
        if self.tb_f_for_hx is None:
            self.tb_f_for_hx = np.copy(self.Tb_f)
        if self.x_for_hx is None:
            self.x_for_hx = np.copy(self.x)
        if self.disable_amp_fault is None:
            self.disable_amp_fault = np.copy(self.ib) * 0
        if self.disable_amp_fault_per is None:
            self.disable_amp_fault_per = np.copy(self.ib) * 0
        if self.time_e is None:
            self.time_e = np.copy(self.dt)
        if self.time_t is None:
            self.time_t = np.copy(self.dt)

        # Initialization time logic
        if init_time_in:
            self.init_time = init_time_in
        else:
            if self.time[0] == 0.:  # no initialization flat detected at beginning of recording
                self.init_time = 1.
            else:
                self.init_time = -4.

        for i in range(n):
            if self.time[i] <= self.init_time:
                lag_reset = True
                if i < n-1:
                    T_lag = self.cTime[i+1] - self.cTime[i]
                else:
                    T_lag = self.cTime[i] - self.cTime[i-1]
            else:
                lag_reset = False
                T_lag = self.cTime[i] - self.cTime[i-1]
            self.ib_lag[i] = IbLag.calculate_tau(float(self.ib[i]), lag_reset, T_lag, ib_lag)

    def assign_all_from(self, x=None, i_end=None):
        """
        Iterates over members of a dataset x, assigns values to numpy.ndarray members
        """
        for name in list(x.dtype.names):
            if i_end is None:
                setattr(self, name, x[name])
            else:
                setattr(self, name, getattr(x, name)[:i_end])

    def truncate(self, i_end=None, key_attr='time'):
        """
        Iterates over members of a self, assigns values to numpy.ndarray members
        up to i_end.
        """
        for attr_name in dir(self):
            # Filter out built-in attributes and methods
            if not attr_name.startswith('__') and not callable(getattr(self, attr_name)):
                member = getattr(self, attr_name)
                if isinstance(member, np.ndarray):
                    # Ensure the slice doesn't exceed the bounds of rap_self.ib
                    end_index = min(i_end, len(getattr(self, key_attr)))

                    # Assign the slice to the numpy.ndarray member
                    # If the target array has a different shape, direct assignment
                    # might fail or reshape the array. Using np.array() ensures
                    # a new array is created with the correct slice.
                    setattr(self, attr_name, getattr(self, attr_name)[:end_index])

    def __str__(self):
        s = "{},".format(self.unit[self.i])
        s += "{},".format(self.hm[self.i])
        # s += "{:13.3f},".format(self.cTime[self.i])
        s += "{:8.3f},".format(self.ib[self.i])
        s += "{:7.2f},".format(self.vsat[self.i])
        s += "{:5.2f},".format(self.dv_dyn[self.i])
        s += "{:5.2f},".format(self.voc_stat[self.i])
        s += "{:5.2f},".format(self.voc_ekf[self.i])
        s += "{:10.6f},".format(self.y_ekf[self.i])
        s += "{:7.3f},".format(self.soc_s[self.i])
        s += "{:5.3f},".format(self.soc_ekf[self.i])
        s += "{:5.3f},".format(self.soc[self.i])
        return s

    def mod(self):
        return self.mod_data[self.zero_end]


class SavedDataSim:
    def __init__(self, time_run, data=None, time_end=None, fake=False, mon_for_fake=None, str_=None):
        self.str = str_
        if data is None:
            self.skip_s = None
            self.i = 0
            self.time = None
            self.time_min = None
            self.time_day = None
            self.unit = None  # text title
            self.cTime = None  # Control time, s
            self.dt_s = None
            self.chm_s = None
            self.qcrs_s = None  # Unit capacity rated scaled, Coulombs
            self.qcap_s = None  # Unit capacity rated scaled, Coulombs
            self.bms_off_s = None
            self.nS_s = None
            self.Tb_f_s = None
            self.vsat_s = None
            self.voc_s = None
            self.voc_stat_s = None
            self.dv_dyn_s = None
            self.dv_hys_s = None
            self.vb_s = None
            self.ib_in_s = None
            self.ib_charge_s = None
            self.ioc_s = None
            self.ib_s = None
            self.ib_dyn_s = None
            self.sat_s = None
            self.delta_q_s = None
            self.soc_s = None
            self.reset_s = None
            self.d_delta_q_s = None
            self.ib_dyn_s_T = None
            self.ib_dyn_s_tau = None
            self.ib_dyn_s_rstate = None
            self.ib_dyn_s_lstate = None
            self.bms_off_s = False
            self.voltage_low_s = False
        else:
            self.i = 0
            self.cTime = np.array(data.c_time)
            self.time = np.array(data.c_time) - time_run
            # Truncate
            if time_end is None:
                i_end = len(self.time)
            else:
                i_end = np.where(self.time <= time_end)[0][-1] + 1
            self.cTime = self.cTime[:i_end]
            self.time = self.time[:i_end]
            self.skip_s =  np.bool(data.skip[:i_end])
            self.dt_s = data.dt_s[:i_end]
            self.time_min = self.time / 60.
            self.time_day = self.time / 3600. / 24.

            self.chm_s = data.chm_s[:i_end]
            if hasattr(data, 'qcrs_s'):
                self.qcrs_s = data.qcrs_s[:i_end]
            if hasattr(data, 'nS_s'):
                self.nS_s = np.array(data.nS_s[:i_end])
            else:
                self.nS_s = np.array(data.bmso_s[:i_end]) * 0 + 1
            self.Tb_f_s = data.Tb_f_s[:i_end]
            self.vb_s = data.vb_s[:i_end]
            self.vsat_s = data.vsat_s[:i_end]
            self.voc_stat_s = data.voc_stat_s[:i_end]
            self.dv_dyn_s = data.dv_dyn_s[:i_end]
            self.voc_s = self.vb_s - self.dv_dyn_s
            self.dv_hys_s = data.dv_hys_s[:i_end]
            self.ib_s = data.ib_s[:i_end]
            self.ib_dyn_s = data.ib_dyn_s[:i_end]
            self.ib_in_s = data.ib_in_s[:i_end]
            self.ib_charge_s = data.ib_charge_s[:i_end]
            self.ioc_s = data.ioc_s[:i_end]
            self.sat_s = data.sat_s[:i_end]
            self.delta_q_s = data.dq_s[:i_end]
            self.qcap_s = data.q_cap_s[:i_end]
            self.soc_s = data.soc_s[:i_end]
            self.reset_s = data.reset_s[:i_end]
            self.d_delta_q_s = data.ddq_s[:i_end]
            self.ib_dyn_s_T = data.ib_dyn_s_T[:i_end]
            self.ib_dyn_s_tau = data.ib_dyn_s_tau[:i_end]
            self.ib_dyn_s_rstate = data.ib_dyn_s_rstate[:i_end]
            self.ib_dyn_s_lstate = data.ib_dyn_s_lstate[:i_end]
            self.bms_off_s = np.bool(np.array(data.bmso_s[:i_end]))
            self.voltage_low_s = np.bool(np.array(data.vlow_s[:i_end]))

        if fake:
            self.ib_in_s = np.copy(mon_for_fake.ib)
            self.ib_dyn_s = np.copy(mon_for_fake.ib_dyn)
            self.time = np.copy(mon_for_fake.time)
            self.dv_dyn_s = np.copy(mon_for_fake.dv_dyn)
            self.dv_hys_s = np.copy(mon_for_fake.dv_hys)
            self.Tb_hdwe = np.copy(mon_for_fake.Tb_rap)
            self.delta_q_s = np.copy(mon_for_fake.delta_q)
            self.delta_q_s = np.copy(mon_for_fake.delta_q)
            self.voc_stat_s = np.copy(mon_for_fake.voc_stat)
            self.qcrs_s = np.copy(mon_for_fake.qcrs)
            self.chm_s = np.copy(mon_for_fake.chm)
            self.sat_s = np.copy(mon_for_fake.sat)
            self.soc_s = np.copy(mon_for_fake.soc_s)
            self.dt_s = np.copy(mon_for_fake.dt)
            self.bms_off_s = np.copy(mon_for_fake.bms_off)
            self.mod_tb = np.bool(np.copy(mon_for_fake.mod_data))

    def __str__(self):
        s = "{},".format(self.unit[self.i])
        # s += "{:13.3f},".format(self.cTime[self.i])
        # s += "{:5.2f},".format(self.Tb_s[self.i])
        s += "{:8.3f},".format(self.vsat_s[self.i])
        s += "{:5.2f},".format(self.voc_stat_s[self.i])
        s += "{:5.2f},".format(self.dv_dyn_s[self.i])
        s += "{:5.2f},".format(self.vb_s[self.i])
        s += "{:8.3f},".format(self.ib_s[self.i])
        s += "{:8.3f},".format(self.ib_dyn_s[self.i])
        s += "{:7.3f},".format(self.sat_s[self.i])
        # s += "{:5.3f},".format(self.ddq_s[self.i])
        s += "{:5.3f},".format(self.delta_q_s[self.i])
        # s += "{:5.3f},".format(self.qcap_s[self.i])
        s += "{:7.3f},".format(self.soc_s[self.i])
        s += "{:d},".format(self.reset_s[self.i])
        return s


if __name__ == '__main__':
    import sys
    import doctest

    doctest.testmod(sys.modules['__main__'])
    if sys.platform == 'darwin':
        import matplotlib
        matplotlib.use('tkagg')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['legend.fontsize'] = 'small'


    def main(data_file_old_txt, unit_key):
        from MonSim import replicate, save_clean_file, save_clean_file_sim
        # Trade study inputs
        # i-->0 provides continuous anchor to reset filter (why?)  i shifts important --> 2 current sensors,
        #   hyst in ekf
        # saturation provides periodic anchor to reset filter
        # reset soc periodically anchor user display
        # tau_sd creating an anchor.   So large it's just a pass through
        # TODO:  temp sensitivities and mitigation

        # Config inputs
        # from MonSimNomConfig import *

        # Transient  inputs
        time_end = None
        zero_zero_in = False
        # time_end = 1500.

        # Load data (must end in .txt) txt_file, type, hdr_key, unit_key
        data_file_clean = write_clean_file(data_file_old_txt, type_='_mon', hdr_key='unit,',
                                           unit_key=unit_key)
        data_file_sim_clean = write_clean_file(data_file_old_txt, type_='_sim', hdr_key='unit_m',
                                               unit_key='unit_sim,')

        # Load
        mon_run_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
        mon_run = SavedData(mon_run_raw, time_end, zero_zero=zero_zero_in, str_='')
        try:
            sim_run_raw = np.genfromtxt(data_file_sim_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
            sim_run = SavedDataSim(mon_run.time_run, sim_run_raw, time_end, str_='_s')
        except IOError:
            sim_run = None

        # Run model
        from MonSim import UserOptions
        replicateOptions = UserOptions(mon_run=mon_run, init_time=1.)
        mon_ver, sim_ver, sim_s_ver = replicate(replicateOptions)
        date_ = datetime.now().strftime("%y%m%d")
        mon_file_save = data_file_clean.replace(".csv", "_rep.csv")
        save_clean_file(mon_ver, mon_file_save, '_mon_rep' + date_)
        if data_file_sim_clean:
            sim_file_save = data_file_sim_clean.replace(".csv", "_rep.csv")
            save_clean_file_sim(sim_s_ver, sim_file_save, '_sim_rep' + date_)

        # Plots
        fig_list = []
        fig_files = []
        date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = data_file_clean.split('/')[-1].replace('.csv', '-') + os.path.split(__file__)[1].split('.')[0]
        plot_title = filename + '   ' + date_time
        fig_list, fig_files = overall_batt(mon_ver, sim_ver, filename, fig_files, plot_title=plot_title,
                                           fig_list=fig_list, suffix='_ver')  # Could be confusing because sim over mon
        fig_list, fig_files = dom_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_ver, filename, fig_files,
                                       plot_title=plot_title, fig_list=fig_list, run_str='', ver_str='_ver')
        # fig_list, fig_files = tune_r(mon_run, mon_ver, sim_s_ver, filename, fig_files,
        #                           plot_title=plot_title, fig_list=fig_list, run_str='', ver_str='_ver')
        unite_pictures_into_pdf(outputPdfName=filename+'_'+date_time+'.pdf',
                                save_pdf_path='../dataReduction/figures')
        cleanup_fig_files(fig_files)

        plt.show()


    # python DataOverModel.py("../dataReduction/rapidTweakRegressionTest20220711.txt", "pro_2022")

    """
    PyCharm Sample Run Configuration Parameters (right click in pyCharm - Modify Run Configuration:
        "../dataReduction/slowTweakRegressionTest20220711.txt" "pro_2022"
        "../dataReduction/serial_20220624_095543.txt"    "pro_2022"
        "../dataReduction/real world rapid 20220713.txt" "soc0_2022"
        "../dataReduction/real world Xp20 20220715.txt" "soc0_2022"
    
    PyCharm Terminal:
    python DataOverModel.py "../dataReduction/serial_20220624_095543.txt" "pro_2022"
    python DataOverModel.py "../dataReduction/ampHiFail20220731.txt" "pro_2022"
    

    android:
    python Python/DataOverModel.py "USBTerminal/serial_20220624_095543.txt" "pro_2022"
    """

    if __name__ == "__main__":
        import sys
        print(sys.argv[1:])
        main(sys.argv[1], sys.argv[2])
