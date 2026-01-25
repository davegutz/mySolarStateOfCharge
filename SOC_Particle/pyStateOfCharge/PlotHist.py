# PlotHist - general trend plotting from Hd Data
# Copyright (C) 2025 Dave Gutz
#
# This library is free software; you can redistribute it and/or
# mrdify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for mrre details.
#
# See http://www.fsf.org/licensing/licenses/lgpl.txt for full license text.

"""Plot battery trends, mainly voc(soc) table, faults (esp wrap logic), EKF initialization.
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
from myFilters import InlineExpLag
from DataOverModel import plq
# below suppresses runtime error display******************
# import os
# os.environ["KIVY_NO_CONSOLELOG"] = "1"
# from kivy.utils import platform  # failed experiment to run BLE data plotting realtime on android
# if platform != 'linux':
#     from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')

plt.rcParams.update({'figure.max_open_warning': 0})


def hs_plot(mr, mv, sr, sv, smv, filename, fig_files=None, plot_title=None, fig_list=None,
            run_str='_run', ver_str='_ver'):
    fig_list.append(plt.figure())  # HS 1
    plt.subplot(221)
    plt.title(plot_title + ' HS 1')
    print('HS 1', end=':  ')
    plq(plt, mr, 'time', mr, 'vb_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'voc_d', color='blue', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'voc', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'voc_stat_f', color='green', linestyle='-.')
    plq(plt, mv, 'time', mv, 'voc_stat', color='cyan', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, mr, 'time', mr, 'soc', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, mr, 'time', mr, 'ib_amp_hdwe_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_amp_hdwe', color='red', linestyle='-.', warn=False)
    plq(plt, mr, 'time', mr, 'ib_noa_hdwe_f', color='green', linestyle='--')
    plq(plt, mv, 'time', mv, 'ib_noa_hdwe', color='blue', linestyle=':', warn=False)
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, mr, 'time_t', mr, 'Tb_f', color='black', linestyle='--')
    plq(plt, mv, 'time', mv, 'Tb_f', color='orange', linestyle=':')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # HS sat
    plt.subplot(321)
    plt.title(plot_title + 'HS sat')
    print('HS sat', end=':  ')
    plq(plt, mr, 'time', mr, 'sat', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'sat', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(322)
    plq(plt, mr, 'time', mr, 'voc_d', color='black', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'voc', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'vsat', color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'vsat', color='red', linestyle=':')
    plq(plt, mr, 'time', mr, 'voc_soc', color='cyan', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_soc', color='black', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(323)
    plq(plt, mr, 'time', mr, 'soc', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(324)
    plq(plt, mr, 'time', mr, 'ib_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(325)
    plq(plt, mr, 'soc', mr, 'voc_d', color='black', linestyle='-', warn=False)
    plq(plt, mr, 'soc', mr, 'voc_soc', color='red', linestyle='-')
    plq(plt, mv, 'soc', mv, 'voc_soc', color='orange', linestyle='--')
    mr.dv = np.array(mr.voc_soc) - np.array(mr.voc_f)
    plq(plt, mr, 'soc', mr, 'dv', add=13., color='blue', linestyle='-')
    mv.dv = np.array(mv.voc_soc) - np.array(mv.voc)
    plq(plt, mv, 'soc', mv, 'dv', add=13., color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(326)
    plq(plt, mr, 'time', mr, 'voc_d', color='black', linestyle='-', warn=False)
    plq(plt, mr, 'time', mr, 'voc_soc', color='red', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_soc', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'dv', add=13., color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv', add=13., color='orange', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # HS 3 Tune
    plt.subplot(331)
    plt.title(plot_title + ' HS 3 Tune')
    print('HS 3 Tune', end=':  ')
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='blue', linestyle='-')
    plt.plot(mv.time, mv.dv_dyn, color='cyan', linestyle='--', label='dv_dyn' + ver_str)
    plq(plt, sr, 'time', sr, 'dv_dyn_s', color='black', linestyle='-.')
    plq(plt, smv, 'time', smv, 'dv_dyn_s', color='magenta', linestyle=':')
    plt.plot(mr.time, mr.dv_hys, color='pink', linestyle='-', label='dv_hys' + run_str)
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(332)
    plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='cyan', linestyle='--')
    plq(plt, sr, 'time', sr, 'soc_s', color='black', linestyle='-.')
    plq(plt, smv, 'time', smv, 'soc_s', color='magenta', linestyle=':')
    plq(plt, mr, 'time', mr, 'soc_ekf', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc_ekf', color='red', linestyle='--')
    plt.xlabel('sec')
    plt.legend(loc=4)
    plt.subplot(333)
    plq(plt, mr, 'time', mr, 'ib_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib', color='orange', linestyle='--')
    plq(plt, smv, 'time', smv, 'ib_in_s', color='red', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(334)
    plq(plt, mr, 'time', mr, 'voc_d', color='blue', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'voc', color='cyan', linestyle='--')
    plq(plt, mr, 'time', mr, 'voc_stat_f', color='orange', linestyle='-')
    plq(plt, smv, 'time', smv, 'voc_stat_s', color='red', linestyle=':')
    plq(plt, mr, 'time', mr, 'vb_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb_s', color='pink', linestyle='--')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(335)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'e_wrap', color='orange', linestyle='--')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(336)
    plq(plt, mr, 'soc', mr, 'vb_f', color='blue', linestyle='-')
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='cyan', linestyle='--')
    plq(plt, mr, 'soc', mr, 'voc_stat_f', color='orange', linestyle='-.')
    plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='red', linestyle=':')
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(337)
    plq(plt, mr, 'time', mr, 'vb_f', color='blue', linestyle='-')
    plt.plot(mv.time, mv.vb, color='cyan', linestyle='--', label='vb' + ver_str)
    plq(plt, sr, 'time', sr, 'vb_s', color='black', linestyle='-.')
    plq(plt, smv, 'time', smv, 'vb_s', color='magenta', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(338)
    plq(plt, mr, 'time', mr, 'dv_hys', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys', color='cyan', linestyle='--')
    plq(plt, sr, 'time', sr, 'dv_hys_s', color='black', linestyle='-.')
    plq(plt, smv, 'time', smv, 'dv_hys_s', color='magenta', linestyle=':')
    plq(plt, mr, 'time', mr, 'sat', add=-0.5, color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'sat', add=-0.5, color='green', linestyle='--')
    plq(plt, sr, 'time', sr, 'sat_s', add=-0.5, color='red', linestyle='-.')
    plq(plt, sv, 'time', sv, 'sat', add=-0.5, color='cyan', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(339)
    plq(plt, mr, 'time_t', mr, 'Tb_f', color='cyan', linestyle='--', stairs=True)
    plq(plt, mv, 'time', mv, 'Tb_f', color='magenta', linestyle=':')
    plt.legend(loc=3)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def tune_hs(mr, mv, smv, filename, fig_files=None, plot_title=None, fig_list=None, run_str='_run', ver_str='_ver'):
    # delineate charging and discharging
    voc_stat_chg = np.copy(mv.voc_stat)
    voc_stat_dis = np.copy(mv.voc_stat)
    if hasattr(smv, 'ib_in_s'):
        for i in range(len(voc_stat_chg)):
            if smv.ib_in_s[i] > -0.1:
                voc_stat_dis[i] = None
            elif smv.ib_in_s[i] < 0.1:
                voc_stat_chg[i] = None

    vb = np.copy(mv.vb)
    voc = np.copy(mv.voc)
    voc_soc = np.copy(mv.voc_soc)
    voc_stat = np.copy(mv.voc_stat)
    ib_f = np.copy(mv.ib)
    ioc_f = None
    if hasattr(mv, 'ioc'):
        ioc_f = np.copy(mv.ioc)
    # to = np.copy(mr.time)
    tv = np.copy(mv.time)
    dv_hys_calc = voc - voc_stat  # assumes Charge Transfer tuned
    dv_hys_req = voc - voc_soc
    dv_hys_calc_f = np.copy(dv_hys_calc)
    dv_hys_req_f = np.copy(dv_hys_req)
    dv_dot_calc = np.copy(dv_hys_calc)
    dv_dot_req = np.copy(dv_hys_req)
    dv_dot_cap = np.copy(dv_hys_calc)
    dv_bleed = np.copy(dv_hys_calc)
    ioc_req = np.copy(dv_hys_req)
    r_calc = np.copy(dv_hys_req)
    r_calc_from_dot = np.copy(dv_hys_req)
    r_req = np.copy(dv_hys_req)
    ioc_calc_from_dot = np.copy(dv_hys_req)

    tau = 20
    cap = 1000
    n = len(dv_hys_req)
    dv_hys_calc_filter = InlineExpLag(tau)
    dv_hys_req_filter = InlineExpLag(tau)
    ib_filter = InlineExpLag(tau)
    ios_filter = InlineExpLag(tau)
    dv_hys_dot_filter = InlineExpLag(tau)
    for i in range(n-1):
        reset = i == 0
        T = tv[i+1]-tv[i]

        dv_hys_calc_f[i] = dv_hys_calc_filter.update(dv_hys_calc[i], T, reset=reset)
        dv_hys_req_f[i] = dv_hys_req_filter.update(dv_hys_req[i], T, reset=reset)
        ib_f[i] = ib_filter.update(mv.ib[i], T, reset=reset)
        if hasattr(mv, 'ioc'):
            ioc_f[i] = ios_filter.update(mv.ioc[i], T, reset=reset)

        dv_dot_calc[i] = dv_hys_calc_filter.rate
        # dv_dot_req[i] = dv_hys_req_filter.rate
        dv_dot_req[i] = dv_hys_dot_filter.update(dv_hys_req_filter.rate, T, reset=reset)
        dv_dot_cap[i] = ib_f[i] / cap
        dv_bleed[i] = dv_dot_cap[i] - dv_dot_req[i]

        ioc_req[i] = dv_bleed[i] * cap
        if abs(ib_f[i]) < 0.5:
            r_req[i] = 0
        else:
            # noinspection PyTypeChecker
            r_req[i] = max(min(dv_hys_req_f[i] / ioc_req[i], 0.1), -0.1)

        ioc_calc_from_dot[i] = ib_f[i] - cap*dv_dot_calc[i]
        if abs(ioc_calc_from_dot[i]) > 1e-9:
            # noinspection PyTypeChecker
            r_calc_from_dot[i] = max(min(dv_hys_calc_f[i] / ioc_calc_from_dot[i], 0.1), -0.1)
        else:
            r_calc_from_dot[i] = 0.
        # ioc_calc_from_dot[i] = mv.ib[i] - cap*dv_dot_calc[i]
        # r_calc_from_dot[i] = max(min(dv_hys_calc[i] / ioc_calc_from_dot[i], 0.1), -0.1)

        if hasattr(mv, 'ioc'):
            if abs(ioc_f[i]) < .5:
                r_calc[i] = 0
            else:
                # noinspection PyTypeChecker
                r_calc[i] = max(min(dv_hys_calc_f[i] / ioc_f[i], 0.1), -0.1)

    dv_dot_calc[n-1] = dv_dot_calc[n-2]
    dv_dot_req[n-1] = dv_dot_req[n-2]
    if hasattr(mv, 'ioc'):
        r_calc[n-1] = r_calc[n-2]
    r_calc_from_dot[n-1] = r_calc_from_dot[n-2]
    ioc_req[n-1] = ioc_req[n-2]
    dv_hys_req_f[n-1] = dv_hys_req_f[n-2]
    dv_hys_calc_f[n-1] = dv_hys_calc_f[n-2]
    ioc_calc_from_dot[n-1] = ioc_calc_from_dot[n-2]
    ib_f[n-1] = ib_f[n-2]
    if hasattr(mv, 'ioc'):
        ioc_f[n-1] = ioc_f[n-2]
    dv_dot_cap[n-1] = dv_dot_cap[n-2]
    dv_bleed[-1] = dv_bleed[-2]

    fig_list.append(plt.figure())  # HS 3 Tune R
    plt.subplot(321)
    plt.title(plot_title + ' HS 3 Tune R')
    print('HS 3 Tune R', end=':  ')
    plt.plot(tv, vb, color='blue', linestyle='-', label='vb_x')
    if hasattr(smv, 'vb_s'):
        plt.plot(tv, smv.vb_s, color='cyan', linestyle='--', label='vb_s_ver')
    plt.plot(tv, voc, color='magenta', linestyle='-.', label='voc_x')
    plt.plot(tv, voc_soc, color='black', linestyle=':', label='voc_soc_x')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(322)
    plq(plt, mr, 'time', mr, 'dv_dyn', color='red', linestyle='-')
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='red', linestyle='-')
    plt.plot(tv, mv.dv_dyn, color='black', linestyle='-', label='dv_dyn_ver')
    plt.plot(tv, dv_hys_calc, color='blue', linestyle='-', label='dv_hys_calc_x')
    plt.plot(tv, dv_hys_req, color='magenta', linestyle='--', label='dv_hys_req_x')
    plt.plot(tv, dv_hys_calc_f, color='red', linestyle='-', label='dv_hys_calc_f_x')
    plt.plot(tv, dv_hys_req_f, color='cyan', linestyle='--', label='dv_hys_req_f_x')
    plt.plot(tv, mv.dv_hys, color='orange', linestyle=':', label='dv_hys_x')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(323)
    plt.plot(tv, r_calc_from_dot, color='cyan', linestyle='--', label='r_calc_from_dot_x')
    if hasattr(mv, 'ioc'):
        plt.plot(tv, r_calc, color='blue', linestyle='-', label='r_calc_x')
    plt.plot(tv, r_req, color='black', linestyle='-', label='r_req_x')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(324)
    plt.plot(tv, ioc_calc_from_dot, color='orange', linestyle='--', label='ioc_calc_from_dot_x')
    plt.plot(tv, mv.ib, color='blue', linestyle='-', label='ib_x')
    plt.plot(tv, ib_f, color='cyan', linestyle='--', label='ib_f_x')
    if hasattr(mv, 'ioc'):
        plt.plot(tv, mv.ioc, color='red', linestyle='-', label='ioc_x')
        plt.plot(tv, ioc_f, color='pink', linestyle='--', label='ioc_f_x')
    plt.plot(tv, ioc_req, color='magenta', linestyle=':', label='ioc_req_x')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(325)
    plt.plot(tv, ioc_req, color='magenta', linestyle='--', label='ioc_req_x')
    plt.plot(tv, mv.ib, color='blue', linestyle='-', label='ib_x')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(326)
    plt.plot(tv, dv_dot_req, color='magenta', linestyle='-', label='dv_dot_req_x')
    plt.plot(tv, dv_dot_cap, color='black', linestyle='--', label='dv_dot_cap_x')
    plt.plot(tv, dv_dot_calc, color='blue', linestyle='-.', label='dv_dot_calc_x')
    plt.xlabel('sec')
    plt.legend(loc=2)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # HS 3 Tune Summ
    plt.subplot(221)
    plt.title(plot_title + ' HS 3 Tune Summ')
    print('HS 3 Tune Summ', end=':  ')
    plq(plt, mr, 'time', mr, 'vb', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'vb_f', color='blue', linestyle='-')
    plq(plt, smv, 'time', smv, 'vb_s', color='magenta', linestyle=':')
    plq(plt, smv, 'time', smv, 'voc_stat_s', color='black', linestyle='-.')
    plt.plot(mv.time, voc_stat_chg, linestyle=':', color='green', label='voc_stat_chg' + ver_str)
    plt.plot(mv.time, voc_stat_dis, linestyle=':', color='red', label='voc_stat_dis' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(222)
    plq(plt, mr, 'soc', mr, 'vb', color='blue', linestyle='-')
    plq(plt, mr, 'soc', mr, 'vb_f', color='blue', linestyle='-')
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='magenta', linestyle=':')
    plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='black', linestyle='-.')
    plt.plot(mv.soc, voc_stat_chg, linestyle=':', color='green', label='voc_stat_chg' + ver_str)
    plt.plot(mv.soc, voc_stat_dis, linestyle=':', color='red', label='voc_stat_dis' + ver_str)
    plt.plot(mr.soc, mr.voc_soc, color='cyan', linestyle='--', label='voc_soc' + run_str)
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(223)
    plq(plt, mr, 'time_t', mr, 'Tb', color='red', linestyle='-', stairs=True)
    plt.plot(mr.time, mr.ib_sel, linestyle='--', color='blue', label='ib_sel' + run_str)
    plq(plt, smv, 'time', smv, 'ib_in_s', color='magenta', linestyle='-.')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(224)
    plq(plt, smv, 'time', smv, 'dv_dyn_s', color='black', linestyle=':')
    plq(plt, smv, 'time', smv, 'dv_hys_s', color='magenta', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")


    return fig_list, fig_files
