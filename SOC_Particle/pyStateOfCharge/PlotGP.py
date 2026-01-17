# PlotGP - general purpose plotting
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


def gp_plot(mr, mv, sr, sv, smv, filename, fig_files=None, plot_title=None, fig_list=None,
            run_str='_run', ver_str='_ver', Battery=None):
    print('gp_plot', end=':  ')
    fig_list.append(plt.figure())  # GP 1
    plt.subplot(221)
    plt.title(plot_title + ' GP 1')
    print('GP 1', end=':  ')
    plq(plt, sr, 'time', sr, 'vb_s', color='black', linestyle='-', label='vb_s' + run_str)
    plq(plt, smv, 'time', smv, 'vb_s', color='orange', linestyle='--', label='vb_s' + ver_str)
    plq(plt, sr, 'time', sr, 'voc_s', color='blue', linestyle='-.', label='voc_s' + run_str)
    plq(plt, smv, 'time', smv, 'voc_s', color='red', linestyle=':', label='voc_s' + ver_str)
    plq(plt, sr, 'time', sr, 'voc_stat_s', color='magenta', linestyle='-.', label='voc_stat_s' + run_str)
    plq(plt, smv, 'time', smv, 'voc_stat_s', color='green', linestyle=':', label='voc_stat_s' + ver_str)
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, sr, 'time', sr, 'dv_hys_s', linestyle='-', color='black', label='dv_hys_s' + run_str)
    plq(plt, smv, 'time', smv, 'dv_hys_s', linestyle='--', color='orange', label='dv_hys_s' + ver_str)
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, sr, 'time', sr, 'soc_s', linestyle='-', color='black', label='soc_s' + run_str)
    plq(plt, smv, 'time', smv, 'soc_s', linestyle='--', color='orange', label='soc_s' + ver_str)
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, sr, 'time', sr, 'ib_in_s', linestyle='-', color='blue', label='ib_in_s' + run_str)
    plq(plt, smv, 'time', smv, 'ib_in_s', linestyle='--', color='red', label='ib_in_s' + ver_str)
    plq(plt, smv, 'time', smv, 'ib_fut_s', linestyle='-.', color='orange', label='ib_fut_s' + ver_str)
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # GP 2
    plt.subplot(221)
    plt.title(plot_title + ' GP 2')
    print('GP 2', end=':  ')
    plq(plt, mr, 'time', mr, 'vb', linestyle='-', color='black', label='vb' + run_str)
    plq(plt, mr, 'time', mr, 'vb_f', linestyle='-', color='black', label='vb_f' + run_str, warn=False)
    plt.plot(mv.time, mv.vb, color='orange', linestyle='--', label='vb' + ver_str)
    plq(plt, mr, 'time', mr, 'voc', linestyle='-', color='blue', label='voc' + run_str)
    plq(plt, mr, 'time', mr, 'voc_d', linestyle='-', color='blue', label='voc_d' + run_str, warn=False)
    plt.plot(mv.time, mv.voc, color='red', linestyle='--', label='voc' + ver_str)
    plq(plt, mr, 'time', mr, 'voc_stat', linestyle='-.', color='cyan', label='voc_stat' + run_str)
    plq(plt, mr, 'time', mr, 'voc_stat_f', linestyle='-.', color='cyan', label='voc_stat_f' + run_str, warn=False)
    plt.plot(mv.time, mv.voc_stat, color='black', linestyle=':', label='voc_stat' + ver_str)
    plt.legend(loc=1)
    plt.subplot(222)
    plt.plot(mr.time, mr.dv_hys, linestyle='-', color='black', label='dv_hys' + run_str)
    plt.plot(mv.time, mv.dv_hys, linestyle='--', color='orange', label='dv_hys' + ver_str)
    plt.legend(loc=1)
    plt.subplot(223)
    plt.plot(mr.time, mr.soc, linestyle='-', color='black', label='soc' + run_str)
    plt.plot(mv.time, mv.soc, linestyle='--', color='orange', label='soc' + ver_str)
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, mr, 'time', mr, 'ib_sel', linestyle='-', color='black', label='ib_sel' + run_str)
    plq(plt, sr, 'time', sr, 'ib_in_s', linestyle='--', color='cyan', label='ib_in_s' + run_str)
    plt.plot(mv.time, mv.ib_charge, linestyle='-.', color='orange', label='ib_charge' + ver_str)
    plq(plt, mr, 'time', mr, 'ib_diff', linestyle=':', color='red', label='ib_diff' + run_str)
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # GP 2 nn lag
    plt.subplot(321)
    plt.title(plot_title + ' GP 2 nn lag')
    print('GP 2 nn lag', end=':  ')
    plt.plot(mr.time, mr.sat, color='black', linestyle='-', label='sat' + run_str)
    plt.plot(mv.time, mv.sat, color='orange', linestyle='--', label='sat' + ver_str)
    plt.legend(loc=1)
    plt.subplot(322)
    plq(plt, mr, 'time', mr, 'voc', linestyle='-', color='black', label='voc' + run_str)
    plq(plt, mr, 'time', mr, 'voc_d', linestyle='-', color='black', label='voc_d' + run_str, warn=False)
    plt.plot(mv.time, mv.voc, color='orange', linestyle='--', label='voc' + ver_str)
    plt.plot(mr.time, mr.vsat, color='blue', linestyle='-.', label='vsat' + run_str)
    plt.plot(mv.time, mv.vsat, color='red', linestyle=':', label='vsat' + ver_str)
    plt.plot(mr.time, mr.voc_soc, color='cyan', linestyle='-', label='voc_soc' + run_str)
    plt.plot(mv.time, mv.voc_soc, color='black', linestyle='--', label='voc_soc' + ver_str)
    plt.legend(loc=1)
    plt.subplot(323)
    plt.plot(mr.time, mr.soc, color='black', linestyle='-', label='soc' + run_str)
    plt.plot(mv.time, mv.soc, color='orange', linestyle='--', label='soc' + ver_str)
    plt.legend(loc=1)
    plt.subplot(324)
    plq(plt, mr, 'time', mr, 'ib', add=10., linestyle='-', color='black', label='ib+10' + run_str)
    plq(plt, mr, 'time', mr, 'ib_f', add=10., linestyle='-', color='black', label='ib_f+10' + run_str, warn=False)
    plt.plot(mv.time, np.array(mv.ib)+10., color='orange', linestyle='--', label='ib+10' + ver_str)
    plt.plot(mr.time, mr.ib_lag, color='blue', linestyle='-', label='ib_lag' + run_str)
    plt.plot(mv.time, mv.ib_lag, color='red', linestyle='--', label='ib_lag' + ver_str)
    plt.legend(loc=1)
    plt.subplot(325)
    plq(plt, mr, 'soc', mr, 'voc', linestyle='-', color='black', label='voc' + run_str)
    plq(plt, mr, 'soc', mr, 'voc_d', linestyle='-', color='black', label='voc_d' + run_str, warn=False)
    plt.plot(mr.soc, mr.voc_soc, color='red', linestyle='-', label='voc_soc' + run_str)
    plt.plot(mv.soc, mv.voc_soc, color='orange', linestyle='--', label='voc_soc' + ver_str)
    if hasattr(mr, 'voc'):
        values = np.array(mr.voc_soc) - np.array(mr.voc)+13.
    else:
        values = np.array(mr.voc_soc) - np.array(mr.voc_d)+13.
    plt.plot(mr.soc, values, color='blue', linestyle='-', label='dv' + run_str + '+13')
    plq(plt, mr, 'soc', mr, 'voc', linestyle='-', color='black', label='voc' + run_str)

    plt.plot(mv.soc, np.array(mv.voc_soc) - np.array(mv.voc)+13., color='orange', linestyle='--',
             label='dv' + ver_str + '+13')
    plt.legend(loc=1)
    plt.subplot(326)
    plq(plt, mr, 'time', mr, 'voc', linestyle='-', color='black', label='voc' + run_str)
    plq(plt, mr, 'time', mr, 'voc_d', linestyle='-', color='black', label='voc_d' + run_str, warn=False)
    plt.plot(mr.time, mr.voc_soc, color='red', linestyle='-', label='voc_soc' + run_str)
    plt.plot(mv.time, mv.voc_soc, color='orange', linestyle='--', label='voc_soc' + ver_str)
    plt.plot(mr.time, values, color='blue', linestyle='-', label='dv' + run_str + '+13')
    plt.plot(mv.time, np.array(mv.voc_soc) - np.array(mv.voc)+13., color='orange', linestyle='--',
             label='dv' + ver_str + '+13')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # GP 3 Tune
    plt.subplot(331)
    plt.title(plot_title + ' GP 3 Tune')
    print('GP 3 Tune', end=':  ')
    plq(plt, mr, 'time', mr, 'dv_dyn', color='blue', linestyle='-', label='dv_dyn' + run_str)
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='blue', linestyle='-', label='dv_dyn_f' + run_str, warn=False)
    plt.plot(mv.time, mv.dv_dyn, color='cyan', linestyle='--', label='dv_dyn' + ver_str)
    plq(plt, sr, 'time', sr, 'dv_dyn_s', color='black', linestyle='-.', label='dv_dyn_s' + run_str)
    plq(plt, smv, 'time', smv, 'dv_dyn_s', color='magenta', linestyle=':', label='dv_dyn_s' + ver_str)
    plt.plot(mr.time, mr.dv_hys, color='pink', linestyle='-', label='dv_hys' + run_str)
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(332)
    plt.plot(mr.time, mr.soc, linestyle='-', color='blue', label='soc' + run_str)
    plt.plot(mv.time, mv.soc, linestyle='--', color='cyan', label='soc' + ver_str)
    plq(plt, sr, 'time', sr, 'soc_s', linestyle='-.', color='black', label='soc_s' + run_str)
    plq(plt, smv, 'time', smv, 'soc_s', linestyle=':', color='magenta', label='soc_s' + ver_str)
    plt.plot(mr.time, mr.soc_ekf, linestyle='-', color='green', label='soc_ekf' + run_str)
    plt.plot(mv.time, mv.soc_ekf, linestyle='--', color='red', label='soc_ekf' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=4)
    plt.subplot(333)
    plq(plt, mr, 'time', mr, 'ibmh', linestyle='-', color='blue', label='ib_amp_hdwe' + run_str)
    plq(plt, mr, 'time', mr, 'ibmm', linestyle='--', color='red', label='ib_amp_model' + run_str)
    plq(plt, mr, 'time', mr, 'ibmkf', linestyle='--', color='black', label='ib_amp_hdwe_kf' + run_str)
    plq(plt, mr, 'time', mr, 'ibnh', linestyle='-.', color='cyan', label='ib_noa_hdwe' + run_str)
    plq(plt, mr, 'time', mr, 'ibnm', linestyle=':', color='magenta', label='ib_noa_model' + run_str)
    plq(plt, mr, 'time', mr, 'ibnkf', linestyle='--', color='black', label='ib_noa_hdwe_kf' + run_str)
    plq(plt, mr, 'time', mr, 'ib_sel', add=-5, linestyle='-', color='blue', label='ib_sel-5' + run_str)
    plq(plt, sr, 'time', sr, 'ib_in_s', add=-5, linestyle='--', color='black', label='ib_in_s-5' + run_str)
    plq(plt, smv, 'time', smv, 'ib_in_s', add=-5, linestyle=':', color='red', label='ib_in_s-5' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(334)
    plq(plt, mr, 'time', mr, 'voc', linestyle='-', color='blue', label='voc' + run_str)
    plq(plt, mr, 'time', mr, 'voc_d', linestyle='-', color='blue', label='voc_d' + run_str, warn=False)
    plt.plot(mv.time, mv.voc, linestyle='--', color='cyan', label='voc' + ver_str)
    plq(plt, mr, 'time', mr, 'voc_stat', add=-1., linestyle='-', color='orange', label='voc_stat' + run_str + '-1')
    plq(plt, mr, 'time', mr, 'voc_stat_f', add=-1., linestyle='-', color='orange', label='voc_stat_f' + run_str + '-1', warn=False)
    plq(plt, sr, 'time', sr, 'voc_stat_s', add=-1., linestyle='-.', color='blue', label='voc_stat_s' + run_str + '-1')
    plq(plt, smv, 'time', smv, 'voc_stat_s', add=-1., linestyle=':', color='red', label='voc_stat_s' + ver_str + '-1')
    plq(plt, sr, 'time', sr, 'vb_s', add=-2., linestyle='-', color='black', label='vb_s' + run_str + '-2')
    plq(plt, sv, 'time', smv, 'vb_s', add=-2., linestyle='--', color='pink', label='vb_s' + ver_str + '-2')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(335)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-', label='e_wrap' + run_str)
    plt.plot(mv.time, mv.e_wrap, color='orange', linestyle='--', label='e_wrap' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(336)
    plq(plt, mr, 'soc', mr, 'vb', color='blue', linestyle='-', label='vb' + run_str)
    plq(plt, mr, 'soc', mr, 'vb_hdwe_f', color='blue', linestyle='-', label='vb_hdwe_f' + run_str)
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='cyan', linestyle='--', label='vb_s' + ver_str)
    plq(plt, mr, 'soc', mr, 'voc_stat', color='orange', linestyle='-.', label='voc_stat' + run_str)
    plq(plt, mr, 'soc', mr, 'voc_stat_f', color='orange', linestyle='-.', label='voc_stat_f' + run_str, warn=False)
    plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='red', linestyle=':', label='voc_stat_s' + ver_str)
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(337)
    plq(plt, mr, 'time', mr, 'vb', color='blue', linestyle='-', label='vb' + run_str)
    plq(plt, mr, 'time', mr, 'vb_hdwe_f', color='blue', linestyle='-', label='vb_hdwe_f' + run_str)
    plt.plot(mv.time, mv.vb, color='cyan', linestyle='--', label='vb' + ver_str)
    plq(plt, sr, 'time', sr, 'vb_s', color='black', linestyle='-.', label='vb_s' + run_str)
    plq(plt, smv, 'time', smv, 'vb_s', color='magenta', linestyle=':', label='vb_s' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(338)
    plt.plot(mr.time, mr.dv_hys, color='blue', linestyle='-', label='dv_hys' + run_str)
    plt.plot(mv.time, mv.dv_hys, color='cyan', linestyle='--', label='dv_hys' + ver_str)
    plq(plt, sr, 'time', sr, 'dv_hys_s', color='black', linestyle='-.', label='dv_hys_s' + run_str, warn=False)
    plq(plt, smv, 'time', smv, 'dv_hys_s', color='magenta', linestyle=':', label='dv_hys_s' + ver_str,
        warn=False)
    plt.plot(mr.time, mr.sat - 0.5, color='black', linestyle='-', label='sat' + run_str + '-0.5')
    plt.plot(mv.time, np.array(mv.sat) - 0.5, color='green', linestyle='--', label='sat' + ver_str + '-0.5')
    plq(plt, sr, 'time', sr, 'sat_s', add=-0.5, color='red', linestyle='-.', label='sat_s' + run_str + '-0.5')
    if hasattr(sv, 'sat'):
        plt.plot(sv.time, np.array(sv.sat) - 0.5, color='cyan', linestyle=':', label='sat_s' + ver_str + '-0.5')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(339)
    plq(plt, mr, 'time', mr, 'Tb_rap', color='blue', linestyle='-', label='Tb_rap' + run_str)
    plq(plt, mr, 'time_t', mr, 'Tb_f', color='cyan', linestyle='--', label='Tb_f' + run_str, stairs=True)
    plq(plt, mv, 'time', mv, 'Tb_rap', color='black', linestyle='-.', label='Tb_rap' + ver_str)
    plq(plt, mv, 'time', mv, 'Tb_f', color='magenta', linestyle=':', label='Tb_f' + ver_str)
    plt.legend(loc=3)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")


    fig_list.append(plt.figure())  # GP 3 KF
    plt.subplot(111)
    plt.title(plot_title + ' GP 3 KF')
    print('GP 3 KF', end=':  ')
    plq(plt, mr, 'time', mr, 'ibmh', linestyle='-', color='blue', label='ib_amp_hdwe' + run_str)
    plq(plt, mr, 'time', mr, 'ibmm', linestyle='--', color='red', label='ib_amp_model' + run_str)
    plq(plt, mr, 'time', mr, 'ibmkf', linestyle='--', color='black', label='ib_amp_hdwe_kf' + run_str)
    plq(plt, mr, 'time', mr, 'ibnh', linestyle='-.', color='blue', label='ib_noa_hdwe' + run_str)
    plq(plt, mr, 'time', mr, 'ibnm', linestyle=':', color='magenta', label='ib_noa_model' + run_str)
    plq(plt, mr, 'time', mr, 'ib_noa_kf', linestyle='--', color='black', label='ib_noa_kf' + run_str)
    plq(plt, mv, 'time', mv, 'iscn_f', linestyle='-.', color='red', label='ib_noa_kf' + ver_str)
    plq(plt, mr, 'time', mr, 'ib_sel', add=-5, linestyle='-', color='blue', label='ib_sel-5' + run_str)
    plq(plt, sr, 'time', sr, 'ib_in_s', add=-5, linestyle='--', color='black', label='ib_in_s-5' + run_str)
    plq(plt, smv, 'time', smv, 'ib_in_s', add=-5, linestyle=':', color='red', label='ib_in_s-5' + ver_str)
    plt.xlabel('sec')
    plt.text(0.5, 0.2, "KF_Q_STD= " + "{:10.6f}".format(Battery.KF_Q_STD) + "KF_R_STD= " + "{:10.6f}".format(Battery.KF_R_STD),
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=12,
             color='blue',
             bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
    plt.legend(loc=3)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")


    return fig_list, fig_files


def tune_r(mr, mv, smv, filename, fig_files=None, plot_title=None, fig_list=None, run_str='_run', ver_str='_ver'):
    print('tune_r', end=':  ')
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
        ioc_f = mv.ioc.copy()
        if hasattr(mv, 'ioc'):
            ioc_f[i] = ios_filter.update(mv.ioc[i], T, reset=reset)
        else:
            ioc_f = None

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
    ioc_f  = ib_f.copy()
    if hasattr(mv, 'ioc'):
        ioc_f[n-1] = ioc_f[n-2]
    else:
        ioc_f = None
    dv_dot_cap[n-1] = dv_dot_cap[n-2]
    dv_bleed[-1] = dv_bleed[-2]

    fig_list.append(plt.figure())  # GP 3 Tune R
    plt.subplot(321)
    plt.title(plot_title + ' GP 3 Tune R')
    print('GP 3 Tune R', end=':  ')
    plt.plot(tv, vb, color='blue', linestyle='-', label='vb_x')
    if hasattr(smv, 'vb_s'):
        plt.plot(tv, smv.vb_s, color='cyan', linestyle='--', label='vb_s_ver')
    plt.plot(tv, voc, color='magenta', linestyle='-.', label='voc_x')
    plt.plot(tv, voc_soc, color='black', linestyle=':', label='voc_soc_x')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(322)
    plq(plt, mr, 'time', mr, 'dv_dyn', color='red', linestyle='-', label='dv_dyn_x' + run_str)
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='red', linestyle='-', label='dv_dyn_x_f' + run_str)
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

    fig_list.append(plt.figure())  # GP 3 Tune Summ
    plt.subplot(221)
    plt.title(plot_title + ' GP 3 Tune Summ')
    print('GP 3 Tune Summ', end=':  ')
    plq(plt, mr, 'time', mr, 'vb', linestyle='-', color='blue', label='vb' + run_str)
    plq(plt, mr, 'time', mr, 'vb_hdwe_f', linestyle='-', color='blue', label='vb_hdwe_f' + run_str)
    plq(plt, smv, 'time', smv, 'vb_s', color='magenta', linestyle=':', label='vb_s' + ver_str)
    plq(plt, smv, 'time', smv, 'voc_stat_s', linestyle='-.', color='black', label='voc_stat_s' + ver_str)
    plt.plot(mv.time, voc_stat_chg, linestyle=':', color='green', label='voc_stat_chg' + ver_str)
    plt.plot(mv.time, voc_stat_dis, linestyle=':', color='red', label='voc_stat_dis' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(222)
    plq(plt, mr, 'time', mr, 'vb', linestyle='-', color='blue', label='vb' + run_str)
    plq(plt, mr, 'time', mr, 'vb_hdwe_f', linestyle='-', color='blue', label='vb_hdwe_f' + run_str)
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='magenta', linestyle=':', label='vb_s' + ver_str)
    plq(plt, smv, 'soc_s', smv, 'voc_stat_s', linestyle='-.', color='black', label='voc_stat_s' + ver_str)
    plt.plot(mv.soc, voc_stat_chg, linestyle=':', color='green', label='voc_stat_chg' + ver_str)
    plt.plot(mv.soc, voc_stat_dis, linestyle=':', color='red', label='voc_stat_dis' + ver_str)
    plt.plot(mr.soc, mr.voc_soc, color='cyan', linestyle='--', label='voc_soc' + run_str)
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(223)
    plq(plt, mr, 'time_t', mr, 'Tb', linestyle='-', color='red', label='Tb' + run_str, stairs=True)
    plt.plot(mr.time, mr.ib_sel, linestyle='--', color='blue', label='ib_sel' + run_str)
    plq(plt, smv, 'time', smv, 'ib_in_s', linestyle='-.', color='magenta', label='ib_in_s' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(224)
    plq(plt, smv, 'time', smv, 'dv_dyn_s', color='black', linestyle=':', label='dv_dyn_s' + ver_str)
    plq(plt, smv, 'time', smv, 'dv_hys_s', color='magenta', linestyle=':', label='dv_hys_s' + ver_str)
    plt.xlabel('sec')
    plt.legend(loc=3)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")


    return fig_list, fig_files
