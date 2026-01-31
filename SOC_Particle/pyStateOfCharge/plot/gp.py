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

""" General data-over-model general plot
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

from myFilters import InlineExpLag
import matplotlib.pyplot as plt
from plot.plq import plq as plq
from Battery import Battery
import numpy as np
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})


def gp_1(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []
    print('gp_plot', end=':  ')
    fig_list.append(plt.figure())  # GP 1
    plt.subplot(221)
    plt.title(plot_title + ' GP 1')
    print('GP 1', end=':  ')
    plq(plt, sr, 'time', sr, 'vb_s', color='black', linestyle='-')
    plq(plt, smv, 'time', smv, 'vb_s', color='orange', linestyle='--')
    plq(plt, sr, 'time', sr, 'voc_s', color='blue', linestyle='-.')
    plq(plt, smv, 'time', smv, 'voc_s', color='red', linestyle=':')
    plq(plt, sr, 'time', sr, 'voc_stat_s', color='magenta', linestyle='-.')
    plq(plt, smv, 'time', smv, 'voc_stat_s', color='green', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, sr, 'time', sr, 'dv_hys_s', color='black', linestyle='-')
    plq(plt, smv, 'time', smv, 'dv_hys_s', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, sr, 'time', sr, 'soc_s', color='black', linestyle='-')
    plq(plt, smv, 'time', smv, 'soc_s', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, sr, 'time', sr, 'ib_in_s', color='blue', linestyle='-')
    plq(plt, smv, 'time', smv, 'ib_in_s', color='red', linestyle='--')
    if not strict_overplot:
        plq(plt, smv, 'time', smv, 'ib_fut_s', color='orange', linestyle='-.')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def gp_2(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []
    fig_list.append(plt.figure())
    plt.subplot(221)
    plt.title(plot_title + ' GP 2')
    print('GP 2', end=':  ')
    plq(plt, mr, 'time', mr, 'vb', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'vb_f', color='black', linestyle='-', warn=False)
    plq(plt, mr, 'time', mr, 'voc', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'voc_d', color='blue', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'voc', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'voc_stat', color='cyan', linestyle='-.')
    plq(plt, mv, 'time', mv, 'voc_stat', color='black', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'voc_stat_f', color='cyan', linestyle='-.', warn=False)
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, mr, 'time', mr, 'dv_hys', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, mr, 'time', mr, 'soc', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, mr, 'time', mr, 'ib_sel', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_sel', color='black', linestyle='--', warn=False)
    plq(plt, sr, 'time', sr, 'ib_in_s', color='cyan', linestyle='--')
    plq(plt, smv, 'time', smv, 'ib_in_s', color='magenta', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_charge', color='cyan', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_charge', color='orange', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_diff', color='red', linestyle=':')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def gp_2_nn_lag(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []
    fig_list.append(plt.figure())
    plt.subplot(321)
    plt.title(plot_title + ' GP 2 nn lag')
    print('GP 2 nn lag', end=':  ')
    plq(plt, mr, 'time', mr, 'sat', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'sat', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(322)
    plq(plt, mr, 'time', mr, 'voc', color='black', linestyle='-')
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
    plq(plt, mr, 'time', mr, 'ib', add=10., color='black', linestyle='-')
    plq(plt, mr, 'time', mr, 'ib_f', add=10., color='black', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'ib', add=+10., color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'ib_lag', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_lag', color='red', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(325)
    plq(plt, mr, 'soc', mr, 'voc', color='black', linestyle='-')
    plq(plt, mv, 'soc', mv, 'voc', color='orange', linestyle='--')
    plq(plt, mr, 'soc', mr, 'voc_d', color='black', linestyle='-', warn=False)
    plq(plt, mr, 'soc', mr, 'voc_soc', color='red', linestyle='-')
    plq(plt, mv, 'soc', mv, 'voc_soc', color='orange', linestyle='--')
    if hasattr(mr, 'voc'):
        mr.dv = np.array(mr.voc_soc) - np.array(mr.voc)
    elif hasattr(mr, 'voc_d'):
        mr.dv = np.array(mr.voc_soc) - np.array(mr.voc_d)
    plq(plt, mr, 'soc', mr, 'dv', add=13, color='blue', linestyle='-')
    mv.dv = np.array(mv.voc_soc) - np.array(mv.voc)
    plq(plt, mv, 'soc', mv, 'dv', add=+13, color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(326)
    plq(plt, mr, 'time', mr, 'voc', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc', color='cyan', linestyle='--')
    plq(plt, mr, 'time', mr, 'voc_d', color='black', linestyle='-.', warn=False)
    plq(plt, mr, 'time', mr, 'voc_soc', color='red', linestyle='-')
    plq(plt, mr, 'time', mr, 'voc_soc', color='red', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_soc', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'dv', add=13, color='blue', linestyle='-')
    mv.dv = np.array(mv.voc_soc) - np.array(mv.voc)
    plq(plt, mv, 'time', mv, 'dv', add=+13, color='orange', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def gp_3_ekf(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())
    plt.subplot(111)
    plt.title(plot_title + ' GP 3 KF')
    print('GP 3 KF', end=':  ')
    plq(plt, mr, 'time', mr, 'ib_amp_hdwe', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'ib_amp_model', color='red', linestyle='--')
    plq(plt, mv, 'time', mv, 'ib_amp_model', color='black', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_amp_hdwe_kf', color='black', linestyle='--')
    plq(plt, mr, 'time', mr, 'ib_noa_hdwe', color='blue', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_noa_model', color='magenta', linestyle='--')
    plq(plt, mv, 'time', mv, 'ib_noa_model', color='cyan', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_noa_kf', color='black', linestyle='--')
    plq(plt, mv, 'time', mv, 'iscn_f', color='red', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_sel', add=-5, color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'ib', add=-10, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib', add=-10, color='cyan', linestyle='--')
    plq(plt, sr, 'time', sr, 'ib_in_s', add=-10, color='orange', linestyle='-.')
    plq(plt, smv, 'time', smv, 'ib_in_s', add=-10, color='red', linestyle=':')
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


def gp_3_tune(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())  # GP 3 Tune
    plt.subplot(331)
    plt.title(plot_title + ' GP 3 Tune')
    print('GP 3 Tune', end=':  ')
    plq(plt, mr, 'time', mr, 'dv_dyn', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='blue', linestyle='-', warn=False)
    plq(plt, mv, 'time', sv, 'dv_dyn', color='cyan', linestyle='--')
    plq(plt, sr, 'time', sr, 'dv_dyn_s', color='black', linestyle='-.')
    plq(plt, smv, 'time', smv, 'dv_dyn_s', color='magenta', linestyle=':')
    plq(plt, mr, 'time', mr, 'dv_hys', color='pink', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys', color='blue', linestyle='--')
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
    # mr.ib_amp_hdwe = mr.ibmh
    # mr.ib_amp_model = mr.ibmm
    plq(plt, mr, 'time', mr, 'ib_amp_hdwe', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'ib_amp_model', color='red', linestyle='--')
    plq(plt, mv, 'time', mv, 'ib_amp_model', color='black', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_amp_hdwe_kf', color='black', linestyle='--')
    plq(plt, mr, 'time', mr, 'ib_noa_hdwe', color='blue', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_noa_model', color='magenta', linestyle='--')
    plq(plt, mv, 'time', mv, 'ib_noa_model', color='cyan', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_noa_kf', color='black', linestyle='--')
    plq(plt, mv, 'time', mv, 'iscn_f', color='red', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_sel', add=-5, color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'ib', add=-10, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib', add=-10, color='cyan', linestyle='--')
    plq(plt, sr, 'time', sr, 'ib_in_s', add=-10, color='orange', linestyle='-.')
    plq(plt, smv, 'time', smv, 'ib_in_s', add=-10, color='red', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(334)
    plq(plt, mr, 'time', mr, 'voc', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'voc_d', color='blue', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'voc', color='cyan', linestyle='--')
    plq(plt, mr, 'time', mr, 'voc_stat', add=-1., color='orange', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_stat', add=-1., color='blue', linestyle='--')
    plq(plt, mr, 'time', mr, 'voc_stat_f', add=-1., color='orange', linestyle='-', warn=False)
    plq(plt, sr, 'time', sr, 'voc_stat_s', add=-1., color='blue', linestyle='-.')
    plq(plt, smv, 'time', smv, 'voc_stat_s', add=-1., color='red', linestyle=':')
    plq(plt, sr, 'time', sr, 'vb_s', add=-2., color='black', linestyle='-')
    plq(plt, sv, 'time', smv, 'vb_s', add=-2., color='pink', linestyle='--')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(335)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'e_wrap', color='orange', linestyle='--')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(336)
    plq(plt, mr, 'soc', mr, 'vb', color='blue', linestyle='-')
    plq(plt, mr, 'soc', mr, 'vb_hdwe_f', color='blue', linestyle='-')
    plq(plt, mv, 'soc', mv, 'vb_hdwe_f', color='cyan', linestyle='-.')
    plq(plt, sr, 'soc_s', sr, 'vb_s', color='red', linestyle='-')
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='cyan', linestyle='--')
    plq(plt, mr, 'soc', mr, 'voc_stat', color='orange', linestyle='-.')
    plq(plt, mr, 'soc', mr, 'voc_stat_f', color='orange', linestyle='-.', warn=False)
    plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='red', linestyle=':')
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(337)
    plq(plt, mr, 'time', mr, 'vb', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'vb_hdwe_f', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb_hdwe_f', color='magenta', linestyle='--')
    plq(plt, sr, 'time', sr, 'vb_s', color='black', linestyle='-.')
    plq(plt, smv, 'time', smv, 'vb_s', color='magenta', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(338)
    plq(plt, mr, 'time', mr, 'dv_hys', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys', color='cyan', linestyle='--')
    plq(plt, sr, 'time', sr, 'dv_hys_s', color='black', linestyle='-.', warn=False)
    plq(plt, smv, 'time', smv, 'dv_hys_s', color='magenta', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'sat', add=-0.5, color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'sat', add=-0.5, color='green', linestyle='--')
    plq(plt, sr, 'time', sr, 'sat_s', add=-0.5, color='red', linestyle='-.')
    if hasattr(sv, 'sat'):
        plq(plt, sv, 'time', sv, 'sat', add=-0.5, color='cyan', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(339)
    plq(plt, mr, 'time', mr, 'Tb_rap', color='blue', linestyle='-')
    plq(plt, mr, 'time_t', mr, 'Tb_f', color='cyan', linestyle='--', stairs=True)
    plq(plt, mv, 'time', mv, 'Tb_rap', color='black', linestyle='-.')
    plq(plt, mv, 'time', mv, 'Tb_f', color='magenta', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def tune_r(mr, mv, smv, filename, fig_files=None, plot_title=None, fig_list=None, run_str='_run', ver_str='_ver'):
    if fig_files is None:
        fig_files = []
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
    for i in range(n - 1):
        reset = i == 0
        T = tv[i + 1] - tv[i]

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

        ioc_calc_from_dot[i] = ib_f[i] - cap * dv_dot_calc[i]
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

    dv_dot_calc[n - 1] = dv_dot_calc[n - 2]
    dv_dot_req[n - 1] = dv_dot_req[n - 2]
    if hasattr(mv, 'ioc'):
        r_calc[n - 1] = r_calc[n - 2]
    r_calc_from_dot[n - 1] = r_calc_from_dot[n - 2]
    ioc_req[n - 1] = ioc_req[n - 2]
    dv_hys_req_f[n - 1] = dv_hys_req_f[n - 2]
    dv_hys_calc_f[n - 1] = dv_hys_calc_f[n - 2]
    ioc_calc_from_dot[n - 1] = ioc_calc_from_dot[n - 2]
    ib_f[n - 1] = ib_f[n - 2]
    ioc_f = ib_f.copy()
    if hasattr(mv, 'ioc'):
        ioc_f[n - 1] = ioc_f[n - 2]
    else:
        ioc_f = None
    dv_dot_cap[n - 1] = dv_dot_cap[n - 2]
    dv_bleed[-1] = dv_bleed[-2]

    fig_list.append(plt.figure())  # GP 3 Tune R
    plt.subplot(321)
    plt.title(plot_title + ' GP 3 Tune R')
    print('GP 3 Tune R', end=':  ')
    plq(plt, mv, 'time', mv, 'vb', color='blue', linestyle='-')
    if hasattr(smv, 'vb_s'):
        plq(plt, mv, 'time', smv, 'vb_s', color='cyan', linestyle='--')
    plq(plt, mv, 'time', mv, 'voc', color='magenta', linestyle='-.')
    plq(plt, mv, 'time', mv, 'voc_soc', color='black', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(322)
    plq(plt, mr, 'time', mr, 'dv_dyn', color='red', linestyle='-')
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='red', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_dyn', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys_calc', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys_req', color='magenta', linestyle='--')
    plq(plt, mv, 'time', mv, 'dv_hys_calc_f', color='red', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys_req_f', color='cyan', linestyle='--')
    plq(plt, mv, 'time', mv, 'dv_hys', color='orange', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(323)
    plq(plt, mv, 'time', mv, 'r_calc_from_dot', color='cyan', linestyle='--')
    if hasattr(mv, 'ioc'):
        plq(plt, mv, 'time', mv, 'r_calc', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'r_req', color='black', linestyle='-')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(324)
    plq(plt, mv, 'time', mv, 'ioc_calc_from_dot', color='orange', linestyle='--')
    plq(plt, mv, 'time', mv, 'mv.ib', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_f', color='cyan', linestyle='--')
    if hasattr(mv, 'ioc'):
        plq(plt, mv, 'time', mv, 'mv.ioc', color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'ioc_f', color='pink', linestyle='--')
    plq(plt, mv, 'time', mv, 'ioc_req', color='magenta', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(325)
    plq(plt, mv, 'time', mv, 'ioc_req', color='magenta', linestyle='--')
    plq(plt, mv, 'time', mv, 'mv.ib', color='blue', linestyle='-')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(326)
    plq(plt, mv, 'time', mv, 'dv_dot_req', color='magenta', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_dot_cap', color='black', linestyle='--')
    plq(plt, mv, 'time', mv, 'dv_dot_calc', color='blue', linestyle='-.')
    plt.xlabel('sec')
    plt.legend(loc=2)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # GP 3 Tune Summ
    plt.subplot(221)
    plt.title(plot_title + ' GP 3 Tune Summ')
    print('GP 3 Tune Summ', end=':  ')
    plq(plt, mr, 'time', mr, 'vb', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'vb_hdwe_f', color='blue', linestyle='-')
    plq(plt, smv, 'time', smv, 'vb_s', color='magenta', linestyle=':')
    plq(plt, smv, 'time', smv, 'voc_stat_s', color='black', linestyle='-.')
    plq(plt, mv, 'time', mv, 'voc_stat_chg', color='green', linestyle=':')
    plq(plt, mv, 'time', mv, 'voc_stat_dis', color='red', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(222)
    plq(plt, mr, 'time', mr, 'vb', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'vb_hdwe_f', color='blue', linestyle='-')
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='magenta', linestyle=':')
    plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='black', linestyle='-.')
    mv.voc_stat_chg = voc_stat_chg
    mv.voc_stat_dis = voc_stat_dis
    plq(plt, mv, 'soc', mv, 'voc_stat_chg', color='green', linestyle=':')
    plq(plt, mv, 'soc', mv, 'voc_stat_dis', color='red', linestyle=':')
    plq(plt, mr, 'soc', mr, 'voc_soc', color='cyan', linestyle='--')
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(223)
    plq(plt, mr, 'time_t', mr, 'Tb', color='red', linestyle='-', stairs=True)
    plq(plt, mr, 'time', mr, 'ib_sel', color='blue', linestyle='--')
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
