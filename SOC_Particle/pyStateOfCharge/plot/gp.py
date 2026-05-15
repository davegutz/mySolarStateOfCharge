# GP_batteryEKF - general purpose battery class for EKF use
# Copyright (C) 2026 Dave Gutz
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
from dataclasses import dataclass
from typing import Optional

from filter.myFilters import InlineExpLag
import matplotlib.pyplot as plt
from plot.plq import plq as plq
from Battery import Battery
import numpy as np
import sys
from plot.PlotOptions import PlotOptions

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})

def gp_1(S:PlotOptions, fig_list=None, fig_files=None):
    if S.run_type == 'HistHist':
        return fig_list, fig_files
    if fig_files is None:
        fig_files = []
    print('gp_plot', end=':  ')

    fig_list.append(plt.figure())  # GP 1
    plt.subplot(221)
    plt.suptitle(S.plot_title + ' GP 1')
    print('GP 1', end=':  ')
    plq(plt, S.sr, 'time', S.sr, 'vb_s', color='black', linestyle='-', warn=not S.ver_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'vb_s', color='orange', linestyle='--', warn=not S.ver_is_run)
    plq(plt, S.sr, 'time', S.sr, 'voc_s', color='blue', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'voc_s', color='red', linestyle=':', warn=not S.ver_is_run)
    plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='magenta', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='green', linestyle=':', warn=not S.ver_is_run)
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, S.sr, 'time', S.sr, 'dv_hys_s', color='black', linestyle='-', warn=not S.ver_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'dv_hys_s', color='orange', linestyle='--', warn=not S.ver_is_run)
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, S.sr, 'time', S.sr, 'soc_s', color='black', linestyle='-', warn=not S.ver_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'soc_s', color='orange', linestyle='--', warn=not S.ver_is_run)
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, S.sr, 'time', S.sr, 'ib_in_s', color='blue', linestyle='-', warn=not S.ver_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'ib_in_s', color='red', linestyle='--', warn=not S.ver_is_run)
    if not S.strict_overplot:
        plq(plt, S.smv, 'time', S.smv, 'ib_fut_s', color='orange', linestyle='-.', warn=not S.ver_is_run)
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files

def gp_2(S:PlotOptions, fig_files=None, fig_list=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())
    plt.subplot(221)
    plt.suptitle(S.plot_title + ' GP 2')
    print('GP 2', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'vb', color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'vb', color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'vb_f', color='black', linestyle='-', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.mv, 'time', S.mv, 'vb_f', color='orange', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_sim)
    plq(plt, S.mr, 'time', S.mr, 'voc', color='blue', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc', color='red', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='cyan', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='black', linestyle=':', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'voc_stat_f', color='cyan', linestyle='-.', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.mv, 'time', S.mv, 'voc_stat_f', color='black', linestyle=':', warn=not S.ver_is_stdy and not S.run_is_run)
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='black', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, S.mr, 'time', S.mr, 'soc', color='black', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, S.mr, 'time', S.mr, 'ib_sel', color='black', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_sel', color='black', linestyle='--', warn=False)
    plq(plt, S.sr, 'time', S.sr, 'ib_in_s', color='cyan', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'ib_in_s', color='magenta', linestyle='-.', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'ib_charge', color='cyan', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_charge', color='orange', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_diff', color='red', linestyle=':')
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def gp_2_nn_lag(S:PlotOptions, fig_files=None, fig_list=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())
    plt.subplot(321)
    plt.suptitle(S.plot_title + ' GP 2 nn lag')
    print('GP 2 nn lag', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'sat', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'sat', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'saturated', color='black', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'saturated', color='orange', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(322)
    plq(plt, S.mr, 'time', S.mr, 'voc', color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'voc_f', color='black', linestyle='-', warn=not S.run_is_run)
    plq(plt, S.mv, 'time', S.mv, 'voc', color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc_f', color='orange', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_sim and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'vsat', color='blue', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'vsat', color='red', linestyle=':')
    plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='cyan', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'voc_soc', color='black', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(323)
    plq(plt, S.mr, 'time', S.mr, 'soc', color='black', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(324)
    plq(plt, S.mr, 'time', S.mr, 'ib', add=10., color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_f', add=10., color='black', linestyle='-', warn=S.run_type=='HistHist')
    plq(plt, S.mv, 'time', S.mv, 'ib', add=10., color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_f', add=10., color='orange', linestyle='--', warn=S.run_type=='HistHist')
    if S.run_type != 'HistSim':
        plq(plt, S.mr, 'time', S.mr, 'ib_lag', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_lag', color='red', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(325)
    plq(plt, S.mr, 'soc', S.mr, 'voc', color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'soc', S.mr, 'voc_f', color='black', linestyle='-', warn=not S.run_is_run)
    plq(plt, S.mv, 'soc', S.mv, 'voc', color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'soc', S.mv, 'voc_f', color='orange', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_sim)
    plq(plt, S.mr, 'soc', S.mr, 'voc_d', color='black', linestyle='-', warn=False)
    plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='red', linestyle='-')
    plq(plt, S.mv, 'soc', S.mv, 'voc_soc', color='orange', linestyle='--')
    if hasattr(S.mr, 'voc'):
        S.mr.dv = np.array(S.mr.voc_soc) - np.array(S.mr.voc)
    elif hasattr(S.mr, 'voc_d'):
        S.mr.dv = np.array(S.mr.voc_soc) - np.array(S.mr.voc_d)
    plq(plt, S.mr, 'soc', S.mr, 'dv', add=13, color='blue', linestyle='-', warn=S.run_is_run)
    if hasattr(S.mr, 'voc'):
        S.mr.dv = np.array(S.mr.voc_soc) - np.array(S.mr.voc)
    elif hasattr(S.mr, 'voc_stat_f'):
        S.mr.dv = np.array(S.mr.voc_soc) - np.array(S.mr.voc_stat_f)
    if hasattr(S.mv, 'voc'):
        S.mv.dv = np.array(S.mv.voc_soc) - np.array(S.mv.voc)
    elif hasattr(S.mv, 'voc_stat_f'):
        S.mv.dv = np.array(S.mv.voc_soc) - np.array(S.mv.voc_stat_f)
    plq(plt, S.mv, 'soc', S.mv, 'dv', add=+13, color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(326)
    plq(plt, S.mr, 'time', S.mr, 'voc', color='black', linestyle='-', warn=S.run_is_run)
    plq(plt, S.mr, 'time', S.mr, 'voc_f', color='black', linestyle='-', warn=S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc', color='cyan', linestyle='--', warn=S.run_is_run)
    plq(plt, S.mv, 'time', S.mv, 'voc_f', color='cyan', linestyle='--', warn=S.ver_is_stdy and S.ver_is_sim)
    plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='red', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'voc_soc', color='orange', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'dv', add=13, color='blue', linestyle='-', warn=S.run_is_run)
    plq(plt, S.mv, 'time', S.mv, 'dv', add=13, color='red', linestyle='--', warn=S.ver_is_sim)
    if hasattr(S.mv, 'voc'):
        S.mv.dv = np.array(S.mv.voc_soc) - np.array(S.mv.voc)
    elif hasattr(S.mv, 'voc_stat_f'):
        S.mv.dv = np.array(S.mv.voc_soc) - np.array(S.mv.voc_stat_f)
    plq(plt, S.mv, 'time', S.mv, 'dv', add=+13, color='orange', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def gp_3_ekf(S:PlotOptions, fig_files=None, fig_list=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())
    plt.subplot(111)
    plt.suptitle(S.plot_title + ' GP 3 KF')
    print('GP 3 KF', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_hdwe', add=-2.5, color='blue', linestyle='-')
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_model', add=-2.5, color='black', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_amp_hdwe', add=-2.5, color='magenta', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'ib_amp_model', add=-2.5, color='red', linestyle=':', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_hdwe_kf', add=-2.5, color='black', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_amp_hdwe_kf', add=-2.5, color='orange', linestyle='-.', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_hdwe', add=2.5, color='blue', linestyle='-')
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_model', add=2.5, color='magenta', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_noa_hdwe', add=2.5, color='magenta', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'ib_noa_model', add=2.5, color='cyan', linestyle=':', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_kf', add=2.5, color='black', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_noa_kf', add=2.5, color='magenta', linestyle='-.', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.mr, 'time', S.mr, 'iscn_f', color='black', linestyle='-.', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'iscn_f', color='red', linestyle=':', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_sel', add=-5, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_sel', add=-5, color='orange', linestyle='--', warn= not S.run_is_run and not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib', add=-10, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib', add=-10, color='cyan', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.sr, 'time', S.sr, 'ib_in_s', add=-10, color='orange', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'ib_in_s', add=-10, color='red', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plt.xlabel('sec')
    plt.text(0.5, 0.2, "KF_Q_STD= " + "{:10.6f}".format(Battery.KF_Q_STD) + "KF_R_STD= " + "{:10.6f}".format(Battery.KF_R_STD),
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=12,
             color='blue',
             bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
    plt.legend(loc=3)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def gp_3_tune(S:PlotOptions, fig_files=None, fig_list=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())  # GP 3 Tune
    plt.subplot(331)
    plt.suptitle(S.plot_title + ' GP 3 Tune')
    print('GP 3 Tune', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'dv_dyn', color='blue', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'dv_dyn', color='red', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'dv_dyn_f', color='blue', linestyle='-', warn=False)
    plq(plt, S.sr, 'time', S.sr, 'dv_dyn_s', color='black', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.sv, 'time', S.sv, 'dv_dyn_s', color='orange', linestyle=':', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.smr, 'time', S.smr, 'dv_dyn_s', color='green', linestyle='-', warn=not S.ver_is_stdy and not S.ver_is_run and not S.run_is_run)
    plq(plt, S.smv, 'time', S.smv, 'dv_dyn_s', color='magenta', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='pink', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='blue', linestyle='--')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(332)
    plq(plt, S.mr, 'time', S.mr, 'soc', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='cyan', linestyle='--')
    plq(plt, S.sr, 'time', S.sr, 'soc_s', color='black', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.sv, 'time', S.sv, 'soc_s', color='pink', linestyle=':', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.smr, 'time', S.smr, 'soc_s', color='green', linestyle='-.', warn=not S.ver_is_stdy and not S.ver_is_run and not S.run_is_run)
    plq(plt, S.smv, 'time', S.smv, 'soc_s', color='magenta', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='red', linestyle='--')
    plt.xlabel('sec')
    plt.legend(loc=4)
    plt.subplot(333)
    # mr.ib_amp_hdwe = mr.ibmh
    # mr.ib_amp_model = mr.ibmm
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_hdwe', add=-2.5, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_amp_hdwe', add=-2.5, color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_model', add=-2.5, color='magenta', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_amp_model', add=-2.5, color='black', linestyle=':', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_hdwe_kf', add=-2.5, color='black', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_hdwe', add=2.5, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_noa_hdwe', add=2.5, color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_model', add=2.5, color='magenta', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_noa_model', add=2.5, color='black', linestyle=':', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_kf', add=2.5, color='black', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'iscn_f', color='red', linestyle='-.', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'ib_sel', add=-5, color='blue', linestyle='-')
    plq(plt, S.mr, 'time', S.mr, 'ib', add=-10, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib', add=-10, color='cyan', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.sr, 'time', S.sr, 'ib_in_s', add=-10, color='orange', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smr, 'time', S.smr, 'ib_in_s', add=-10, color='pink', linestyle='-.', warn=not S.ver_is_stdy and not S.ver_is_run and not S.run_is_run)
    plq(plt, S.smv, 'time', S.smv, 'ib_in_s', add=-10, color='red', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plt.xlabel('sec')
    saved_fontsize = plt.rcParams['legend.fontsize']
    plt.rcParams['legend.fontsize'] = '6'
    plt.legend(loc=3)
    plt.rcParams['legend.fontsize'] = saved_fontsize
    plt.subplot(334)
    plq(plt, S.mr, 'time', S.mr, 'voc_f', color='blue', linestyle='-', warn=not S.run_is_run)
    plq(plt, S.mr, 'time', S.mr, 'voc', color='blue', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc', color='cyan', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'voc_stat_f', add=-1., color='orange', linestyle='-', warn=not S.run_is_run)
    plq(plt, S.mv, 'time', S.mv, 'voc_stat_f', add=-1., color='blue', linestyle='--', warn=not S.ver_is_sim and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'voc_stat', add=-1., color='orange', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc_stat', add=-1., color='blue', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', add=-1., color='blue', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smr, 'time', S.smr, 'voc_stat_s', add=-1., color='black', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run and not S.run_is_run)
    plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', add=-1., color='red', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'vsat', color='orange', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'vsat', color='blue', linestyle='--')
    if S.run_type == 'HistSim':
        plq(plt, S.mr, 'time', S.mr, 'vb_f', add=-2., color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'vb', add=-2., color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'vb', add=-2., color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'vb', add=-2., color='green', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.sr, 'time', S.sr, 'vb_s', add=-2., color='red', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'vb_s', add=-2., color='pink', linestyle='--', warn=not S.ver_is_stdy and not S.run_is_stdy)
    plt.xlabel('sec')
    saved_fontsize = plt.rcParams['legend.fontsize']
    plt.rcParams['legend.fontsize'] = '6'
    plt.legend(loc=2)
    plt.rcParams['legend.fontsize'] = saved_fontsize
    plt.subplot(335)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_n', add=1, color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_n', add=1, color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_n_filt', add=1, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_n_filt', add=1, color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'e_wrap', color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap', color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_filt', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_filt', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_m', add=-1, color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_m', add=-1, color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_m_filt', add=-1, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_m_filt', add=-1, color='red', linestyle='--')
    plt.xlabel('sec')
    plt.rcParams['legend.fontsize'] = 6
    plt.legend(loc=2)
    plt.rcParams['legend.fontsize'] = 'small'
    plt.subplot(336)
    plq(plt, S.mr, 'soc', S.mr, 'vb', color='blue', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'soc', S.mr, 'vb_hdwe_f', color='blue', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'soc', S.mv, 'vb_hdwe_f', color='cyan', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.sr, 'soc_s', S.sr, 'vb_s', color='red', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'soc_s', S.smv, 'vb_s', color='cyan', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='orange', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'soc', S.mr, 'voc_stat_f', color='orange', linestyle='-.', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.smr, 'soc_s', S.smr, 'voc_stat_s', color='black', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_run and not S.run_is_run)
    plq(plt, S.smv, 'soc_s', S.smv, 'voc_stat_s', color='red', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(337)
    plq(plt, S.mr, 'time', S.mr, 'vb', color='blue', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'vb', color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'vb_hdwe_f', color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'vb_hdwe_f', color='magenta', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.sr, 'time', S.sr, 'vb_s', color='black', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smr, 'time', S.smr, 'vb_s_r', color='blue', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_run and not S.run_is_run)
    plq(plt, S.smv, 'time', S.smv, 'vb_s_v', color='magenta', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(338)
    plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='cyan', linestyle='--')
    plq(plt, S.sr, 'time', S.sr, 'dv_hys_s', color='black', linestyle='-.', warn=False)
    plq(plt, S.smr, 'time', S.smr, 'dv_hys_s', color='black', linestyle='-', warn=False)
    plq(plt, S.smv, 'time', S.smv, 'dv_hys_s', color='magenta', linestyle=':', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'sat', add=-0.5, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'sat', add=-0.5, color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'saturated', add=-0.5, color='black', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'saturated', add=-0.5, color='green', linestyle=':')
    plq(plt, S.sr, 'time', S.sr, 'sat_s', add=-0.5, color='red', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.sv, 'time', S.sv, 'sat_s', add=-0.5, color='orange', linestyle=':', warn=not S.run_is_stdy and not S.run_is_run)
    if hasattr(S.sr, 'model_saturated'):
        plq(plt, S.sr, 'time', S.sr, 'model_saturated', add=-0.5, color='blue', linestyle='-.')
    if hasattr(S.sv, 'model_saturated'):
        plq(plt, S.sv, 'time', S.sv, 'model_saturated', add=-0.5, color='cyan', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(339)
    plq(plt, S.mr, 'time', S.mr, 'Tb', color='blue', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'Tb', color='black', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time_t', S.mr, 'Tb_f', color='cyan', linestyle='-.', stairs=True)
    plq(plt, S.mv, 'time_t', S.mv, 'Tb_f', color='magenta', linestyle=':', stairs=True, warn=S.run_type=='RunRun')
    plq(plt, S.mv, 'time', S.mv, 'Tb_f', color='magenta', linestyle=':', stairs=True, warn=S.run_type!='RunRun')
    plt.xlabel('sec')
    plt.legend(loc=3)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def tune_r(S:PlotOptions, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []
    print('tune_r', end=':  ')
    # delineate charging and discharging
    voc_stat_chg = np.copy(S.mv.voc_stat)
    voc_stat_dis = np.copy(S.mv.voc_stat)
    if hasattr(S.smv, 'ib_in_s'):
        for i in range(len(voc_stat_chg)):
            if S.smv.ib_in_s[i] > -0.1:
                voc_stat_dis[i] = None
            elif S.smv.ib_in_s[i] < 0.1:
                voc_stat_chg[i] = None

    vb = np.copy(S.mv.vb)
    voc = np.copy(S.mv.voc)
    voc_soc = np.copy(S.mv.voc_soc)
    voc_stat = np.copy(S.mv.voc_stat)
    ib_f = np.copy(S.mv.ib)
    tv = np.copy(S.mv.time)
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
        ib_f[i] = ib_filter.update(S.mv.ib[i], T, reset=reset)
        ioc_f = S.mv.ioc.copy()
        if hasattr(S.mv, 'ioc'):
            ioc_f[i] = ios_filter.update(S.mv.ioc[i], T, reset=reset)
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

        if hasattr(S.mv, 'ioc'):
            if abs(ioc_f[i]) < .5:
                r_calc[i] = 0
            else:
                # noinspection PyTypeChecker
                r_calc[i] = max(min(dv_hys_calc_f[i] / ioc_f[i], 0.1), -0.1)

    dv_dot_calc[n - 1] = dv_dot_calc[n - 2]
    dv_dot_req[n - 1] = dv_dot_req[n - 2]
    if hasattr(S.mv, 'ioc'):
        r_calc[n - 1] = r_calc[n - 2]
    r_calc_from_dot[n - 1] = r_calc_from_dot[n - 2]
    ioc_req[n - 1] = ioc_req[n - 2]
    dv_hys_req_f[n - 1] = dv_hys_req_f[n - 2]
    dv_hys_calc_f[n - 1] = dv_hys_calc_f[n - 2]
    ioc_calc_from_dot[n - 1] = ioc_calc_from_dot[n - 2]
    ib_f[n - 1] = ib_f[n - 2]
    ioc_f = ib_f.copy()
    if hasattr(S.mv, 'ioc'):
        ioc_f[n - 1] = ioc_f[n - 2]
    else:
        ioc_f = None
    dv_dot_cap[n - 1] = dv_dot_cap[n - 2]
    dv_bleed[-1] = dv_bleed[-2]

    fig_list.append(plt.figure())  # GP 3 Tune R
    plt.subplot(321)
    plt.suptitle(S.plot_title + ' GP 3 Tune R')
    print('GP 3 Tune R', end=':  ')
    plq(plt, S.mv, 'time', S.mv, 'vb', color='blue', linestyle='-')
    if hasattr(S.smv, 'vb_s'):
        plq(plt, S.mv, 'time', S.smv, 'vb_s', color='cyan', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'voc', color='magenta', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'voc_soc', color='black', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(322)
    plq(plt, S.mr, 'time', S.mr, 'dv_dyn', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_dyn', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'dv_dyn_f', color='black', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_dyn_f', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'dv_hys_calc', add=0.5, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys_calc', add=0.5, color='red', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys_req', color='magenta', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys_calc_f', color='red', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys_req_f', color='cyan', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='orange', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(323)
    plq(plt, S.mv, 'time', S.mv, 'Noner_calc_from_dot', color='cyan', linestyle='--')
    if hasattr(S.mv, 'ioc'):
        plq(plt, S.mv, 'time', S.mv, 'Noner_calc', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'Noner_req', color='black', linestyle='-')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(324)
    plq(plt, S.mv, 'time', S.mv, 'Noneioc_calc_from_dot', color='orange', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'Nonemv.ib', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'Noneib_f', color='cyan', linestyle='--')
    if hasattr(S.mv, 'ioc'):
        plq(plt, S.mv, 'time', S.mv, 'Nonemv.ioc', color='red', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'Noneioc_f', color='pink', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'Noneioc_req', color='magenta', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(325)
    plq(plt, S.mv, 'time', S.mv, 'Noneioc_req', color='magenta', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'Nonemv.ib', color='blue', linestyle='-')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(326)
    plq(plt, S.mv, 'time', S.mv, 'Nonedv_dot_req', color='magenta', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'Nonedv_dot_cap', color='black', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'Nonedv_dot_calc', color='blue', linestyle='-.')
    plt.xlabel('sec')
    plt.legend(loc=2)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # GP 3 Tune Summ
    plt.subplot(221)
    plt.suptitle(S.plot_title + ' GP 3 Tune Summ')
    print('GP 3 Tune Summ', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'vb', color='blue', linestyle='-')
    plq(plt, S.mr, 'time', S.mr, 'vb_hdwe_f', color='blue', linestyle='-')
    plq(plt, S.smv, 'time', S.smv, 'vb_s', color='magenta', linestyle=':')
    plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='black', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'Nonevoc_stat_chg', color='green', linestyle=':')
    plq(plt, S.mv, 'time', S.mv, 'voc_stat_dis', color='red', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(222)
    plq(plt, S.mr, 'time', S.mr, 'vb', color='blue', linestyle='-')
    plq(plt, S.mr, 'time', S.mr, 'vb_hdwe_f', color='blue', linestyle='-')
    plq(plt, S.smv, 'soc_s', S.smv, 'vb_s', color='magenta', linestyle=':')
    plq(plt, S.smv, 'soc_s', S.smv, 'voc_stat_s', color='black', linestyle='-.')
    S.mv.voc_stat_chg = voc_stat_chg
    S.mv.voc_stat_dis = voc_stat_dis
    plq(plt, S.mv, 'soc', S.mv, 'voc_stat_chg', color='green', linestyle=':')
    plq(plt, S.mv, 'soc', S.mv, 'voc_stat_dis', color='red', linestyle=':')
    plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='cyan', linestyle='--')
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(223)
    plq(plt, S.mr, 'time_t', S.mr, 'Tb', color='red', linestyle='-', stairs=True)
    plq(plt, S.mr, 'time', S.mr, 'ib_sel', color='blue', linestyle='--')
    plq(plt, S.smv, 'time', S.smv, 'ib_in_s', color='magenta', linestyle='-.')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(224)
    plq(plt, S.smv, 'time', S.smv, 'dv_dyn_s', color='black', linestyle=':')
    plq(plt, S.smv, 'time', S.smv, 'dv_hys_s', color='magenta', linestyle=':')
    plt.xlabel('sec')
    plt.legend(loc=3)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:

        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
