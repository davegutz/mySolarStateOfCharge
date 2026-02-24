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

""" General data-over-model general plot of embedded simulation sim_s
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import matplotlib.pyplot as plt
from plot.PlotOptions import PlotOptions
from plot.plq import plq as plq
import numpy as np
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})


def sim_s_plots(S:PlotOptions, fig_files=None, fig_list=None):
    if not fig_files:
        fig_files = []
    if not fig_list:
        fig_list = []
    print('sim_s_plot', end=':  ')

    if S.sr and S.smv:
        fig_list.append(plt.figure())  # sim_s  1
        plt.subplot(331)
        plt.suptitle(S.plot_title + ' sim_s 1')
        print('sim_s 1', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'ib_sel', color='blue', linestyle='-')
        plq(plt, S.mr, 'time', S.mr, 'ib', add=-5, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib', add=-5, color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'ib_in_s', color='green', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'ib_in_s', color='orange', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'ib_charge', add=-1, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_charge', add=-1, color='orange', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'ib_charge_s', add=-1, color='blue', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'ib_charge_s', add=-1, color='red', linestyle=':')
        plq(plt, S.sr, 'time', S.sr, 'ib_in_s', add=+1, color='green', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'ib_in_s', add=+1, color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(332)
        plq(plt, S.sr, 'time', S.sr, 'soc_s', color='magenta', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', color='green', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(333)
        plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='magenta', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='green', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'vsat_s', color='blue', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'vsat_s', color='cyan', linestyle=':')
        plq(plt, S.sr, 'time', S.sr, 'vb_s', color='orange', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'vb_s', color='black', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(334)
        plq(plt, S.sr, 'time', S.sr, 'Tb_f_s', color='magenta', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'Tb_f_s', color='green', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(335)
        plq(plt, S.sr, 'time', S.sr, 'dv_dyn_s', color='magenta', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'dv_dyn_s', color='green', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'dv_dyn', color='blue', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'dv_dyn', color='cyan', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(336)
        plq(plt, S.mr, 'time', S.mr, 'ib_sel', color='blue', linestyle='-')
        plq(plt, S.sr, 'time', S.sr, 'ib_s', color='red', linestyle='--')
        plq(plt, S.smv, 'time', S.smv, 'ib_s', color='cyan', linestyle='-.')
        plq(plt, S.mr, 'time', S.mr, 'ioc', color='cyan', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ioc', color='magenta', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'ioc_s', color='green', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'ioc_s', color='black', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(337)
        plq(plt, S.sr, 'time', S.sr, 'delta_q_s', color='magenta', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'delta_q_s', color='green', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(338)
        plq(plt, S.mr, 'time', S.mr, 'soft_reset', add=4, color='blue', linestyle='-')
        plq(plt, S.mr, 'time', S.mr, 'soft_reset_sim', add=4, color='red', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'reset', add=4, color='cyan', linestyle='-.')
        plq(plt, S.mr, 'time', S.mr, 'reset_temp', add=4, color='black', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'reset_all_faults', add=2, color='black', linestyle=':')
        plq(plt, S.sr, 'time', S.sr, 'reset_s', color='magenta', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'reset_s', color='green', linestyle='-.')
        plq(plt, S.mr, 'time', S.mr, 'init_mon', add=-2, color='blue', linestyle='-')
        plq(plt, S.mr, 'time', S.mr, 'init_sim', add=-2, color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'sat_s', add=-4, color='blue', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'sat_s', add=-4, color='red', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(339)
        plq(plt, S.mr, 'time', S.mr, 'chm', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'chm', color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'chm_s', color='green', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'chm_s', color='orange', linestyle=':')
        plq(plt, S.sv, 'time', S.sv, 'chm', add=+4, color='red', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'chm_s', add=+4, color='red', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'chm_s', add=+4, color='black', linestyle='--')
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # sim_s  2
        plt.subplot(331)
        if S.strict_overplot:
            plt.suptitle(S.plot_title + ' sim_s 2')
            print('sim_s 2', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'vb', color='red', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'vb', color='blue', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'vb_s', color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'vb_s', color='orange', linestyle=':')
        plt.legend(loc=1)
        if not S.strict_overplot:
            plt.subplot(333)
            plq(plt, S.mr, 'time', S.mr, 'vb', color='red', linestyle='-')
            plq(plt, S.mr, 'time', S.mr, 'voc', color='black', linestyle='--')
            plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='blue', linestyle='-.')
            plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='orange', linestyle=':')
            plq(plt, S.mr, 'time', S.mr, 'vb_hdwe', color='cyan', linestyle=':')
            plt.legend(loc=1)
        if not S.strict_overplot:
            plt.subplot(332)
            plq(plt, S.sr, 'time', S.sr, 'vb_s', color='red', linestyle='-')
            plq(plt, S.sr, 'time', S.sr, 'voc_s', color='black', linestyle='--')
            plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='blue', linestyle='-.')
            plt.legend(loc=1)
        if not S.strict_overplot:
            plt.subplot(334)
            plq(plt, S.mv, 'time', S.mv, 'vb', color='red', linestyle='-')
            plq(plt, S.mv, 'time', S.mv, 'voc', color='black', linestyle='--')
            plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='blue', linestyle='-.')
            plt.legend(loc=1)
        if not S.strict_overplot:
            plt.subplot(335)
            plq(plt, S.smv, 'time', S.smv, 'vb_s', color='red', linestyle='-')
            plq(plt, S.smv, 'time', S.smv, 'voc_s', color='black', linestyle='--')
            plq(plt, S.mv, 'time', S.mv, 'voc_soc', color='blue', linestyle='-.')
            plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='orange', linestyle=':')
            plt.legend(loc=1)
        plt.subplot(337)
        plq(plt, S.mr, 'time', S.mr, 'soc', color='red', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc', color='blue', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='black', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(338)
        plq(plt, S.sr, 'time', S.sr, 'soc_s', color='orange', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', color='cyan', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(336)
        plq(plt, S.mr, 'time', S.mr, 'voc', color='red', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'voc', color='black', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'voc_s', color='blue', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'voc_s', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(339)
        plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='red', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='blue', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='orange', linestyle=':')
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # sim_s  2a
        plt.subplot(221)
        plt.suptitle(S.plot_title + ' sim_s 2a')
        print('sim_s 2a', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'vb', color='black', linestyle='-')
        if S.strict_overplot:
            plq(plt, S.mv, 'time', S.mv, 'vb', color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'vb_s', color='green', linestyle='-.')
        if S.strict_overplot:
            plq(plt, S.smv, 'time', S.smv, 'vb_s', color='orange', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'voc', color='brown', linestyle='-')
        if S.strict_overplot:
            plq(plt, S.mv, 'time', S.mv, 'voc', color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'voc_s', color='blue', linestyle='-.')
        if S.strict_overplot:
            plq(plt, S.smv, 'time', S.smv, 'voc_s', color='red', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='lightgreen', linestyle='-')
        if S.strict_overplot:
            plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='cyan', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='magenta', linestyle='-.')
        if S.strict_overplot:
            plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='red', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(222)
        plq(plt, S.mr, 'time', S.mr, 'e_wrap', color='magenta', linestyle='-')
        if S.strict_overplot:
            plq(plt, S.mv, 'time', S.mv, 'e_wrap', color='blue', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'e_wrap_filt', color='red', linestyle='-.')
        if S.strict_overplot:
            plq(plt, S.mv, 'time', S.mv, 'e_wrap_filt', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(223)
        plq(plt, S.mr, 'time', S.mr, 'dv_dyn', color='black', linestyle='-')
        if S.strict_overplot:
            plq(plt, S.mv, 'time', S.mv, 'dv_dyn', color='cyan', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'dv_dyn_s', color='red', linestyle='--')
        if S.strict_overplot:
            plq(plt, S.smv, 'time', S.smv, 'dv_dyn_s', color='blue', linestyle='--')
        if S.sr.voc_s is None:
            S.sr.dv_hyst_s_est = None
        else:
            S.sr.dv_hyst_s_est = S.sr.voc_s - S.sr.voc_stat_s
        plq(plt, S.sr, 'time', S.sr, 'dv_hyst_s_est', color='cyan', linestyle='-')
        if S.smv.voc_s is None:
            S.smv.dv_hyst_s_est = None
        else:
            S.smv.dv_hyst_s_est = np.array(S.smv.voc_s) - np.array(S.smv.voc_stat_s)
        plq(plt, S.smv, 'time', S.smv, 'dv_hyst_s_est', color='magenta', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(224)
        plq(plt, S.mr, 'time', S.mr, 'ib_charge', color='red', linestyle='-')
        if S.strict_overplot:
            plq(plt, S.mv, 'time', S.mv, 'ib_charge', color='cyan', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'ib_charge_s', color='red', linestyle='-.')
        if S.strict_overplot:
            plq(plt, S.smv, 'time', S.smv, 'ib_charge_s', color='cyan', linestyle=':')
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # sim_s  3
        plt.subplot(321)
        plt.suptitle(S.plot_title + ' sim_s 3')
        print('sim_s 3', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'soc', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc', color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'soc_s', color='green', linestyle='-.')
        plq(plt, S.sv, 'time', S.sv, 'soc', color='orange', linestyle=':')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', color='orange', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='cyan', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='magenta', linestyle='--')
        S.sv.soc_s = np.array(S.sv.soc_s)
        S.smv.soc_s = np.array(S.smv.soc_s)
        plq(plt, S.sv, 'time', S.sv, 'soc', add=-.2, color='orange', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', add=-.2, color='orange', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', add=-.2, color='black', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(322)
        if S.mr.vb_hdwe is not None and max(S.mr.vb_hdwe) > 1.:
            plq(plt, S.mr, 'soc', S.mr, 'vb_hdwe', color='magenta', linestyle=':')
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='cyan', linestyle='-.')
        plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='blue', linestyle='-')
        plq(plt, S.mv, 'soc', S.mv, 'voc_soc', color='red', linestyle='--')
        plq(plt, S.sr, 'soc_s', S.sr, 'voc_stat_s', color='green', linestyle='-.')
        plq(plt, S.sv, 'soc', S.sv, 'voc_stat', color='orange', linestyle=':')
        plq(plt, S.smv, 'soc_s', S.smv, 'voc_stat_s', color='orange', linestyle=':')
        plq(plt, S.mr, 'soc', S.mr, 'vsat', color='red', linestyle='-')
        plq(plt, S.mv, 'soc', S.mv, 'vsat', color='black', linestyle='--')
        plq(plt, S.mv, 'soc', S.mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(323)
        if S.mr.voc_soc is not None:
            S.mr.e_wrap = np.array(S.mr.voc_soc) - np.array(S.mr.voc_stat)
            plq(plt, S.mr, 'time', S.mr, 'e_wrap', color='blue', linestyle='-')
        S.mv.e_wrap = np.array(S.mv.voc_soc) - np.array(S.mv.voc_stat)
        plq(plt, S.mv, 'time', S.mv, 'e_wrap', color='red', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(324)
        plq(plt, S.mr, 'time', S.mr, 'voc', color='black', linestyle='-')
        plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='blue', linestyle='--')
        plq(plt, S.mv, 'time', S.mv, 'voc_soc', color='green', linestyle='-.')
        plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='red', linestyle=':')
        plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='cyan', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='magenta', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(325)
        if S.mr.vb_hdwe is not None and max(S.mr.vb_hdwe) > 1.:
            plq(plt, S.mr, 'time', S.mr, 'vb_hdwe', color='magenta', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='cyan', linestyle='-.')
        plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'voc_soc', color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='green', linestyle='-.')
        plq(plt, S.sv, 'time', S.sv, 'voc_stat', color='orange', linestyle=':')
        plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='orange', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'vsat', color='red', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'vsat', color='black', linestyle='--')
        plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(326)
        plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='blue', linestyle='-')
        plq(plt, S.mv, 'soc', S.mv, 'voc_soc', color='red', linestyle='--')
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='red', linestyle='-.')
        plq(plt, S.mv, 'soc', S.mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # sim_s  4
        plt.subplot(221)
        plt.suptitle(S.plot_title + ' sim_s 4')
        print('sim_s 4', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'soc', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc', color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'soc_s', color='green', linestyle='-.')
        plq(plt, S.sv, 'time', S.sv, 'soc', color='orange', linestyle=':')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', color='orange', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='cyan', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='magenta', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(223)
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='magenta', linestyle='-')
        plq(plt, S.mv, 'soc', S.mv, 'voc_stat', color='black', linestyle='--')
        plq(plt, S.mr, 'soc', S.mr, 'voc_ekf', color='green', linestyle='-.')
        plq(plt, S.mv, 'soc', S.mv, 'voc_ekf', color='cyan', linestyle=':')
        plq(plt, S.mr, 'soc', S.mr, 'vb_hdwe', color='magenta', linestyle='-')
        plq(plt, S.mv, 'soc', S.mv, 'vb_hdwe', color='black', linestyle='--')
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='red', linestyle='-.')
        plq(plt, S.smv, 'soc_s', S.smv, 'voc_stat_s', color='blue', linestyle=':')
        plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='green', linestyle='-.')
        plq(plt, S.mv, 'soc', S.mv, 'voc_soc', color='orange', linestyle=':')
        if min(S.mv.voc_stat) < 4.:
            xmin = 0
        else:
            xmin = 12.5
        plt.ylim(xmin, 14.5)
        plt.legend(loc=1)
        plt.subplot(222)
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='magenta', linestyle='-')
        plq(plt, S.mv, 'soc', S.mv, 'voc_stat', color='black', linestyle='--')
        plq(plt, S.mr, 'soc', S.mr, 'voc_ekf', color='green', linestyle='-.')
        plq(plt, S.mv, 'soc', S.mv, 'voc_ekf', color='cyan', linestyle=':')
        plq(plt, S.mr, 'soc', S.mr, 'vb_hdwe', color='magenta', linestyle='-')
        plq(plt, S.mv, 'soc', S.mv, 'vb_hdwe', color='black', linestyle='--')
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='red', linestyle='-.')
        plq(plt, S.smv, 'soc_s', S.smv, 'voc_stat_s', color='blue', linestyle=':')
        plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='green', linestyle='-.')
        plq(plt, S.mv, 'soc', S.mv, 'voc_soc', color='orange', linestyle=':')
        plt.ylim(xmin, 14.5)
        plt.legend(loc=1)
        plt.subplot(224)
        plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='magenta', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='black', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'voc_ekf', color='green', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'voc_ekf', color='cyan', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'vb_hdwe', color='magenta', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'vb_hdwe', color='black', linestyle='--')
        plq(plt, S.sv, 'time', S.sv, 'voc_stat', color='red', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='blue', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='green', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'voc_soc', color='orange', linestyle=':')
        plt.ylim(xmin, 14.5)
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
