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

""" General data-over-model general plot of embedded simulation sim_s
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq
import numpy as np
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})


def sim_s_plots(mr, mv, sr, sv, smv, filename, fig_files=None, plot_title=None, fig_list=None,
               run_str='_run', ver_str='_ver', strict_overplot=False):
    print('sim_s_plot', end=':  ')
    if sr and smv:
        fig_list.append(plt.figure())  # sim_s  1
        plt.subplot(331)
        plt.title(plot_title + ' sim_s 1')
        print('sim_s 1', end=':  ')
        plq(plt, mr, 'time', mr, 'ib_sel', color='blue', linestyle='-')
        plq(plt, mr, 'time', mr, 'ib', add=-5, color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib', add=-5, color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'ib_in_s', color='green', linestyle='-.')
        plq(plt, smv, 'time', smv, 'ib_in_s', color='orange', linestyle=':')
        plq(plt, mr, 'time', mr, 'ib_charge', add=-1, color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib_charge', add=-1, color='orange', linestyle='--')
        plq(plt, sr, 'time', sr, 'ib_charge_s', add=-1, color='blue', linestyle='-.')
        plq(plt, smv, 'time', smv, 'ib_charge_s', add=-1, color='red', linestyle=':')
        plq(plt, sr, 'time', sr, 'ib_in_s', add=+1, color='green', linestyle='-.')
        plq(plt, smv, 'time', smv, 'ib_in_s', add=+1, color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(332)
        plq(plt, sr, 'time', sr, 'soc_s', color='magenta', linestyle='-')
        plq(plt, smv, 'time', smv, 'soc_s', color='green', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(333)
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='magenta', linestyle='-')
        plq(plt, smv, 'time', smv, 'voc_stat_s', color='green', linestyle='--')
        plq(plt, sr, 'time', sr, 'vsat_s', color='blue', linestyle='-.')
        plq(plt, smv, 'time', smv, 'vsat_s', color='cyan', linestyle=':')
        plq(plt, sr, 'time', sr, 'vb_s', color='orange', linestyle='-.')
        plq(plt, smv, 'time', smv, 'vb_s', color='black', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(334)
        plq(plt, sr, 'time', sr, 'Tb_f_s', color='magenta', linestyle='-')
        plq(plt, smv, 'time', smv, 'Tb_f_s', color='green', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(335)
        plq(plt, sr, 'time', sr, 'dv_dyn_s', color='magenta', linestyle='-')
        plq(plt, smv, 'time', smv, 'dv_dyn_s', color='green', linestyle='--')
        plq(plt, mr, 'time', mr, 'dv_dyn', color='blue', linestyle='-.')
        plq(plt, mv, 'time', mv, 'dv_dyn', color='cyan', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(336)
        plq(plt, mr, 'time', mr, 'ib_sel', color='blue', linestyle='-')
        plq(plt, sr, 'time', sr, 'ib_s', color='red', linestyle='--')
        plq(plt, mr, 'time', mr, 'ioc', color='cyan', linestyle='-')
        plq(plt, mv, 'time', mv, 'ioc', color='magenta', linestyle='--')
        plq(plt, sr, 'time', sr, 'ioc_s', color='green', linestyle='-.')
        plq(plt, sv, 'time', sv, 'ioc', color='black', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(337)
        plq(plt, sr, 'time', sr, 'delta_q_s', color='magenta', linestyle='-')
        plq(plt, smv, 'time', smv, 'delta_q_s', color='green', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(338)
        plq(plt, mr, 'time', mr, 'soft_reset', add=4, color='blue', linestyle='-')
        plq(plt, mr, 'time', mr, 'soft_reset_sim', add=4, color='red', linestyle='--')
        plq(plt, mr, 'time', mr, 'reset', add=4, color='cyan', linestyle='-.')
        plq(plt, mr, 'time', mr, 'reset_temp', add=4, color='black', linestyle=':')
        plq(plt, mr, 'time', mr, 'reset_all_faults', add=2, color='black', linestyle=':')
        plq(plt, sr, 'time', sr, 'reset_s', color='magenta', linestyle='-')
        plq(plt, smv, 'time', smv, 'reset_s', color='green', linestyle='-.')
        plq(plt, mr, 'time', mr, 'init_mon', add=-2, color='blue', linestyle='-')
        plq(plt, mr, 'time', mr, 'init_sim', add=-2, color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'sat_s', add=-4, color='blue', linestyle='-')
        plq(plt, smv, 'time', smv, 'sat_s', add=-4, color='red', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(339)
        plq(plt, mr, 'time', mr, 'chm', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'chm', color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'chm_s', color='green', linestyle='-.')
        plq(plt, smv, 'time', smv, 'chm_s', color='orange', linestyle=':')
        plq(plt, sv, 'time', sv, 'chm', add=+4, color='red', linestyle='-')
        plq(plt, smv, 'time', smv, 'chm_s', add=+4, color='red', linestyle='-')
        plq(plt, smv, 'time', smv, 'chm_s', add=+4, color='black', linestyle='--')
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # sim_s  2
        plt.subplot(331)
        if strict_overplot:
            plt.title(plot_title + ' sim_s 2')
            print('sim_s 2', end=':  ')
        plq(plt, mr, 'time', mr, 'vb', color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'vb', color='blue', linestyle='--')
        plq(plt, sr, 'time', sr, 'vb_s', color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'vb_s', color='orange', linestyle=':')
        plt.legend(loc=1)
        if not strict_overplot:
            plt.subplot(333)
            plq(plt, mr, 'time', mr, 'vb', color='red', linestyle='-')
            plq(plt, mr, 'time', mr, 'voc', color='black', linestyle='--')
            plq(plt, mr, 'time', mr, 'voc_stat', color='blue', linestyle='-.')
            plq(plt, mr, 'time', mr, 'voc_soc', color='orange', linestyle=':')
            plq(plt, mr, 'time', mr, 'vb_h', color='cyan', linestyle=':')
            plt.legend(loc=1)
        if not strict_overplot:
            plt.subplot(332)
            plq(plt, sr, 'time', sr, 'vb_s', color='red', linestyle='-')
            plq(plt, sr, 'time', sr, 'voc_s', color='black', linestyle='--')
            plq(plt, sr, 'time', sr, 'voc_stat_s', color='blue', linestyle='-.')
            plt.legend(loc=1)
        if not strict_overplot:
            plt.subplot(334)
            plq(plt, mv, 'time', mv, 'vb', color='red', linestyle='-')
            plq(plt, mv, 'time', mv, 'voc', color='black', linestyle='--')
            plq(plt, mv, 'time', mv, 'voc_stat', color='blue', linestyle='-.')
            plt.legend(loc=1)
        if not strict_overplot:
            plt.subplot(335)
            plq(plt, smv, 'time', smv, 'vb_s', color='red', linestyle='-')
            plq(plt, smv, 'time', smv, 'voc_s', color='black', linestyle='--')
            plq(plt, mv, 'time', mv, 'voc_soc', color='blue', linestyle='-.')
            plq(plt, smv, 'time', smv, 'voc_stat_s', color='orange', linestyle=':')
            plt.legend(loc=1)
        plt.subplot(337)
        plq(plt, mr, 'time', mr, 'soc', color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'soc', color='blue', linestyle='--')
        plq(plt, mr, 'time', mr, 'soc_ekf', color='black', linestyle='-.')
        plq(plt, mv, 'time', mv, 'soc_ekf', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(338)
        plq(plt, sr, 'time', sr, 'soc_s', color='orange', linestyle='-')
        plq(plt, smv, 'time', smv, 'soc_s', color='cyan', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(336)
        plq(plt, mr, 'time', mr, 'voc', color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'voc', color='black', linestyle='--')
        plq(plt, sr, 'time', sr, 'voc_s', color='blue', linestyle='-.')
        plq(plt, smv, 'time', smv, 'voc_s', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(339)
        plq(plt, mr, 'time', mr, 'voc_stat', color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'voc_stat', color='blue', linestyle='--')
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'voc_stat_s', color='orange', linestyle=':')
        plt.legend(loc=1)

        fig_list.append(plt.figure())  # sim_s  2a
        plt.subplot(221)
        plt.title(plot_title + ' sim_s 2a')
        print('sim_s 2a', end=':  ')
        plq(plt, mr, 'time', mr, 'vb', color='black', linestyle='-')
        if strict_overplot:
            plq(plt, mv, 'time', mv, 'vb', color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'vb_s', color='green', linestyle='-.')
        if strict_overplot:
            plq(plt, smv, 'time', smv, 'vb_s', color='orange', linestyle=':')
        plq(plt, mr, 'time', mr, 'voc', color='brown', linestyle='-')
        if strict_overplot:
            plq(plt, mv, 'time', mv, 'voc', color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'voc_s', color='blue', linestyle='-.')
        if strict_overplot:
            plq(plt, smv, 'time', smv, 'voc_s', color='red', linestyle=':')
        plq(plt, mr, 'time', mr, 'voc_stat', color='lightgreen', linestyle='-')
        if strict_overplot:
            plq(plt, mv, 'time', mv, 'voc_stat', color='cyan', linestyle='--')
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='magenta', linestyle='-.')
        if strict_overplot:
            plq(plt, smv, 'time', smv, 'voc_stat_s', color='red', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(222)
        plq(plt, mr, 'time', mr, 'e_wrap', color='magenta', linestyle='-')
        if strict_overplot:
            plq(plt, mv, 'time', mv, 'e_wrap', color='blue', linestyle='--')
        plq(plt, mr, 'time', mr, 'e_wrap_filt', color='magenta', linestyle='-.')
        if strict_overplot:
            plq(plt, mv, 'time', mv, 'e_wrap_filt', color='blue', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(223)
        plq(plt, mr, 'time', mr, 'dv_dyn', color='black', linestyle='-')
        if strict_overplot:
            plq(plt, mv, 'time', mv, 'dv_dyn', color='cyan', linestyle='--')
        plq(plt, sr, 'time', sr, 'dv_dyn_s', color='red', linestyle='--')
        if strict_overplot:
            plq(plt, smv, 'time', smv, 'dv_dyn_s', color='blue', linestyle='--')
        if sr.voc_s is None:
            sr.dv_hyst_s_est = None
        else:
            sr.dv_hyst_s_est = sr.voc_s - sr.voc_stat_s
        plq(plt, sr, 'time', sr, 'dv_hyst_s_est', color='cyan', linestyle='-')
        if smv.voc_s is None:
            smv.dv_hyst_s_est = None
        else:
            smv.dv_hyst_s_est = np.array(smv.voc_s) - np.array(smv.voc_stat_s)
        plq(plt, smv, 'time', smv, 'dv_hyst_s_est', color='magenta', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(224)
        plq(plt, mr, 'time', mr, 'ib_charge', color='red', linestyle='-')
        if strict_overplot:
            plq(plt, mv, 'time', mv, 'ib_charge', color='cyan', linestyle='--')
        plq(plt, sr, 'time', sr, 'ib_charge_s', color='red', linestyle='-.')
        if strict_overplot:
            plq(plt, smv, 'time', smv, 'ib_charge_s', color='cyan', linestyle=':')
        plt.legend(loc=1)

        fig_list.append(plt.figure())  # sim_s  3
        plt.subplot(321)
        plt.title(plot_title + ' sim_s 3')
        print('sim_s 3', end=':  ')
        plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'soc', color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'soc_s', color='green', linestyle='-.')
        plq(plt, sv, 'time', sv, 'soc', color='orange', linestyle=':')
        plq(plt, smv, 'time', smv, 'soc_s', color='orange', linestyle=':')
        plq(plt, mr, 'time', mr, 'soc_ekf', color='cyan', linestyle='-')
        plq(plt, mv, 'time', mv, 'soc_ekf', color='magenta', linestyle='--')
        sv.soc = np.array(sv.soc)
        sv.soc_s = np.array(sv.soc_s)
        plq(plt, sv, 'time', sv, 'soc', add=-.2, color='orange', linestyle='-')
        plq(plt, smv, 'time', smv, 'soc_s', add=-.2, color='orange', linestyle='-')
        plq(plt, smv, 'time', smv, 'soc_s', add=-.2, color='black', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(322)
        if mr.vb_h is not None and max(mr.vb_h) > 1.:
            plq(plt, mr, 'soc', mr, 'vb_h', color='magenta', linestyle=':')
        plq(plt, mr, 'soc', mr, 'voc_stat', color='cyan', linestyle='-.')
        plq(plt, mr, 'soc', mr, 'voc_soc', color='blue', linestyle='-')
        plq(plt, mv, 'soc', mv, 'voc_soc', color='red', linestyle='--')
        plq(plt, sr, 'soc_s', sr, 'voc_stat_s', color='green', linestyle='-.')
        plq(plt, sv, 'soc', sv, 'voc_stat', color='orange', linestyle=':')
        plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='orange', linestyle=':')
        plq(plt, mr, 'soc', mr, 'vsat', color='red', linestyle='-')
        plq(plt, mv, 'soc', mv, 'vsat', color='black', linestyle='--')
        plq(plt, mv, 'soc', mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(323)
        if mr.voc_soc is not None:
            mr.e_wrap = np.array(mr.voc_soc) - np.array(mr.voc_stat)
            plq(plt, mr, 'time', mr, 'e_wrap', color='blue', linestyle='-')
        mv.e_wrap = np.array(mv.voc_soc) - np.array(mv.voc_stat)
        plq(plt, mv, 'time', mv, 'e_wrap', color='red', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(324)
        plq(plt, mr, 'time', mr, 'voc', color='black', linestyle='-')
        plq(plt, mr, 'time', mr, 'voc_soc', color='blue', linestyle='--')
        plq(plt, mv, 'time', mv, 'voc_soc', color='green', linestyle='-.')
        plq(plt, mr, 'time', mr, 'voc_stat', color='red', linestyle=':')
        plq(plt, mv, 'time', mv, 'voc_stat', color='cyan', linestyle='--')
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='magenta', linestyle='-.')
        plq(plt, smv, 'time', smv, 'voc_stat_s', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(325)
        if mr.vb_h is not None and max(mr.vb_h) > 1.:
            plq(plt, mr, 'time', mr, 'vb_h', color='magenta', linestyle='--')
        plq(plt, mr, 'time', mr, 'voc_stat', color='cyan', linestyle='-.')
        plq(plt, mr, 'time', mr, 'voc_soc', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'voc_soc', color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='green', linestyle='-.')
        plq(plt, sv, 'time', sv, 'voc_stat', color='orange', linestyle=':')
        plq(plt, smv, 'time', smv, 'voc_stat_s', color='orange', linestyle=':')
        plq(plt, mr, 'time', mr, 'vsat', color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'vsat', color='black', linestyle='--')
        plq(plt, mv, 'time', mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(326)
        plq(plt, mr, 'soc', mr, 'voc_soc', color='blue', linestyle='-')
        plq(plt, mv, 'soc', mv, 'voc_soc', color='red', linestyle='--')
        plq(plt, mr, 'soc', mr, 'voc_stat', color='red', linestyle='-.')
        plq(plt, mv, 'soc', mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)

        fig_list.append(plt.figure())  # sim_s  4
        plt.subplot(221)
        plt.title(plot_title + ' sim_s 4')
        print('sim_s 4', end=':  ')
        plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'soc', color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'soc_s', color='green', linestyle='-.')
        plq(plt, sv, 'time', sv, 'soc', color='orange', linestyle=':')
        plq(plt, smv, 'time', smv, 'soc_s', color='orange', linestyle=':')
        plq(plt, mr, 'time', mr, 'soc_ekf', color='cyan', linestyle='-')
        plq(plt, mv, 'time', mv, 'soc_ekf', color='magenta', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(223)
        plq(plt, mr, 'soc', mr, 'voc_stat', color='magenta', linestyle='-')
        plq(plt, mv, 'soc', mv, 'voc_stat', color='black', linestyle='--')
        plq(plt, mr, 'soc', mr, 'voc_ekf', color='green', linestyle='-.')
        plq(plt, mv, 'soc', mv, 'voc_ekf', color='cyan', linestyle=':')
        plq(plt, mr, 'soc', mr, 'vb_h', color='magenta', linestyle='-')
        plq(plt, mv, 'soc', mv, 'vb_hdwe', color='black', linestyle='--')
        plq(plt, mr, 'soc', mr, 'voc_stat', color='red', linestyle='-.')
        plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='blue', linestyle=':')
        plq(plt, mr, 'soc', mr, 'voc_soc', color='green', linestyle='-.')
        plq(plt, mv, 'soc', mv, 'voc_soc', color='orange', linestyle=':')
        if min(mv.voc_stat) < 4.:
            xmin = 0
        else:
            xmin = 12.5
        plt.ylim(xmin, 14.5)
        plt.legend(loc=1)
        plt.subplot(222)
        plq(plt, mr, 'soc', mr, 'voc_stat', color='magenta', linestyle='-')
        plq(plt, mv, 'soc', mv, 'voc_stat', color='black', linestyle='--')
        plq(plt, mr, 'soc', mr, 'voc_ekf', color='green', linestyle='-.')
        plq(plt, mv, 'soc', mv, 'voc_ekf', color='cyan', linestyle=':')
        plq(plt, mr, 'soc', mr, 'vb_h', color='magenta', linestyle='-')
        plq(plt, mv, 'soc', mv, 'vb_hdwe', color='black', linestyle='--')
        plq(plt, mr, 'soc', mr, 'voc_stat', color='red', linestyle='-.')
        plq(plt, smv, 'soc_s', smv, 'voc_stat_s', color='blue', linestyle=':')
        plq(plt, mr, 'soc', mr, 'voc_soc', color='green', linestyle='-.')
        plq(plt, mv, 'soc', mv, 'voc_soc', color='orange', linestyle=':')
        plt.ylim(xmin, 14.5)
        plt.legend(loc=1)
        plt.subplot(224)
        plq(plt, mr, 'time', mr, 'voc_stat', color='magenta', linestyle='-')
        plq(plt, mv, 'time', mv, 'voc_stat', color='black', linestyle='--')
        plq(plt, mr, 'time', mr, 'voc_ekf', color='green', linestyle='-.')
        plq(plt, mv, 'time', mv, 'voc_ekf', color='cyan', linestyle=':')
        plq(plt, mr, 'time', mr, 'vb_h', color='magenta', linestyle='-')
        plq(plt, mv, 'time', mv, 'vb_hdwe', color='black', linestyle='--')
        plq(plt, sv, 'time', sv, 'voc_stat', color='red', linestyle='-.')
        plq(plt, smv, 'time', smv, 'voc_stat_s', color='blue', linestyle=':')
        plq(plt, mr, 'time', mr, 'voc_soc', color='green', linestyle='-.')
        plq(plt, mv, 'time', mv, 'voc_soc', color='orange', linestyle=':')
        plt.ylim(xmin, 14.5)
        plt.legend(loc=1)
    return fig_list, fig_files
