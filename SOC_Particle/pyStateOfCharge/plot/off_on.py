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

""" General data-over-model general plot of off/on/off behavior
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq
import sys
from plot.PlotOptions import PlotOptions
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})


def off_on_plots(S:PlotOptions, fig_files=None, fig_list=None):
    print('off_on_plot', end=':  ')

    if S.sr and S.smv:
        fig_list.append(plt.figure())  # 7 off/on sim
        plt.subplot(321)
        plt.suptitle(S.plot_title + ' off/on sim 1')
        print('off/on sim 1', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'vb_s', color='black', linestyle='-')
        plq(plt, S.sv, 'time', S.sv, 'vb', color='cyan', linestyle='--')
        plq(plt, S.smv, 'time', S.smv, 'vb_s', color='red', linestyle='-.')
        plq(plt, S.sr, 'time', S.sr, 'voc_s', color='blue', linestyle='-')
        plq(plt, S.sv, 'time', S.sv, 'voc', color='magenta', linestyle='--')
        plq(plt, S.smv, 'time', S.smv, 'voc_s', color='cyan', linestyle='-.')
        plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='black', linestyle='-')
        plq(plt, S.sv, 'time', S.sv, 'voc_stat', color='orange', linestyle='--')
        plq(plt, S.smv, 'time', S.smv, 'voc_stat_s', color='red', linestyle='-.')
        plt.legend(loc=1)
        plt.subplot(322)
        plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='red', linestyle='--')
        plq(plt, S.sv, 'time', S.sv, 'dv_hys', color='orange', linestyle='-.')
        plq(plt, S.sr, 'time', S.sr, 'dv_hys_s', add=0.01, color='black', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'dv_hys_s', add=0.01, color='cyan', linestyle='-.')
        plt.legend(loc=1)
        plt.subplot(323)
        plq(plt, S.sr, 'time', S.sr, 'soc_s', color='black', linestyle='-')
        plq(plt, S.sv, 'time', S.sv, 'soc', color='orange', linestyle='--')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', color='red', linestyle='-.')
        plt.xlabel('sec')
        plt.legend(loc=1)
        plt.subplot(324)
        plq(plt, S.sr, 'time', S.sr, 'ib_s', color='black', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'ib_s', color='orange', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'ib_dyn_s', color='blue', linestyle='-')
        plq(plt, S.sv, 'time', S.sv, 'ib_dyn', color='red', linestyle='--')
        plt.xlabel('sec')
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # 8 off/on mon 1
        plt.subplot(321)
        plt.suptitle(S.plot_title + ' off/on mon 1')
        print('off/on mon 1', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'vb', color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'vb', color='green', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'voc', color='blue', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'voc', color='cyan', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='magenta', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(322)
        plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='orange', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(323)
        plq(plt, S.mr, 'time', S.mr, 'soc', color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc', color='orange', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(324)
        plq(plt, S.mr, 'time', S.mr, 'ib_sel', color='red', linestyle='-')
        plq(plt, S.mr, 'time', S.mr, 'ib', add=-5, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib', add=-5, color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'ib_in_s', color='cyan', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'ib_charge', color='blue', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'ib_charge', color='orange', linestyle=':')
        plt.legend(loc=1)
        if not S.strict_overplot and hasattr(S.mr, 'vr'):
            plt.subplot(326)
            plq(plt, S.mr, 'time', S.mr, 'vr', color='green', linestyle='-')
            plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # 9 off/on soc
        plt.subplot(321)
        plt.suptitle(S.plot_title + ' off/on soc')
        print('off/on soc', end=':  ')
        plq(plt, S.mr, 'time', S.mr, 'qcrs', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'qcrs', color='magenta', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'qcrs_s', color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'qcrs_s', color='cyan', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'q_capacity', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'q_capacity', color='magenta', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'q_capacity', color='black', linestyle='-.', warn=S.run_type=='HistHist')
        plq(plt, S.sv, 'time', S.sv, 'q_capacity', color='orange', linestyle=':')
        plt.xlabel('sec')
        plt.legend(loc=2)
        plt.subplot(322)
        plq(plt, S.mr, 'time', S.mr, 'delta_q', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'delta_q', color='magenta', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'delta_q_s', color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'delta_q_s', color='cyan', linestyle=':')
        plt.legend(loc=2)
        plt.subplot(323)
        plq(plt, S.mr, 'time', S.mr, 'soc', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc', color='magenta', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'soc_s', color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'soc_s', color='cyan', linestyle=':')
        plt.xlabel('sec')
        plt.legend(loc=2)
        plt.subplot(324)
        plq(plt, S.mr, 'time', S.mr, 'ib_charge', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_charge', color='magenta', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'ib_charge_s', color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'ib_charge_s', color='cyan', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'reset', color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'reset', color='magenta', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'reset_s', color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'reset_s', color='red', linestyle=':')
        plq(plt, S.mr, 'time', S.mr, 'sat', add=2, color='blue', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'sat', add=2, color='red', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'saturated', add=2, color='black', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'saturated', add=2, color='orange', linestyle=':')
        plq(plt, S.sr, 'time', S.sr, 'sat_s', add=2, color='black', linestyle='-.')
        plq(plt, S.smv, 'time', S.smv, 'sat_s', add=2, color='red', linestyle=':')
        plt.legend(loc=2)
        plt.subplot(325)
        ymax = max([max(sublist) for sublist in [S.mr.Tb, S.mv.Tb, S.mr.Tb_f, S.mr.Tb_f, S.smv.Tb_f_s]])
        ymin = min([min(sublist) for sublist in [S.mr.Tb, S.mv.Tb, S.mr.Tb_f, S.mr.Tb_f, S.smv.Tb_f_s]])
        ymin_int = int(ymin)
        f_add = 2
        f_add_str = str(f_add)
        ymax_int = int(ymax) + 1 + f_add
        diff = ymax_int - ymin
        plq(plt, S.mr, 'time_t', S.mr, 'Tb_hdwe', color='black', linestyle='-', stairs=True)
        plq(plt, S.mv, 'time', S.mv, 'Tb_hdwe', color='green', linestyle='--')
        plq(plt, S.smv, 'time', S.smv, 'Tb_s', color='cyan', linestyle='-.')
        # plq(plt, S.mr, 'time', S.mr, 'Tb_rap', color='green', linestyle='--', label='Tb_mon' + run_str)
        plq(plt, S.mr, 'time_t', S.mr, 'Tb_f', add=f_add, color='black', linestyle='-.', stairs=True)
        plq(plt, S.mv, 'time', S.mv, 'Tb_f', add=f_add, color='orange', linestyle=':', stairs=True)
        plq(plt, S.sr, 'time', S.sr, 'Tb_f_s', add=f_add+.1, color='blue', linestyle='-')
        plq(plt, S.smv, 'time', S.smv, 'Tb_f_s', add=f_add+.1, color='red', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'reset_temp', add=int(ymax+f_add), slr=0.1*diff, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'reset_temp', add=int(ymax+f_add), slr=0.1*diff, color='green', linestyle='--')
        plt.xlabel('sec')
        plt.ylim(ymin_int, )
        plt.legend(loc=2)
        plt.subplot(326)
        from Battery import Battery
        import numpy as np
        S.mr.reset_temp_scl = np.array(S.mr.reset_temp) * Battery.T_RLIM
        S.mv.reset_temp_scl = np.array(S.mv.reset_temp) * Battery.T_RLIM
        plq(plt, S.mr, 'time_t', S.mr, 'Tb_f_rate', add=0.002, color='red', linestyle='-', stairs=True)
        plq(plt, S.mv, 'time', S.mv, 'Tb_f_rate', add=0.002, color='blue', linestyle='--', stairs=True)
        plq(plt, S.mr, 'time_t', S.mr, 'Tb_hdwe_filt_rate', color='black', linestyle='-', stairs=True)
        plq(plt, S.mv, 'time', S.mv, 'Tb_hdwe_filt_rate', color='green', linestyle='--', stairs=True)
        plq(plt, S.mr, 'time', S.mr, 'reset_temp_scl', add=-0.002, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'reset_temp_scl', add=-0.002, color='green', linestyle='--')
        plt.xlabel('sec')
        plt.legend(loc=2)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
