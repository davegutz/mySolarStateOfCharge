# PlotSimS - general purpose plotting, Off / On related
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
import matplotlib.pyplot as plt
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


def off_on_plot(mr, mv, sr, sv, smv, filename, fig_files=None, plot_title=None, fig_list=None,
                run_str='_run', ver_str='_ver', strict_overplot=False):
    print('off_on_plot', end=':  ')
    if sr and smv:
        fig_list.append(plt.figure())  # 7 off/on sim
        plt.subplot(321)
        plt.title(plot_title + ' off/on sim 1')
        print('off/on sim 1', end=':  ')
        plq(plt, mr, 'time', mr, 'vb_s', color='black', linestyle='-')
        plq(plt, sv, 'time', sv, 'vb', color='cyan', linestyle='--')
        plq(plt, smv, 'time', smv, 'vb_s', color='red', linestyle='-.')
        plq(plt, sr, 'time', sr, 'voc_s', color='blue', linestyle='-')
        plq(plt, sv, 'time', sv, 'voc', color='magenta', linestyle='--')
        plq(plt, smv, 'time', smv, 'voc_s', color='cyan', linestyle='-.')
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='black', linestyle='-')
        plq(plt, sv, 'time', sv, 'voc_stat', color='orange', linestyle='--')
        plq(plt, smv, 'time', smv, 'voc_stat_s', color='red', linestyle='-.')
        plt.legend(loc=1)
        plt.subplot(322)
        plq(plt, sr, 'time', sr, 'dv_hys_s', color='black', linestyle='-')
        plq(plt, sv, 'time', sv, 'dv_hys', color='orange', linestyle='--')
        plq(plt, smv, 'time', smv, 'dv_hys_s', color='cyan', linestyle='-.')
        plt.legend(loc=1)
        plt.subplot(323)
        plq(plt, sr, 'time', sr, 'soc_s', color='black', linestyle='-')
        plq(plt, sv, 'time', sv, 'soc', color='orange', linestyle='--')
        plq(plt, smv, 'time', smv, 'soc_s', color='red', linestyle='-.')
        plt.xlabel('sec')
        plt.legend(loc=1)
        plt.subplot(324)
        plq(plt, sr, 'time', sr, 'ib_s', color='black', linestyle='-')
        plq(plt, smv, 'time', smv, 'ib_s', color='orange', linestyle='--')
        plq(plt, sr, 'time', sr, 'ib_dyn_s', color='blue', linestyle='-')
        plq(plt, sv, 'time', sv, 'ib_dyn', color='red', linestyle='--')
        plt.xlabel('sec')
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # 8 off/on mon 1
        plt.subplot(321)
        plt.title(plot_title + ' off/on mon 1')
        print('off/on mon 1', end=':  ')
        plq(plt, mr, 'time', mr, 'vb', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'vb', color='green', linestyle='--')
        plq(plt, mr, 'time', mr, 'voc', color='blue', linestyle='-.')
        plq(plt, mv, 'time', mv, 'voc', color='cyan', linestyle=':')
        plq(plt, mr, 'time', mr, 'voc_stat', color='magenta', linestyle='-.')
        plq(plt, mv, 'time', mv, 'voc_stat', color='orange', linestyle=':')
        plt.legend(loc=1)
        plt.subplot(322)
        plq(plt, mr, 'time', mr, 'dv_hys', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'dv_hys', color='orange', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(323)
        plq(plt, mr, 'time', mr, 'soc', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'soc', color='orange', linestyle='--')
        plt.legend(loc=1)
        plt.subplot(324)
        plq(plt, mr, 'time', mr, 'ib_sel', color='red', linestyle='-')
        plq(plt, mr, 'time', mr, 'ib', add=-5, color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib', add=-5, color='red', linestyle='--')
        plq(plt, sr, 'time', sr, 'ib_in_s', color='cyan', linestyle='--')
        plq(plt, mr, 'time', mr, 'ib_charge', color='blue', linestyle='-.')
        plq(plt, mv, 'time', mv, 'ib_charge', color='orange', linestyle=':')
        plt.legend(loc=1)
        if not strict_overplot and hasattr(mr, 'vr'):
            plt.subplot(326)
            plq(plt, mr, 'time', mr, 'vr', color='green', linestyle='-')
            plt.legend(loc=1)

        fig_list.append(plt.figure())  # 9 off/on soc
        plt.subplot(321)
        plt.title(plot_title + ' off/on soc')
        print('off/on soc', end=':  ')
        plq(plt, mr, 'time', mr, 'qcrs', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'qcrs', color='magenta', linestyle='--')
        plq(plt, sr, 'time', sr, 'qcrs_s', color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'qcrs_s', color='cyan', linestyle=':')
        plq(plt, mr, 'time', mr, 'q_capacity', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'q_capacity', color='magenta', linestyle='--')
        plq(plt, sv, 'time', sv, 'q_capacity', color='black', linestyle=':')
        plt.xlabel('sec')
        plt.legend(loc=2)
        plt.subplot(322)
        plq(plt, mr, 'time', mr, 'delta_q', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'delta_q', color='magenta', linestyle='--')
        plq(plt, sr, 'time', sr, 'delta_q_s', color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'delta_q_s', color='cyan', linestyle=':')
        plt.legend(loc=2)
        plt.subplot(323)
        plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'soc', color='magenta', linestyle='--')
        plq(plt, sr, 'time', sr, 'soc_s', color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'soc_s', color='cyan', linestyle=':')
        plt.xlabel('sec')
        plt.legend(loc=2)
        plt.subplot(324)
        plq(plt, mr, 'time', mr, 'ib_charge', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'ib_charge', color='magenta', linestyle='--')
        plq(plt, sr, 'time', sr, 'ib_charge_s', color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'ib_charge_s', color='cyan', linestyle=':')
        plq(plt, mr, 'time', mr, 'reset', color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'reset', color='magenta', linestyle='--')
        plq(plt, sr, 'time', sr, 'reset_s', color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'reset_s', color='red', linestyle=':')
        plq(plt, mr, 'time', mr, 'sat', add=2, color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'sat', add=2, color='magenta', linestyle='--')
        plq(plt, sr, 'time', sr, 'sat_s', add=2, color='black', linestyle='-.')
        plq(plt, smv, 'time', smv, 'sat_s', add=2, color='red', linestyle=':')
        plt.legend(loc=2)
        plt.subplot(325)
        ymax = max([max(sublist) for sublist in [mr.Tb_rap, mr.Tb, mv.Tb, smv.Tb_s, mr.Tb_f, mr.Tb_f, smv.Tb_f_s]])
        ymin = min([min(sublist) for sublist in [mr.Tb_rap, mr.Tb, mv.Tb, smv.Tb_s, mr.Tb_f, mr.Tb_f, smv.Tb_f_s]])
        ymin_int = int(ymin)
        f_add = 2
        f_add_str = str(f_add)
        ymax_int = int(ymax) + 1 + f_add
        diff = ymax_int - ymin
        plq(plt, mr, 'time_t', mr, 'Tb_hdwe', color='black', linestyle='-', stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_hdwe', color='green', linestyle='--')
        plq(plt, smv, 'time', smv, 'Tb_s', color='cyan', linestyle='-.')
        # plq(plt, mr, 'time', mr, 'Tb_rap', color='green', linestyle='--', label='Tb_mon' + run_str)
        plq(plt, mr, 'time', mr, 'Tb_f_rap', add=f_add, color='blue', linestyle='-')
        plq(plt, mv, 'time', mv, 'Tb_f_rap', add=f_add, color='magenta', linestyle='--')
        plq(plt, mr, 'time_t', mr, 'Tb_f', add=f_add, color='black', linestyle='-.', stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_f', add=f_add, color='orange', linestyle=':', stairs=True)
        plq(plt, sr, 'time', sr, 'Tb_f_s', add=f_add+.1, color='blue', linestyle='-')
        plq(plt, smv, 'time', smv, 'Tb_f_s', add=f_add+.1, color='red', linestyle='--')
        plq(plt, mr, 'time', mr, 'reset_temp', add=int(ymax+f_add), slr=0.1*diff, color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'reset_temp', add=int(ymax+f_add), slr=0.1*diff, color='green', linestyle='--')
        plt.xlabel('sec')
        plt.ylim(ymin_int, )
        plt.legend(loc=2)
        plt.subplot(326)
        from Battery import Battery
        import numpy as np
        mr.reset_temp_scl = np.array(mr.reset_temp) * Battery.T_RLIM
        mv.reset_temp_scl = np.array(mv.reset_temp) * Battery.T_RLIM
        plq(plt, mr, 'time', mr, 'Tb_f_rate_rap', add=0.004, color='cyan', linestyle='-')
        plq(plt, mv, 'time', mv, 'Tb_f_rate_rap', add=0.004, color='orange', linestyle='--')
        plq(plt, mr, 'time_t', mr, 'Tb_f_rate', add=0.002, color='red', linestyle='-', stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_f_rate', add=0.002, color='blue', linestyle='--')
        plq(plt, mr, 'time_t', mr, 'Tb_hdwe_filt_rate', color='black', linestyle='-', stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_hdwe_filt_rate', color='green', linestyle='--')
        plq(plt, mr, 'time', mr, 'reset_temp_scl', add=-0.002, color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'reset_temp_scl', add=-0.002, color='green', linestyle='--')
        plt.xlabel('sec')
        plt.legend(loc=2)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
