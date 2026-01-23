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
        plq(plt, mr, 'time', mr, 'vb_s', color='black', linestyle='-', label='vb_s' + run_str)
        plq(plt, sv, 'time', sv, 'vb', color='cyan', linestyle='--', label='vb_s' + ver_str)
        plq(plt, smv, 'time', smv, 'vb_s', color='red', linestyle='-.', label='vb_s' + ver_str)
        plq(plt, sr, 'time', sr, 'voc_s', color='blue', linestyle='-', label='voc_s' + run_str)
        plq(plt, sv, 'time', sv, 'voc', color='magenta', linestyle='--', label='voc_s' + ver_str)
        plq(plt, smv, 'time', smv, 'voc_s', color='cyan', linestyle='-.', label='voc_s' + ver_str)
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='black', linestyle='-', label='voc_stat_s' + run_str)
        plq(plt, sv, 'time', sv, 'voc_stat', color='orange', linestyle='--', label='voc_stat_s' + ver_str)
        plq(plt, smv, 'time', smv, 'voc_stat_s', color='red', linestyle='-.', label='voc_stat_s' + ver_str)
        plt.legend(loc=1)
        plt.subplot(322)
        plt.plot(sr.time, sr.dv_hys_s, linestyle='-', color='black', label='dv_hys_s' + run_str)
        plq(plt, sv, 'time', sv, 'dv_hys', linestyle='--', color='orange', label='dv_hys' + ver_str)
        plq(plt, smv, 'time', smv, 'dv_hys_s', linestyle='-.', color='cyan', label='dv_hys_s' + ver_str)
        plt.legend(loc=1)
        plt.subplot(323)
        plt.plot(sr.time, sr.soc_s, linestyle='-', color='black', label='soc_s' + run_str)
        plq(plt, sv, 'time', sv, 'soc', linestyle='--', color='orange', label='soc_s' + ver_str)
        plq(plt, smv, 'time', smv, 'soc_s', linestyle='-.', color='red', label='soc_s' + ver_str)
        plt.xlabel('sec')
        plt.legend(loc=1)
        plt.subplot(324)
        plq(plt, sr, 'time', sr, 'ib_s', linestyle='-', color='black', label='ib_s' + run_str)
        plq(plt, smv, 'time', smv, 'ib_s', linestyle='--', color='orange', label='ib_s' + ver_str)
        plq(plt, sr, 'time', sr, 'ib_dyn_s', linestyle='-', color='blue', label='ib_dyn_s' + run_str)
        plq(plt, sv, 'time', sv, 'ib_dyn', linestyle='--', color='red', label='ib_dyn_s' + ver_str)
        plt.xlabel('sec')
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # 8 off/on mon 1
        plt.subplot(321)
        plt.title(plot_title + ' off/on mon 1')
        print('off/on mon 1', end=':  ')
        plt.plot(mr.time, mr.vb, color='black', linestyle='-', label='vb' + run_str)
        plt.plot(mv.time, mv.vb, color='green', linestyle='--', label='vb' + ver_str)
        plt.plot(mr.time, mr.voc, color='blue', linestyle='-.', label='voc' + run_str)
        plt.plot(mv.time, mv.voc, color='cyan', linestyle=':', label='voc' + ver_str)
        plt.plot(mr.time, mr.voc_stat, color='magenta', linestyle='-.', label='voc_stat' + run_str)
        plt.plot(mv.time, mv.voc_stat, color='orange', linestyle=':', label='voc_stat' + ver_str)
        plt.legend(loc=1)
        plt.subplot(322)
        plt.plot(mr.time, mr.dv_hys, linestyle='-', color='black', label='dv_hys' + run_str)
        plt.plot(mv.time, mv.dv_hys, linestyle='--', color='orange', label='dv_hys' + ver_str)
        plt.legend(loc=1)
        plt.subplot(323)
        plt.plot(mr.time, mr.soc, linestyle='-', color='black', label='soc' + run_str)
        plt.plot(mv.time, mv.soc, linestyle='--', color='orange', label='soc' + ver_str)
        plt.legend(loc=1)
        plt.subplot(324)
        plt.plot(mr.time, mr.ib_sel, linestyle='-', color='red', label='ib_sel' + run_str)
        plq(plt, mr, 'time', mr, 'ib', add=-5, linestyle='-', color='black', label='ib' + run_str + '-5')
        plq(plt, mv, 'time', mv, 'ib', add=-5, linestyle='--', color='red', label='ib' + ver_str + '-5')
        plt.plot(sr.time, sr.ib_in_s, linestyle='--', color='cyan', label='ib_in_s' + run_str)
        plt.plot(mr.time, mr.ib_charge, linestyle='-.', color='blue', label='ib_charge' + run_str)
        plt.plot(mv.time, mv.ib_charge, linestyle=':', color='orange', label='ib_charge' + ver_str)
        plt.legend(loc=1)
        if not strict_overplot and hasattr(mr, 'vr'):
            plt.subplot(326)
            plq(plt, mr, 'time', mr, 'vr', color='green', linestyle='-', label='vr' + run_str)
            plt.legend(loc=1)

        fig_list.append(plt.figure())  # 9 off/on soc
        plt.subplot(321)
        plt.title(plot_title + ' off/on soc')
        print('off/on soc', end=':  ')
        plq(plt, mr, 'time', mr, 'qcrs', color='blue', linestyle='-', label='qcrs' + run_str)
        plq(plt, mv, 'time', mv, 'qcrs', color='magenta', linestyle='--', label='qcrs' + ver_str)
        plq(plt, sr, 'time', sr, 'qcrs_s', color='black', linestyle='-.', label='qcrs_s' + run_str)
        plq(plt, smv, 'time', smv, 'qcrs_s', color='cyan', linestyle=':', label='qcrs_s' + ver_str)
        plq(plt, mr, 'time', mr, 'q_capacity', color='blue', linestyle='-', label='q_capacity' + run_str)
        plq(plt, mv, 'time', mv, 'q_capacity', color='magenta', linestyle='--', label='q_capacity' + ver_str)
        plq(plt, sv, 'time', sv, 'q_capacity', color='black', linestyle=':', label='q_capacity_s' + ver_str)
        plt.xlabel('sec')
        plt.legend(loc=2)
        plt.subplot(322)
        plq(plt, mr, 'time', mr, 'delta_q', color='blue', linestyle='-', label='delta_q' + run_str)
        plq(plt, mv, 'time', mv, 'delta_q', color='magenta', linestyle='--', label='delta_q' + ver_str)
        plq(plt, sr, 'time', sr, 'delta_q_s', color='black', linestyle='-.', label='delta_q_s' + run_str)
        plq(plt, smv, 'time', smv, 'delta_q_s', color='cyan', linestyle=':', label='delta_q_s' + ver_str)
        plt.legend(loc=2)
        plt.subplot(323)
        plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-', label='soc' + run_str)
        plq(plt, mv, 'time', mv, 'soc', color='magenta', linestyle='--', label='soc' + ver_str)
        plq(plt, sr, 'time', sr, 'soc_s', color='black', linestyle='-.', label='soc_s' + run_str)
        plq(plt, smv, 'time', smv, 'soc_s', color='cyan', linestyle=':', label='soc_s' + ver_str)
        plt.xlabel('sec')
        plt.legend(loc=2)
        plt.subplot(324)
        plq(plt, mr, 'time', mr, 'ib_charge', color='blue', linestyle='-', label='ib_charge' + run_str)
        plq(plt, mv, 'time', mv, 'ib_charge', color='magenta', linestyle='--', label='ib_charge' + ver_str)
        plq(plt, sr, 'time', sr, 'ib_charge_s', color='black', linestyle='-.', label='ib_charge_s' + run_str)
        plq(plt, smv, 'time', smv, 'ib_charge_s', color='cyan', linestyle=':', label='ib_charge_s' + ver_str)
        plq(plt, mr, 'time', mr, 'reset', color='blue', linestyle='-', label='reset' + run_str)
        plq(plt, mv, 'time', mv, 'reset', color='magenta', linestyle='--', label='reset' + ver_str)
        plq(plt, sr, 'time', sr, 'reset_s', color='black', linestyle='-.', label='reset_s' + run_str)
        plq(plt, smv, 'time', smv, 'reset_s', color='red', linestyle=':', label='reset_s' + ver_str)
        plq(plt, mr, 'time', mr, 'sat', add=2, color='blue', linestyle='-', label='sat' + run_str + ' +2')
        plq(plt, mv, 'time', mv, 'sat', add=2, color='magenta', linestyle='--', label='sat' + ver_str + ' +2')
        plq(plt, sr, 'time', sr, 'sat_s', add=2, color='black', linestyle='-.', label='sat_s' + run_str + ' +2')
        plq(plt, smv, 'time', smv, 'sat_s', add=2, color='red', linestyle=':', label='sat_s' + ver_str + ' +2')
        plt.legend(loc=2)
        plt.subplot(325)
        ymax = max([max(sublist) for sublist in [mr.Tb_rap, mr.Tb, mv.Tb, smv.Tb_s, mr.Tb_f, mr.Tb_f, smv.Tb_f_s]])
        ymin = min([min(sublist) for sublist in [mr.Tb_rap, mr.Tb, mv.Tb, smv.Tb_s, mr.Tb_f, mr.Tb_f, smv.Tb_f_s]])
        ymin_int = int(ymin)
        f_add = 2
        f_add_str = str(f_add)
        ymax_int = int(ymax) + 1 + f_add
        diff = ymax_int - ymin
        plq(plt, mr, 'time_t', mr, 'Tb_hdwe', color='black', linestyle='-', label='Tb_hdwe' + run_str, stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_hdwe', color='green', linestyle='--', label='Tb_hdwe' + ver_str)
        plq(plt, smv, 'time', smv, 'Tb_s', color='cyan', linestyle='-.', label='Tb_s' + ver_str)
        # plq(plt, mr, 'time', mr, 'Tb_rap', color='green', linestyle='--', label='Tb_mon' + run_str)
        plq(plt, mr, 'time', mr, 'Tb_f_rap',  add=f_add, color='blue', linestyle='-', label='Tb_f_rap' + run_str + ' +' + f_add_str)
        plq(plt, mv, 'time', mv, 'Tb_f_rap',  add=f_add, color='magenta', linestyle='--', label='Tb_f_rap' + ver_str + ' +' + f_add_str)
        plq(plt, mr, 'time_t', mr, 'Tb_f',  add=f_add, color='black', linestyle='-.', label='Tb_f' + run_str + ' +' + f_add_str, stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_f',  add=f_add, color='orange', linestyle=':', label='Tb_f' + ver_str + ' +' + f_add_str, stairs=True)
        plq(plt, sr, 'time', sr, 'Tb_f_s', add=f_add+.1, color='blue', linestyle='-', label='Tb_f_s' + run_str + ' +' + f_add_str)
        plq(plt, smv, 'time', smv, 'Tb_f_s', add=f_add+.1, color='red', linestyle='--', label='Tb_f_s' + ver_str + ' +' + f_add_str)
        plq(plt, mr, 'time', mr, 'reset_temp', slr=0.1*diff, add=int(ymax+f_add), color='black', linestyle='-', label='reset_temp' + run_str)
        plq(plt, mv, 'time', mv, 'reset_temp', slr=0.1*diff, add=int(ymax+f_add), color='green', linestyle='--', label='reset_temp' + ver_str)
        plt.xlabel('sec')
        plt.ylim(ymin_int, )
        plt.legend(loc=2)
        plt.subplot(326)
        from Battery import Battery
        import numpy as np
        mr.reset_temp_scl = np.array(mr.reset_temp) * Battery.T_RLIM
        mv.reset_temp_scl = np.array(mv.reset_temp) * Battery.T_RLIM
        plq(plt, mr, 'time', mr, 'Tb_f_rate_rap', add=0.004, color='cyan', linestyle='-', label='Tb_f_rate_rap' + run_str + '+ 0.004')
        plq(plt, mv, 'time', mv, 'Tb_f_rate_rap', add=0.004, color='orange', linestyle='--', label='Tb_f_rate_rap' + ver_str + '+ 0.004')
        plq(plt, mr, 'time_t', mr, 'Tb_f_rate', add=0.002, color='red', linestyle='-', label='Tb_f_rate' + run_str + '+ 0.002', stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_f_rate', add=0.002, color='blue', linestyle='--', label='Tb_f_rate' + ver_str + '+ 0.002')
        plq(plt, mr, 'time_t', mr, 'Tb_hdwe_filt_rate', color='black', linestyle='-', label='Tb_hdwe_filt_rate' + run_str, stairs=True)
        plq(plt, mv, 'time', mv, 'Tb_hdwe_filt_rate', color='green', linestyle='--', label='Tb_hdwe_filt_rate' + ver_str)
        plq(plt, mr, 'time', mr, 'reset_temp_scl', add=-0.002, color='black', linestyle='-', label='reset_temp' + run_str + '- 0.002')
        plq(plt, mv, 'time', mv, 'reset_temp_scl', add=-0.002, color='green', linestyle='--', label='reset_temp' + ver_str + '- 0.002')
        plt.xlabel('sec')
        plt.legend(loc=2)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
