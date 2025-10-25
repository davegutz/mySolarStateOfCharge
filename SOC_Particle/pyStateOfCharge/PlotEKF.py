# PlotSimS - general purpose plotting, EKF related
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
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})


def ekf_plot(mr, mv, sr, sv, smv, filename, fig_files=None, plot_title=None, fig_list=None,
             run_str='_run', ver_str='_ver'):
    if sr and smv:
        if mr.Fx is not None:  # ekf
            fig_list.append(plt.figure())  # EKF  1
            plt.subplot(331)
            plt.title(plot_title + ' EKF 1')
            plq(plt, mr, 'time_e', mr, 'u', color='blue', linestyle='-', label='u' + run_str, stairs=True)
            plt.plot(mv.time, mv.u_ekf, color='red', linestyle='--', label='u' + ver_str)
            plt.legend(loc=1)
            plt.subplot(332)
            plq(plt, mr, 'time_e', mr, 'z', color='blue', linestyle='-', label='z' + run_str, stairs=True)
            plt.plot(mv.time, mv.z_ekf, color='red', linestyle='--', label='z' + ver_str)
            plt.legend(loc=1)
            plt.subplot(333)
            plt.plot(smv.time, smv.reset_s, color='green', linestyle='-', label='reset_s' + ver_str)
            plt.plot(sr.time, sr.sat_s, color='blue', linestyle='--', label='sat_s' + run_str)
            plt.plot(smv.time, smv.sat_s, color='red', linestyle='-.', label='sat_s' + ver_str)
            plt.plot(mv.time, mv.reset_ekf, color='orange', linestyle=':', label='reset_ekf' + ver_str)
            plt.legend(loc=1)
            plt.subplot(334)
            plq(plt, mr, 'time_e', mr, 'Fx', color='blue', linestyle='-', label='Fx' + run_str, stairs=True)
            plt.plot(mv.time, mv.Fx, color='red', linestyle='--', label='Fx' + ver_str)
            plt.legend(loc=1)
            plt.subplot(335)
            plq(plt, mr, 'time_e', mr, 'Bu', color='blue', linestyle='-', label='Bu' + run_str, stairs=True)
            plt.plot(mv.time, mv.Bu, color='red', linestyle='--', label='Bu' + ver_str)
            plt.legend(loc=1)
            plt.subplot(336)
            plq(plt, mr, 'time_e', mr, 'Q', color='blue', linestyle='-', label='Q' + run_str, stairs=True)
            plt.plot(mv.time, mv.Q, color='red', linestyle='--', label='Q' + ver_str)
            plt.legend(loc=1)
            plt.subplot(337)
            plq(plt, mr, 'time_e', mr, 'R', color='blue', linestyle='-', label='R' + run_str, stairs=True)
            plt.plot(mv.time, mv.R, color='red', linestyle='--', label='R' + ver_str)
            plt.legend(loc=1)
            plt.subplot(338)
            plq(plt, mr, 'time_e', mr, 'P', color='blue', linestyle='-', label='P' + run_str, stairs=True)
            plt.plot(mv.time, mv.P, color='red', linestyle='--', label='P' + ver_str)
            plt.legend(loc=1)
            plt.subplot(339)
            plq(plt, mr, 'time_e', mr, 'S', color='blue', linestyle='-', label='S' + run_str, stairs=True)
            plt.plot(mv.time, mv.S, color='red', linestyle='--', label='S' + ver_str)
            plt.legend(loc=1)
            fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            plt.savefig(fig_file_name, format="png")

            fig_list.append(plt.figure())  # EKF  2
            plt.subplot(331)
            plt.title(plot_title + ' EKF 2')
            plq(plt, mr, 'time_e', mr, 'K', color='blue', linestyle='-', label='K' + run_str, stairs=True)
            plt.plot(mv.time, mv.K, color='red', linestyle='--', label='K' + ver_str)
            plt.legend(loc=1)
            plt.subplot(332)
            plq(plt, mr, 'time_e', mr, 'x', color='blue', linestyle='-', label='x' + run_str, stairs=True)
            plt.plot(mv.time, mv.x_ekf, color='red', linestyle='--', label='x' + ver_str)
            plq(plt, mr, 'time', mr, 'soc_ekf', color='cyan', linestyle='-.', label='x=soc_ekf' + run_str, stairs=True)
            plt.plot(mv.time, mv.soc_ekf, color='orange', linestyle=':', label='x=soc_ekf' + ver_str)
            plt.legend(loc=1)
            plt.subplot(333)
            plq(plt, mr, 'time_e', mr, 'y', color='blue', linestyle='-', label='y' + run_str, stairs=True)
            plt.plot(mv.time, mv.y_ekf, color='red', linestyle='--', label='y' + ver_str)
            plt.legend(loc=1)
            plt.subplot(334)
            plq(plt, mr, 'time_e', mr, 'x_prior', color='blue', linestyle='-', label='x_prior' + run_str, stairs=True)
            plt.plot(mv.time, mv.x_prior, color='red', linestyle='--', label='x_prior' + ver_str)
            plt.legend(loc=1)
            plt.subplot(335)
            plq(plt, mr, 'time_e', mr, 'P_prior', color='blue', linestyle='-', label='P_prior' + run_str, stairs=True)
            plt.plot(mv.time, mv.P_prior, color='red', linestyle='--', label='P_prior' + ver_str)
            plt.legend(loc=1)
            plt.subplot(336)
            plq(plt, mr, 'time_e', mr, 'x_post', color='blue', linestyle='-', label='x_post' + run_str, stairs=True)
            plt.plot(mv.time, mv.x_post, color='red', linestyle='--', label='x_post' + ver_str)
            plt.legend(loc=1)
            plt.subplot(337)
            plq(plt, mr, 'time_e', mr, 'P_post', color='blue', linestyle='-', label='P_post' + run_str, stairs=True)
            plt.plot(mv.time, mv.P_post, color='red', linestyle='--', label='P_post' + ver_str)
            plt.legend(loc=1)
            plt.subplot(338)
            # plq(plt, mr, 'time', mr, 'voc_stat', color='blue', linestyle='-', label='voc_stat' + run_str)
            plq(plt, mv, 'time', mv, 'voc_stat_ekf', color='magenta', linestyle='-.', label='voc_stat_ekf' + ver_str)
            plq(plt, mr, 'time_e', mr, 'z', color='blue', linestyle='-', label='z' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'z_ekf', color='red', linestyle='--', label='z' + ver_str)
            plq(plt, mr, 'time_e', mr, 'hx', color='cyan', linestyle='-', label='hx' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'hx', color='orange', linestyle='--', label='hx' + ver_str)
            plt.legend(loc=1)
            plt.subplot(339)
            plq(plt, mr, 'time_e', mr, 'H', color='blue', linestyle='-', label='H' + run_str, stairs=True)
            plt.plot(mv.time, mv.H, color='red', linestyle='--', label='H' + ver_str)
            plt.legend(loc=1)
            fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            plt.savefig(fig_file_name, format="png")

            fig_list.append(plt.figure())  # EKF2a
            plt.subplot(311)
            plt.title(plot_title + ' EKF 2a')
            plq(plt, mr, 'time', mr, 'voc_stat', add=-0.0, color='red', linestyle='-', label='voc_stat-0.0' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'voc_stat', add=-0.0,  color='black', linestyle='--', label='voc_stat-0.0' + ver_str, stairs=True)
            plq(plt, mr, 'time_e', mr, 'z', color='cyan', linestyle='-', label='z=voc_stat_f' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'z_ekf', color='orange', linestyle='--', label='z=voc_stat_f' + ver_str, stairs=True)
            plq(plt, mr, 'time_e', mr, 'hx', color='magenta', linestyle='-', label='hx(x)' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'hx', color='green', linestyle='--', label='hx(x)' + ver_str, stairs=True)
            plt.legend(loc=1)
            plt.subplot(312)
            plq(plt, mr, 'x', mr, 'hx', color='red', linestyle='-', label='hx(x)' + run_str)
            plq(plt, mv, 'x_ekf', mv, 'hx', color='black', linestyle='--', label='hx(x)' + ver_str)
            plt.legend(loc=1)
            plt.subplot(313)
            plq(plt, mr, 'time', mr, 'dt', color='red', linestyle='-', label='dt' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'dt', color='black', linestyle='--', label='dt' + ver_str, stairs=True)
            plq(plt, mr, 'time_e', mr, 'dt_ekf', color='blue', linestyle='-.', label='dt_eframe' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'dt_eframe', color='orange', linestyle=':', label='dt_eframe' + ver_str, stairs=True)
            plt.legend(loc=1)
            fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            plt.savefig(fig_file_name, format="png")

            fig_list.append(plt.figure())  # EKF3
            plt.subplot(221)
            plt.title(plot_title + ' EKF 3')
            plt.plot(mr.time, mr.ib, color='red', linestyle='-', label='ib' + run_str)
            plt.plot(mv.time, mv.ib, color='black', linestyle='--', label='ib' + ver_str)
            plq(plt, mr, 'time_e', mr, 'u', color='cyan', linestyle='-.', label='u' + run_str, stairs=True)
            plt.plot(mv.time, mv.u_ekf, color='orange', linestyle=':', label='u' + ver_str)
            plt.legend(loc=1)
            plt.subplot(222)
            plq(plt, mr, 'time', mr, 'vb', color='red', linestyle='-', label='vb' + run_str)
            plq(plt, mv, 'time', mv, 'vb', color='black', linestyle='--', label='vb' + ver_str)
            plq(plt, mr, 'time_e', mr, 'z', color='cyan', linestyle='-.', label='z=voc_stat_f' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'z_ekf', color='orange', linestyle='-.', label='z=voc_stat_f' + ver_str)
            plt.legend(loc=1)
            plt.subplot(223)
            plt.plot(mr.time, mr.soc, color='red', linestyle='-', label='soc' + run_str)
            plt.plot(mv.time, mv.soc, color='black', linestyle='--', label='soc' + ver_str)
            plq(plt, mr, 'time', mr, 'soc_ekf', color='cyan', linestyle='-.', label='x=soc_ekf' + run_str, stairs=True)
            plt.plot(mv.time, mv.soc_ekf, color='orange', linestyle=':', label='x=soc_ekf' + ver_str)
            plt.legend(loc=1)
            plt.subplot(224)
            plq(plt, mr, 'time', mr, 'voc_ekf', color='red', linestyle='-', label='voc_ekf(soc) = hx' + run_str,
                stairs=True)
            plq(plt, mv, 'time', mv, 'voc_ekf', color='black', linestyle='--', label='voc_ekf(soc) = hx' + ver_str)
            plq(plt, mr, 'time_e', mr, 'z', color='cyan', linestyle='-.', label='z=voc_stat_f' + run_str, stairs=True)
            plq(plt, mv, 'time', mv, 'z_ekf', color='orange', linestyle='-.', label='z=voc_stat_f' + ver_str)
            plt.legend(loc=1)

    if mr.voc_soc is not None:
        fig_list.append(plt.figure())  # EKF  4
        plt.subplot(111)
        plt.title(plot_title + ' EKF 4')
        plt.plot(mr.soc, mr.voc_stat, color='red', linestyle='-', label='voc_stat' + run_str)
        plt.plot(mr.soc, mr.voc_soc, color='black', linestyle=':', label='voc_soc' + run_str)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # Hyst 1
        plt.subplot(331)
        plt.title(plot_title + ' Hyst 1')
        # plt.plot(mr.time, mr.dv_hys_required, linestyle='-', color='black', label='dv_hys_required'+run_str)
        plq(plt, mr, 'time', mr, 'e_wrap', slr=-1., linestyle='-', color='red', label='-e_wrap' + run_str)
        plq(plt, mv, 'time', mr, 'e_wrap', slr=-1., linestyle='--', color='blue', label='-e_wrap' + ver_str)
        plq(plt, mr, 'time', mr, 'e_wrap_filt', slr=-1., linestyle='-', color='black', label='-e_wrap_filt' + run_str)
        plq(plt, mv, 'time', mv, 'e_wrap_filt', slr=-1., linestyle='--', color='cyan', label='-e_wrap_filt' + run_str)
        plt.plot(mr.time, mr.dv_hys, linestyle='-.', color='orange', label='dv_hys' + run_str)
        plt.plot(mv.time, mv.dv_hys, marker='.', markersize='1', markevery=48, linestyle='None', color='magenta',
                 label='dv_hys' + ver_str)
        plq(plt, sr, 'time', sr, 'dv_hys_s', linestyle=':', color='cyan', label='dv_hys_s' + run_str)
        plq(plt, sv, 'time', smv, 'dv_hys', marker='.', markersize='1', markevery=64, linestyle='None',
            color='black', label='dv_hys_s' + ver_str)
        plq(plt, sv, 'time', sv, 'dv_hys_s', marker='.', markersize='1', markevery=64, linestyle='None',
            color='black', label='dv_hys_s' + ver_str)
        plt.xlabel('sec')
        plt.legend(loc=4)
        plt.subplot(332)
        # plt.plot(mr.soc, mr.dv_hys_required, linestyle='-', color='black', label='dv_hys_required'+run_str)
        plq(plt, mr, 'soc', mr, 'e_wrap', slr=-1., linestyle='--', color='red', label='-e_wrap' + run_str)
        plt.plot(mr.soc, mr.dv_hys, linestyle='-.', color='orange', label='dv_hys' + run_str)
        plt.plot(mv.soc, mv.dv_hys, marker='.', markersize='1', markevery=4, linestyle='None', color='magenta',
                 label='dv_hys' + ver_str)
        plq(plt, sr, 'soc_s', sr, 'dv_hys_s', linestyle=':', color='cyan', label='dv_hys_s' + run_str)
        plq(plt, sv, 'soc', sv, 'dv_hys', marker='.', markersize='1', markevery=5, linestyle='None',
            color='black', label='dv_hys_s' + ver_str)
        plq(plt, sv, 'soc_s', sv, 'dv_hys_s', marker='.', markersize='1', markevery=5, linestyle='None',
            color='black', label='dv_hys_s' + ver_str)
        plt.xlabel('soc')
        plt.legend(loc=4)
        plt.subplot(333)
        plt.plot(mr.time, mr.soc, linestyle='-', color='green', label='soc' + run_str)
        plq(plt, sr, 'time', sr, 'soc_s', linestyle='--', color='blue', label='soc_s' + run_str)
        plt.plot(mv.time, mv.soc, linestyle='-.', color='red', label='soc' + ver_str)
        plq(plt, sv, 'time', sv, 'soc', linestyle=':', color='cyan', label='soc_s' + ver_str)
        plq(plt, sv, 'time', sv, 'soc_s', linestyle=':', color='cyan', label='soc_s' + ver_str)
        plt.xlabel('sec')
        plt.legend(loc=4)
        plt.subplot(334)
        plq(plt, mr, 'time', mr, 'ib_sel', linestyle='-', color='black', label='ib_sel' + run_str)
        plt.plot(mr.time, mr.ioc, linestyle='--', color='cyan', label='ioc' + run_str)
        plt.xlabel('sec')
        plt.legend(loc=4)
        plt.subplot(335)
        plq(plt, mr, 'soc', mr, 'ib_sel', linestyle='-', color='black', label='ib_sel' + run_str)
        plt.plot(mr.soc, mr.ioc, linestyle='--', color='cyan', label='ioc' + run_str)
        plt.xlabel('soc')
        plt.legend(loc=4)
        plt.subplot(336)
        plq(plt, sr, 'time', sr, 'vb_s', color='black', linestyle='-', label='vb_s' + run_str)
        plt.plot(mr.time, mr.vb, color='orange', linestyle='--', label='vb' + run_str)
        plq(plt, sr, 'time', sr, 'voc_stat_s', color='magenta', linestyle='-', label='voc_stat_s' + run_str)
        plt.plot(mr.time, mr.voc_stat, color='pink', linestyle='--', label='voc_stat' + run_str)
        plt.plot(mr.time, mr.voc_soc, marker='.', markersize='1', markevery=32, linestyle='None', color='black',
                 label='voc_soc' + run_str)
        plt.legend(loc=1)
        plt.subplot(337)
        plt.plot(mr.time, mr.vb, linestyle='-', color='green', label='vb' + run_str)
        plt.plot(mr.time, mr.voc, linestyle='--', color='red', label='voc' + run_str)
        plt.plot(mr.time, mr.voc_stat, linestyle='-.', color='pink', label='voc_stat' + run_str)
        plt.plot(mr.time, mr.voc_soc, marker='.', markersize='1', markevery=32, linestyle='None', color='black',
                 label='voc_soc' + run_str)
        plt.xlabel('sec')
        plt.legend(loc=4)
        plt.subplot(338)
        plt.plot(mr.soc, mr.vb, linestyle='-', color='green', label='vb' + run_str)
        plt.plot(mr.soc, mr.voc, linestyle='--', color='red', label='voc' + run_str)
        plt.plot(mr.soc, mr.voc_stat, linestyle='-.', color='blue', label='voc_stat' + run_str)
        plt.plot(mr.soc, mr.voc_soc, linestyle=':', color='black', label='voc_soc' + run_str)
        plt.xlabel('soc')
        plt.legend(loc=4)
        plt.subplot(339)
        plt.plot(mr.time, mr.vb, color='green', linestyle='-', label='vb' + run_str)
        plt.plot(mv.time, mv.vb, color='orange', linestyle='--', label='vb' + ver_str)
        plt.plot(mr.time, mr.voc, color='blue', linestyle='-.', label='voc' + run_str)
        plt.plot(mv.time, mv.voc, marker='+', markersize='3', markevery=4, linestyle='None', color='black',
                 label='voc' + ver_str)
        plt.plot(mr.time, mr.voc_stat, color='magenta', linestyle='-', label='voc_stat' + run_str)
        plt.plot(mv.time, mv.voc_stat, color='pink', linestyle='--', label='voc_stat' + ver_str)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")
    return fig_list, fig_files
