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

""" General data-over-model
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""
import matplotlib.pyplot as plt
from plot.plq import plq as plq
from plot.PlotOptions import PlotOptions

def init_1(S:PlotOptions, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []
    fig_list.append(plt.figure())
    plt.subplot(221)
    plt.suptitle(S.plot_title + ' init 1')
    print('init 1', end=':  ')
    plq(plt, S.sr, 'time', S.sr, 'reset_s', color='black', linestyle='-')
    plq(plt, S.smv, 'time', S.smv, 'reset_s', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'reset', color='magenta', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'reset', color='cyan', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, S.sr, 'time', S.sr, 'Tb_s', color='black', linestyle='-')
    plq(plt, S.sv, 'time', S.sv, 'Tb', color='red', linestyle='--')
    plq(plt, S.mr, 'time_t', S.mr, 'Tb', color='blue', linestyle='-.', stairs=True)
    plq(plt, S.mv, 'time', S.mv, 'Tb', color='green', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, S.sr, 'time', S.sr, 'soc_s', color='black', linestyle='-')
    plq(plt, S.sv, 'time', S.sv, 'soc', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'soc', color='blue', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='green', linestyle=':')
    plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='orange', linestyle='None', marker='^', markersize='5', markevery=32)
    plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='cyan', linestyle='None', marker='+', markersize='5', markevery=32)
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:

        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files

def init_1a(S:PlotOptions, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())
    plt.subplot(221)
    plt.suptitle(S.plot_title + ' 1a')
    print('1a', end=':  ')
    if hasattr(S.mr, 'mod_data') and S.mr.mod_data[0] != 0 and S.strict_overplot:
        plq(plt, S.mr, 'time', S.mr, 'ib_amp_model', color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_amp_model', color='red', linestyle='-.')
        plq(plt, S.mr, 'time', S.mr, 'ib_noa_model', color='green', linestyle='--')
        plq(plt, S.mv, 'time', S.mv, 'ib_noa_model', color='blue', linestyle=':')
    else:
        plq(plt, S.mr, 'time', S.mr, 'ib_amp_hdwe', color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_amp_hdwe', color='red', linestyle='-.')
        plq(plt, S.mr, 'time', S.mr, 'ib_noa_hdwe', color='green', linestyle='--')
        plq(plt, S.mv, 'time', S.mv, 'ib_noa_hdwe', color='blue', linestyle=':')
    if not S.strict_overplot:
        plq(plt, S.mr, 'time', S.mr, 'ib_sel', add=+0, color='red', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_sel', add=+0, color='black', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_charge', add=+0, color='orange', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'ib_charge', add=+0, color='blue', linestyle=':')
    plt.legend(loc=1)
    if not S.strict_overplot and hasattr(S.mr, 'ib_sel_stat'):
        plt.subplot(222)
        plq(plt, S.mr, 'time', S.mr, 'ib_choice', color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_choice', color='red', linestyle='--', warn=False)
        plq(plt, S.mr, 'time', S.mr, 'ib_dec', add=2, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_dec', add=2, color='red', linestyle='--', warn=False)
        plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap', color='black', linestyle='-', stairs=True)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap', color='red', linestyle='--', stairs=True)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_filt', color='black', linestyle='-.', stairs=True)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_filt', color='red', linestyle=':', stairs=True)
    plq(plt, S.mr, 'time', S.mr, 'y_ekf', slr=-1, color='green', linestyle='-.', stairs=True)
    plq(plt, S.mv, 'time', S.mv, 'y_ekf', slr=-1, color='orange', linestyle=':', stairs=True)
    plq(plt, S.mr, 'time', S.mr, 'y_ekf_f', slr=-1, color='black', linestyle='-.', stairs=True)
    plq(plt, S.mv, 'time', S.mv, 'y_ekf_f', slr=-1, color='orange', linestyle=':', stairs=True)
    plt.legend(loc=1)
    if not S.strict_overplot:
        plt.subplot(224)
        plq(plt, S.mr, 'time', S.mr, 'cc_dif', color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'cc_dif', color='red', linestyle='--', warn=False)
        plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files

def dom_2(S:PlotOptions, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(321)
    plt.suptitle(S.plot_title + ' DOM 2')
    print('DOM 2', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'dv_dyn', color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'dv_dyn_f', color='green', linestyle='-', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'dv_dyn', color='orange', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'dv_dyn_f', color='orange', linestyle='--', warn=False)
    plt.legend(loc=1)
    plt.subplot(322)
    plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='orange', linestyle='--', warn=not S.ver_is_stdy)
    if not S.strict_overplot:
        plq(plt, S.mr, 'time', S.mr, 'voc_stat_f', color='green', linestyle='-.', warn=False)
        plq(plt, S.mv, 'time', S.mv, 'voc_stat_f', color='red', linestyle=':', warn=not S.ver_is_run)
    plt.legend(loc=1)
    plt.subplot(323)
    plq(plt, S.mr, 'time', S.mr, 'voc', color='green', linestyle='-', stairs=True, warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'voc_f', color='green', linestyle='-', stairs=True, warn=S.run_type=='HistHist')
    plq(plt, S.mr, 'time', S.mr, 'voc_d', color='green', linestyle='-', stairs=True, warn=False)
    plq(plt, S.mv, 'time', S.mv, 'voc', color='orange', linestyle='--', stairs=True, warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'voc_f', color='orange', linestyle='--', stairs=True, warn=S.run_type=='HistHist')
    plq(plt, S.mr, 'time', S.mr, 'voc_ekf', color='blue', linestyle='-.', stairs=True)
    plq(plt, S.mv, 'time', S.mv, 'voc_ekf', color='red', linestyle=':', stairs=True)
    plt.legend(loc=1)
    plt.subplot(324)
    plq(plt, S.mr, 'time', S.mr, 'y_ekf', color='green', linestyle='-', stairs=True, warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'y_ekf', color='orange', linestyle='--', stairs=True, warn=not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'y_ekf_f', color='blue', linestyle='-.', stairs=True, warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'y_ekf_f', color='orange', linestyle=':', stairs=True, warn=not S.run_is_run)
    plt.legend(loc=1)
    plt.subplot(325)
    plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='cyan', linestyle='--')
    if not S.strict_overplot:
        plq(plt, S.smv, 'time', S.smv, 'dv_hys_s', add=0.1, color='red', linestyle='-', warn=False)
        plq(plt, S.sr, 'time', S.sr, 'dv_hys_s', add=-0.1, color='magenta', linestyle='-', warn=False)
    plt.legend(loc=1)
    plt.subplot(326)
    plq(plt, S.mr, 'time_t', S.mr, 'Tb', color='green', linestyle='-', stairs=True, warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time_t', S.mr, 'Tb_f', color='green', linestyle='-', stairs=True)
    plq(plt, S.mv, 'time_t', S.mv, 'Tb', color='orange', linestyle='--', stairs=True, warn=not S.ver_is_run and not S.ver_is_sim and not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'Tb', color='red', linestyle='-.', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mv, 'time', S.mv, 'Tb_f', color='orange', linestyle='--', stairs=True, warn=not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'chm', color='black', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'chm', color='orange', linestyle='--')
    plq(plt, S.sr, 'time', S.sr, 'chm_s', color='cyan', linestyle='--', warn=not S.run_is_stdy)
    plt.ylim(0., 50.)
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files

def dom_3(S:PlotOptions, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(221)
    plt.suptitle(S.plot_title + ' DOM 3')
    print('DOM 3', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'soc', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='red', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(222)
    plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='orange', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(223)
    plq(plt, S.mr, 'time', S.mr, 'soc_s', color='blue', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'soc_s', color='red', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(224)
    plq(plt, S.mr, 'time', S.mr, 'soc', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'soc_s', color='green', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'soc_s', color='orange', linestyle=':')
    plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='cyan', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='black', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files

def dom_4(S:PlotOptions, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(131)
    plt.suptitle(S.plot_title + ' DOM 4')
    print('DOM 4', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'soc', color='orange', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='green', linestyle='--')
    plq(plt, S.sr, 'time', S.sr, 'soc_s', color='red', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'soc_s', color='black', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='cyan', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(132)
    plq(plt, S.mr, 'time', S.mr, 'vb', color='orange', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'vb_f', color='orange', linestyle='-', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'vb_hdwe', color='cyan', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'vb', color='green', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'vb_f', color='green', linestyle='-.', warn=S.run_type=='HistHist')
    plq(plt, S.mr, 'time', S.mr, 'vb_s', color='red', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'vb_s', color='black', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plt.legend(loc=1)
    plt.subplot(133)
    plq(plt, S.mr, 'soc', S.mr, 'vb', color='orange', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'soc', S.mr, 'vb_f', color='orange', linestyle='-', warn=False)
    plq(plt, S.mr, 'soc', S.mr, 'vb_hdwe', color='cyan', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'soc', S.mv, 'vb', color='green', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'soc', S.mv, 'vb_f', color='green', linestyle='-.', warn=S.run_type=='HistHist')
    plq(plt, S.mr, 'soc_s', S.mr, 'vb_s', color='red', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'soc_s', S.smv, 'vb_s', color='black', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files

def dom_4a(S:PlotOptions, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(311)
    plt.suptitle(S.plot_title + ' DOM 4a')
    print('DOM 4a', end=':  ')
    plq(plt, S.mr, 'time', S.mr, 'ib', color='orange', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_f', color='orange', linestyle='-', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'ib', color='green', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_f', color='green', linestyle='--', warn=S.run_type=='HistHist')
    plq(plt, S.mr, 'time', S.mr, 'ib_charge', color='red', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_charge_f', color='red', linestyle='-.', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'ib_charge', color='black', linestyle=':', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_charge_f', color='black', linestyle=':', warn=S.run_type=='HistHist')
    plt.legend(loc=1)
    plt.subplot(312)
    plq(plt, S.sr, 'time', S.sr, 'soc_s', color='orange', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.smv, 'time', S.smv, 'soc_s', color='green', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_run)
    plq(plt, S.mr, 'time', S.mr, 'soc', color='red', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'soc', color='black', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(313)
    plq(plt, S.mr, 'time', S.mr, 'Tb', color='red', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'Tb_f', color='red', linestyle='-', warn=S.run_type=='HistHist')
    plq(plt, S.mv, 'time', S.mv, 'Tb', color='cyan', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'Tb_f', color='cyan', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_run)
    plt.legend(loc=1)
    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


def ekf_plots(S:PlotOptions, fig_list=None, fig_files=None):
    print('ekf_plot', end=':  ')
    if S.sr and S.smv:
        if S.mr.Fx is not None:  # ekf
            fig_list.append(plt.figure())  # EKF  1
            plt.subplot(331)
            plt.suptitle(S.plot_title + ' EKF 1')
            print('EKF 1', end=':  ')
            plq(plt, S.mr, 'time_e', S.mr, 'u', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'u_ekf', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(332)
            plq(plt, S.mr, 'time_e', S.mr, 'z', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'z', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(333)
            plq(plt, S.smr, 'time', S.smr, 'reset_s', color='green', linestyle='-', stairs=True, warn=S.run_type=='RunRun')
            plq(plt, S.smv, 'time', S.smv, 'reset_s', color='green', linestyle='-', stairs=True)
            plq(plt, S.sr, 'time', S.sr, 'sat_s', color='blue', linestyle='--', stairs=True)
            plq(plt, S.sv, 'time', S.sv, 'sat_s', color='red', linestyle='-.', stairs=True, warn=S.run_type=='RunRun')
            plq(plt, S.smv, 'time', S.smv, 'sat_s', color='red', linestyle='-.', stairs=True, warn=S.run_type=='RunRun')
            plq(plt, S.mv, 'time', S.mv, 'reset_ekf', color='orange', linestyle=':', stairs=True, warn=S.run_type!='RunRun')
            plt.legend(loc=1)
            plt.subplot(334)
            plq(plt, S.mr, 'time_e', S.mr, 'Fx', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'Fx', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(335)
            plq(plt, S.mr, 'time_e', S.mr, 'Bu', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'Bu', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(336)
            plq(plt, S.mr, 'time_e', S.mr, 'Q', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'Q', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(337)
            plq(plt, S.mr, 'time_e', S.mr, 'R', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'R', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(338)
            plq(plt, S.mr, 'time_e', S.mr, 'P', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'P', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(339)
            plq(plt, S.mr, 'time_e', S.mr, 'S', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'S', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            if S.save_plots and not S.terse:
                plt.savefig(fig_file_name, format="png")

            fig_list.append(plt.figure())  # EKF  2
            plt.subplot(331)
            plt.suptitle(S.plot_title + ' EKF 2')
            print('EKF 2', end=':  ')
            plq(plt, S.mr, 'time_e', S.mr, 'K', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'K', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(332)
            plq(plt, S.mr, 'time_e', S.mr, 'x', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='red', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='cyan', linestyle='-.', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='orange', linestyle=':', stairs=True)
            plt.legend(loc=1)
            plt.subplot(333)
            plq(plt, S.mr, 'time', S.mr, 'y_ekf', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'y_ekf', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(334)
            plq(plt, S.mr, 'time_e', S.mr, 'x_prior', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'x_prior', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(335)
            plq(plt, S.mr, 'time_e', S.mr, 'P_prior', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'P_prior', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(336)
            plq(plt, S.mr, 'time_e', S.mr, 'x_post', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'x_post', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(337)
            plq(plt, S.mr, 'time_e', S.mr, 'P_post', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'P_post', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(338)
            plq(plt, S.mv, 'time', S.mv, 'voc_stat_ekf', color='magenta', linestyle='-.', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'z', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'z', color='red', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'hx', color='cyan', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'hx', color='orange', linestyle='--')
            plt.legend(loc=1)
            plt.subplot(339)
            plq(plt, S.mr, 'time_e', S.mr, 'H', color='blue', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'H', color='red', linestyle='--', stairs=True)
            plt.legend(loc=1)
            fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            if S.save_plots and not S.terse:
                plt.savefig(fig_file_name, format="png")

            fig_list.append(plt.figure())  # EKF2a
            plt.subplot(311)
            plt.suptitle(S.plot_title + ' EKF 2a')
            print('EKF 2a', end=':  ')
            plq(plt, S.mr, 'time', S.mr, 'voc_stat', add=-0.0, color='red', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'voc_stat', add=-0.0, color='black', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'z', color='cyan', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'z', color='orange', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'hx', color='magenta', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'hx', color='green', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(312)
            plq(plt, S.mr, 'x', S.mr, 'hx', color='red', linestyle='-', stairs=True)
            plq(plt, S.mv, 'x_ekf', S.mv, 'hx', color='black', linestyle='--', stairs=True)
            plt.legend(loc=1)
            plt.subplot(313)
            plq(plt, S.mr, 'time', S.mr, 'dt', color='red', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'dt', color='black', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'dt_ekf', color='blue', linestyle='-.', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'dt_eframe', color='orange', linestyle=':', stairs=True)
            plt.legend(loc=1)
            fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            if S.save_plots and not S.terse:
                plt.savefig(fig_file_name, format="png")

            fig_list.append(plt.figure())  # EKF3
            plt.subplot(221)
            plt.suptitle(S.plot_title + ' EKF 3')
            print('EKF 3', end=':  ')
            plq(plt, S.mr, 'time', S.mr, 'ib', color='red', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'ib', color='black', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'u', color='cyan', linestyle='-.', stairs=True)
            if S.run_type == 'RunRun':
                plq(plt, S.mv, 'time_e', S.mv, 'u', color='orange', linestyle=':', stairs=True)
            else:
                plq(plt, S.mv, 'time', S.mv, 'u_ekf', color='orange', linestyle=':', stairs=True)
            plt.legend(loc=1)
            plt.subplot(222)
            plq(plt, S.mr, 'time', S.mr, 'vb', color='red', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'vb', color='black', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'z', color='cyan', linestyle='-.', stairs=True)
            if S.run_type == 'RunRun':
                plq(plt, S.mv, 'time_e', S.mv, 'z', color='orange', linestyle=':', stairs=True)
            else:
                plq(plt, S.mv, 'time', S.mv, 'z', color='orange', linestyle=':', stairs=True)
            plt.legend(loc=1)
            plt.subplot(223)
            plq(plt, S.mr, 'time', S.mr, 'soc', color='red', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'soc', color='black', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time', S.mr, 'soc_ekf', color='cyan', linestyle='-.', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'soc_ekf', color='orange', linestyle=':', stairs=True)
            plt.legend(loc=1)
            plt.subplot(224)
            plq(plt, S.mr, 'time', S.mr, 'voc_ekf', color='red', linestyle='-', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'voc_ekf', color='black', linestyle='--', stairs=True)
            plq(plt, S.mr, 'time_e', S.mr, 'z', color='cyan', linestyle='-.', stairs=True)
            plq(plt, S.mv, 'time', S.mv, 'z', color='orange', linestyle=':', stairs=True)
            plt.legend(loc=1)
            fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            if S.save_plots and not S.terse:
                plt.savefig(fig_file_name, format="png")

    if S.mr.voc_soc is not None:
        fig_list.append(plt.figure())  # EKF  4
        plt.subplot(111)
        plt.suptitle(S.plot_title + ' EKF 4')
        print('EKF 4', end=':  ')
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='red', linestyle='-', warn=not S.run_is_stdy)
        plq(plt, S.mr, 'soc', S.mr, 'voc_stat_f', color='red', linestyle='-', warn=not S.run_is_run)
        plq(plt, S.mv, 'soc', S.mv, 'voc_stat', color='blue', linestyle='--', warn=not S.ver_is_stdy)
        plq(plt, S.mv, 'soc', S.mv, 'voc_stat_f', color='blue', linestyle='--', warn=not S.ver_is_sim and not S.ver_is_run)
        plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='black', linestyle='-.')
        plq(plt, S.mv, 'soc', S.mv, 'voc_soc', color='green', linestyle=':')
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # Hyst 1
        plt.subplot(331)
        plt.suptitle(S.plot_title + ' Hyst 1')
        print('Hyst 1', end=':  ')
        # plq(plt, S.mr, 'time', S.mr, 'dv_hys_required, linestyle='-', color='black', label='dv_hys_required'+run_str)
        plq(plt, S.mr, 'time', S.mr, 'e_wrap_m', slr=-1, color='red', linestyle='-', warn=not S.run_is_stdy)
        plq(plt, S.mr, 'time', S.mr, 'e_wrap_m_filt', slr=-1, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'e_wrap_m', slr=-1, color='blue', linestyle='--', warn=not S.ver_is_stdy)
        plq(plt, S.mv, 'time', S.mv, 'e_wrap_m_filt', slr=-1, color='cyan', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='orange', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='magenta', linestyle='None', marker='.', markersize='1',
            markevery=48, warn=not S.ver_is_stdy)
        plq(plt, S.sr, 'time', S.sr, 'dv_hys_s', color='cyan', linestyle=':', warn=not S.run_is_stdy)
        plq(plt, S.sv, 'time', S.sv, 'dv_hys', color='black', linestyle='None', marker='.', markersize='1',
            markevery=64, warn=not S.ver_is_stdy and not S.ver_is_sim and not S.ver_is_run)
        plq(plt, S.smv, 'time', S.smv, 'dv_hys_s', color='black', linestyle='None', marker='.', markersize='1',
            markevery=64, warn=not S.ver_is_stdy and not S.ver_is_sim and not S.ver_is_run)
        plt.xlabel('sec')
        plt.legend(loc=4)

        plt.subplot(332)
        plq(plt, S.mr, 'time', S.mr, 'e_wrap_n', slr=-1, color='red', linestyle='-', warn=not S.run_is_stdy)
        plq(plt, S.mr, 'time', S.mr, 'e_wrap_n_filt', slr=-1, color='black', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'e_wrap_n', slr=-1, color='blue', linestyle='--', warn=not S.ver_is_stdy)
        plq(plt, S.mv, 'time', S.mv, 'e_wrap_n_filt', slr=-1, color='cyan', linestyle='--')
        plq(plt, S.mr, 'time', S.mr, 'dv_hys', color='orange', linestyle='-.', warn=not S.run_is_stdy)
        plq(plt, S.mv, 'time', S.mv, 'dv_hys', color='magenta', linestyle='None', marker='.', markersize='1',
            markevery=48)
        plq(plt, S.sr, 'time', S.sr, 'dv_hys_s', color='cyan', linestyle=':', warn=not S.run_is_stdy)
        plq(plt, S.sv, 'time', S.sv, 'dv_hys', color='black', linestyle='None', marker='.', markersize='1',
            markevery=64, warn=not S.ver_is_run and not S.ver_is_stdy)
        plq(plt, S.smv, 'time', S.smv, 'dv_hys_s', color='black', linestyle='None', marker='.', markersize='1',
            markevery=64, warn=not S.ver_is_stdy and not S.ver_is_sim and not S.ver_is_run)
        plt.xlabel('sec')
        plt.legend(loc=4)

        plt.subplot(333)
        plq(plt, S.mr, 'time', S.mr, 'soc', color='green', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'soc', color='red', linestyle='--')
        plq(plt, S.sr, 'time', S.sr, 'soc_s', color='blue', linestyle='-.', warn=not S.run_is_stdy)
        plq(plt, S.sv, 'time', S.sv, 'soc', color='cyan', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
        plq(plt, S.smv, 'time', S.smv, 'soc_s', color='cyan', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_run)
        plt.xlabel('sec')
        plt.legend(loc=4)
        if not S.strict_overplot:
            plt.subplot(334)
            plq(plt, S.mr, 'time', S.mr, 'ib_sel', color='black', linestyle='-')
            plq(plt, S.mv, 'soc', S.mv, 'ib_sel', color='cyan', linestyle='--', warn=(S.run_type!='HistHist')and(S.run_type!='RunSim') and(S.run_type!='HistSim'))
            plq(plt, S.mr, 'time', S.mr, 'ioc', color='cyan', linestyle='-.', warn=not S.run_is_stdy)
            plt.xlabel('sec')
            plt.legend(loc=4)
            plt.subplot(335)
            plq(plt, S.mr, 'soc', S.mr, 'ib_sel', color='black', linestyle='-')
            plq(plt, S.mv, 'soc', S.mv, 'ib_sel', color='cyan', linestyle='--', warn=(S.run_type!='HistHist')and(S.run_type!='RunSim') and(S.run_type!='HistSim'))
            plq(plt, S.mr, 'soc', S.mr, 'ioc', color='cyan', linestyle='-.', warn=not S.run_is_stdy)
            plt.xlabel('soc')
            plt.legend(loc=4)
            plt.subplot(336)
            plq(plt, S.sr, 'time', S.sr, 'vb_s', color='black', linestyle='-', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'time', S.mr, 'vb', color='orange', linestyle='--', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'time', S.mr, 'vb_f', color='orange', linestyle='--',
                warn=not S.run_is_stdy and not S.run_is_run)
            plq(plt, S.sr, 'time', S.sr, 'voc_stat_s', color='magenta', linestyle='-', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='pink', linestyle='--', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'time', S.mr, 'voc_stat_f', color='pink', linestyle='--', warn=S.run_type=='HistHist')
            plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='black', linestyle='None', marker='.', markersize='1', markevery=32)
            plt.legend(loc=1)
            plt.subplot(337)
            plq(plt, S.mr, 'time', S.mr, 'vb', color='green', linestyle='-', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'time', S.mr, 'vb_f', color='green', linestyle='-', warn=S.run_type=='HistHist')
            plq(plt, S.mr, 'time', S.mr, 'voc', color='red', linestyle='--', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='pink', linestyle='-.', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'time', S.mr, 'voc_stat_f', color='pink', linestyle='-.', warn=S.run_type=='HistHist')
            plq(plt, S.mr, 'time', S.mr, 'voc_soc', color='black', linestyle='None', marker='.', markersize='1', markevery=32)
            plt.xlabel('sec')
            plt.legend(loc=4)
            plt.subplot(338)
            plq(plt, S.mr, 'soc', S.mr, 'vb', color='green', linestyle='-', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'soc', S.mr, 'vb_f', color='green', linestyle='-', warn=S.run_type=='HistHist')
            plq(plt, S.mr, 'soc', S.mr, 'voc', color='red', linestyle='--', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'soc', S.mr, 'voc_f', color='red', linestyle='--', warn=S.run_type=='HistHist')
            plq(plt, S.mr, 'soc', S.mr, 'voc_stat', color='blue', linestyle='-.', warn=not S.run_is_stdy)
            plq(plt, S.mr, 'soc', S.mr, 'voc_stat_f', color='blue', linestyle='-.', warn=S.run_type=='HistHist')
            plq(plt, S.mr, 'soc', S.mr, 'voc_soc', color='black', linestyle=':')
            plt.xlabel('soc')
            plt.legend(loc=4)
        plt.subplot(339)
        plq(plt, S.mr, 'time', S.mr, 'vb', color='green', linestyle='-', warn=not S.run_is_stdy)
        plq(plt, S.mr, 'time', S.mr, 'vb_f', color='green', linestyle='-', warn=S.run_type=='HistHist')
        plq(plt, S.mv, 'time', S.mv, 'vb', color='orange', linestyle='--', warn=not S.ver_is_stdy)
        plq(plt, S.mv, 'time', S.mv, 'vb_f', color='orange', linestyle='--', warn=S.run_type=='HistHist')
        plq(plt, S.mr, 'time', S.mr, 'voc', color='blue', linestyle='-.', warn=not S.run_is_stdy)
        plq(plt, S.mr, 'time', S.mr, 'voc_f', color='blue', linestyle='-.', warn=S.run_type=='HistHist')
        plq(plt, S.mv, 'time', S.mv, 'voc', color='black', linestyle='None', marker='+', markersize='3',
            markevery=4, warn=not S.ver_is_stdy)
        plq(plt, S.mv, 'time', S.mv, 'voc_f', color='black', linestyle='None', marker='+', markersize='3',
            markevery=4, warn=S.run_type=='HistHist')
        plq(plt, S.mr, 'time', S.mr, 'voc_stat', color='magenta', linestyle='-', warn=not S.run_is_stdy)
        plq(plt, S.mr, 'time', S.mr, 'voc_stat_f', color='magenta', linestyle='-', warn=S.run_type=='HistHist')
        plq(plt, S.mv, 'time', S.mv, 'voc_stat', color='pink', linestyle='--', warn=not S.ver_is_stdy)
        plq(plt, S.mv, 'time', S.mv, 'voc_stat_f', color='pink', linestyle='--', warn=S.run_type=='HistHist')
        plt.legend(loc=1)
        fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if S.save_plots and not S.terse:
            plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
