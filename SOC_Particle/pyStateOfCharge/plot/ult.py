# Copyright (C) 2026 Dave Gutz
#
# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
# type: ignore
#
# pylint: disable=invalid-name, no-member, attribute-defined-outside-init
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

""" ult the ultimate general overview plots
Dependencies:
    - numpy      (everything)
    - plq       GP plotter function
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq
from plot.PlotOptions import PlotOptions


# noinspection PyPep8Naming
def ult_1(S:PlotOptions, fig_files=None, fig_list=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())  # Ult 1
    plt.subplot(331)
    plt.suptitle(S.plot_title + ' Ult 1')
    plt.rcParams['legend.fontsize'] = 6
    print('Ult 1', end=':  ')
    # noinspection PyRedundantParentheses
    if (hasattr(S.mr, 'mib') and all(S.mr.mib == 0)) or (hasattr(S.mr, 'mod_data') and all(S.mr.mod_data < 64)):
        plq(plt, S.mr, 'time', S.mr, 'ib_amp_hdwe', color='green', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_amp_hdwe', color='red', linestyle='--', warn=False)
        plq(plt, S.mr, 'time', S.mr, 'ib_noa_hdwe', color='blue', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'ib_noa_hdwe', color='orange', linestyle=':', warn=False)
    elif (hasattr(S.mr, 'mod') and all(S.mr.mod >= 255.)) or (not (S.strict_overplot)
                                                              and not(S.run_type=='HistSim' or S.run_type=='HistHist')):
        plq(plt, S.mr, 'time', S.mr, 'ib_amp_model', add=1., color='green', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'ib_amp_model', add=1., color='red', linestyle='--', warn=False)
        plq(plt, S.mr, 'time', S.mr, 'ib_noa_model', add=1., color='blue', linestyle='-.')
        plq(plt, S.mv, 'time', S.mv, 'ib_noa_model', add=1., color='orange', linestyle=':', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_diff', color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_diff', color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ib_diff_f', color='cyan', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ibd_thr', color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ibd_thr', slr=-1, color='red', linestyle='--', warn=not S.run_is_stdy)
    plt.legend(loc=1)
    plt.subplot(334)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap', color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap', color='magenta', linestyle='--', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_filt', color='blue', linestyle='-', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_filt', color='orange', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_n', color='green', linestyle='-.', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_n', color='pink', linestyle=':', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_n_filt', color='cyan', linestyle='--', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_n_filt', color='green', linestyle='-.')
    plq(plt, S.mr, 'time', S.mr, 'cc_dif', color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'cc_dif', color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ewnhi_thr', color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ewnhi_thr', color='orange', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ewnlo_thr', color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ewnlo_thr', color='orange', linestyle='-.', warn=not S.ver_is_stdy)
    # if active standby
    # plq(plt, S.mr, 'time', S.mr, 'ewhi_thr', color='red', linestyle='-.')
    # plq(plt, S.mr, 'time', S.mr, 'ewlo_thr', color='red', linestyle='-.')
    plt.ylim(-4, 4)
    plt.legend(loc=1)
    plt.subplot(332)
    plq(plt, S.mr, 'time', S.mr, 'tb_sel', add=+6, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'tb_sel', add=+6, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'vb_sel', add=+2, color='magenta', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'vb_sel', add=+2, color='cyan', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'Tb_flt', color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'Tb_flt', color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'Tb_fa', color='magenta', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'Tb_fa', color='cyan', linestyle=':', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_choice', add=-2, color='black', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_choice', add=-2, color='blue', linestyle='--', warn=not S.ver_is_stdy and not S.ver_is_sim)
    plq(plt, S.mr, 'time', S.mr, 'reset', add=-4, color='magenta', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'reset', add=-4, color='cyan', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(337)
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_n_filt', add=1, color='green', linestyle='-', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_n_filt', add=1, color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_n_trim', add=1, color='magenta', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_n_trim', add=1, color='cyan', linestyle=':', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ewnhi_thr', add=1, color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ewnhi_thr', add=1, color='orange', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ewnlo_thr', add=1, color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ewnlo_thr', add=1, color='orange', linestyle='-.', warn=not S.ver_is_stdy)

    plq(plt, S.mr, 'time', S.mr, 'e_wrap_m_filt', add=-1, color='green', linestyle='-', warn=False)
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_m_filt', add=-1, color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'e_wrap_m_trim', add=-1, color='magenta', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'e_wrap_m_trim', add=-1, color='cyan', linestyle=':', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ewmhi_thr', add=-1, color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ewmhi_thr', add=-1, color='orange', linestyle='-.', warn=not S.ver_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'ewmlo_thr', add=-1, color='red', linestyle='--', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ewmlo_thr', add=-1, color='orange', linestyle='-.', warn=not S.ver_is_stdy)
    plt.ylim(-4, 4)
    plt.legend(loc=1)
    plt.subplot(338)
    plq(plt, S.mr, 'time', S.mr, 'cc_dif', color='black', linestyle='-')
    plq(plt, S.mr, 'time', S.mr, 'ccd_thr', color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'ccd_thr', slr=-1, color='red', linestyle='--')
    plt.ylim(-.01, .01)
    plt.legend(loc=3)
    plt.subplot(133)
    plq(plt, S.mr, 'time', S.mr, 'wrap_lo_fa', add=+58, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_lo_fa', add=+58, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_lo_flt', add=+56, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_lo_flt', add=+56, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_hi_fa', add=+54, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_hi_fa', add=+54, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_hi_flt', add=+52, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_hi_flt', add=+52, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_lo_n_fa', add=+50, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_lo_n_fa', add=+50, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_lo_n_flt', add=+48, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_lo_n_flt', add=+48, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_hi_n_fa', add=+46, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_hi_n_fa', add=+46, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_hi_n_flt', add=+44, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_hi_n_flt', add=+44, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_lo_m_fa', add=+42, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_lo_m_fa', add=+42, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_lo_m_flt', add=+40, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_lo_m_flt', add=+40, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_hi_m_fa', add=+38, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_hi_m_fa', add=+38, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_hi_m_flt', add=+36, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_hi_m_flt', add=+36, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'vc_fa', add=+34, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'vc_fa', add=+34, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'vc_flt', add=+32, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'vc_flt', add=+32, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_bare_flt', add=+30, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_noa_bare_flt', add=+30, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_bare_flt', add=+28, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_amp_bare_flt', add=+28, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_dscn_fa', add=+26, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_dscn_fa', add=+26, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_dscn_flt', add=+24, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_dscn_flt', add=+24, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_diff_fa', add=+22, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_diff_fa', add=+22, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_diff_flt', add=+20, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_diff_flt', add=+20, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wv_fa', add=+18, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wv_fa', add=+18, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'red_loss', add=+16, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'red_loss', add=+16, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_lo_fa', add=+14, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_lo_flt', add=+14, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'wrap_hi_fa', add=+12, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'wrap_hi_flt', add=+12, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ccd_fa', add=+10, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ccd_fa', add=+10, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_noa_fa', add=+8, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_noa_fa', add=+8, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_amp_fa', add=+6, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'ib_amp_fa', add=+6, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'vb_fa_lt', add=+4, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'vb_fa_lt', add=+4, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'Tb_fa', add=+2, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'Tb_fa', add=+2, color='red', linestyle='--', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'ib_dec', color='blue', linestyle='-.', warn=not S.run_is_stdy and not S.run_is_run)
    plq(plt, S.mv, 'time', S.mv, 'ib_dec', color='orange', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_sim)
    plq(plt, S.mr, 'time', S.mr, 'time_long', add=-10, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'accy', add=-12, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'off', add=-14, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'SAT', add=-16, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'flt_ekf', add=-18, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'flt_tb', add=-20, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'fail_vb', add=-22, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'fail_ibm', add=-24, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'fail_ib', add=-26, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'red_loss', add=-28, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'diff_ib', add=-30, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mr, 'time', S.mr, 'conn', add=-32, color='green', linestyle='-', warn=not S.run_is_stdy)
    # enum  dispw {conn = 0, diff_ib = 1, red_loss = 2, fail_ib = 3, fail_ibm = 4, fail_vb = 5, flt_tb = 6, flt_ekf = 7, SAT = 8, off = 9, accy = 10, time_long = 11, Count};
    plt.legend(loc=1)
    plt.subplot(335)
    plq(plt, S.mr, 'time', S.mr, 'bms_off', add=+4, color='green', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'bms_off', add=+4, color='red', linestyle='--')
    if S.sr is not None:
        plq(plt, S.sr, 'time', S.sr, 'bms_off_s', add=+4, color='blue', linestyle='-.')
    if S.run_type != 'HistSim' and hasattr(S.mr, 'mod_data') and hasattr(S.mv, 'mod_data'):
        mod_min = min(min(S.mr.mod_data), min(S.mv.mod_data))
        plq(plt, S.mr, 'time', S.mr, 'mod_data', add=-mod_min, color='green', linestyle='-')
        plq(plt, S.mv, 'time', S.mv, 'mod_data', add=-mod_min, color='red', linestyle='--')
    if S.smv is not None:
        if hasattr(S.smv, 'bmso_s'):
            plq(plt, S.smv, 'time', S.smv, 'bmso_s', add=+4, color='orange', linestyle=':')
        elif hasattr(S.smv, 'bms_off_s'):
            plq(plt, S.smv, 'time', S.smv, 'bms_off_s', add=+4, color='orange', linestyle=':')
    plq(plt, S.mr, 'time', S.mr, 'sat', add=+2, color='blue', linestyle='-')
    plq(plt, S.mv, 'time', S.mv, 'sat', add=+2, color='red', linestyle='--')
    plq(plt, S.mr, 'time', S.mr, 'saturated', add=+2, color='black', linestyle='-.')
    plq(plt, S.mv, 'time', S.mv, 'saturated', add=+2, color='orange', linestyle=':')
    plq(plt, S.mr, 'time', S.mr, 'sel', color='black', linestyle='-.')
    plq(plt, S.mr, 'time', S.mr, 'ib_choice', add=-2, color='green', linestyle='-', warn=not S.run_is_stdy)
    plq(plt, S.mv, 'time', S.mv, 'ib_choice', add=-2, color='red', linestyle=':', warn=not S.ver_is_stdy and not S.ver_is_sim)
    plq(plt, S.mr, 'time', S.mr, 'vb_sel', add=-2, color='black', linestyle='--')
    plq(plt, S.mv, 'time', S.mv, 'vb_sel', add=-2, color='orange', linestyle='-.', warn=False)
    plq(plt, S.mr, 'time', S.mr, 'preserving', add=-2, color='blue', linestyle='-.')
    plt.legend(loc=1)
    plt.rcParams['legend.fontsize'] = 'small'

    fig_file_name = S.filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if S.save_plots and not S.terse:

        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
