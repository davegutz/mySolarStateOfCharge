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

""" ult the ultimate general overview plots
Dependencies:
    - numpy      (everything)
    - plq       GP plotter function
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq


def ult_1(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())  # Ult 1
    plt.subplot(331)
    plt.title(plot_title + ' Ult 1')
    plt.rcParams['legend.fontsize'] = 6
    print('Ult 1', end=':  ')
    plq(plt, mr, 'time', mr, 'ib_amp_hdwe', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_amp_hdwe', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_noa_hdwe', color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ib_noa_hdwe', color='orange', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'ib_amp_model', add=1., color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_amp_model', add=1., color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_noa_model', add=1., color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ib_noa_model', add=1., color='orange', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'ib_diff_f', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_diff_f', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ibd_thr', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'ibd_thr', slr=-1, color='red', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(334)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'e_wrap', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'e_wrap_filt', color='black', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_filt', color='orange', linestyle='--')
    plq(plt, mr, 'time', mr, 'e_w_f', color='black', linestyle='-.', warn=False)
    plq(plt, mr, 'time', mr, 'e_wrap_n', color='green', linestyle='-.')
    plq(plt, mv, 'time', mv, 'e_wrap_n', color='pink', linestyle=':')
    plq(plt, mr, 'time', mr, 'e_wrap_n_filt', color='cyan', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'e_wn_f', color='cyan', linestyle='--', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_n_filt', color='green', linestyle='-.')
    plq(plt, mr, 'time', mr, 'cc_dif', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'cc_dif', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ewnhi_thr', color='red', linestyle='--')
    plq(plt, mv, 'time', mv, 'ewnhi_thr', color='orange', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ewnlo_thr', color='red', linestyle='--')
    plq(plt, mv, 'time', mv, 'ewnlo_thr', color='orange', linestyle='-.')
    # if active standby
    # plq(plt, mr, 'time', mr, 'ewhi_thr', color='red', linestyle='-.')
    # plq(plt, mr, 'time', mr, 'ewlo_thr', color='red', linestyle='-.')
    plt.ylim(-1, 1)
    plt.legend(loc=1)
    plt.subplot(332)
    plq(plt, mr, 'time', mr, 'tb_sel', add=+6, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'tb_sel', add=+6, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'vb_sel', add=+2, color='magenta', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb_sel', add=+2, color='cyan', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'tb_flt', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'tb_flt', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'tb_fa', color='magenta', linestyle='-.')
    plq(plt, mv, 'time', mv, 'tb_fa', color='cyan', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'ib_choice', add=-2, color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_choice', add=-2, color='blue', linestyle='--', warn=False)
    plt.legend(loc=1)
    plt.subplot(337)
    plq(plt, mr, 'time', mr, 'e_wrap_m_filt', color='green', linestyle='-', warn=False)
    plq(plt, mr, 'time', mr, 'e_wm_f', color='green', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_m_filt', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'e_wrap_m_trim', color='magenta', linestyle='-.')
    plq(plt, mr, 'time', mr, 'e_wm_t', color='magenta', linestyle='-.', warn=False)
    plq(plt, mv, 'time', mv, 'e_wrap_m_trim', color='cyan', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'ewmhi_thr', color='red', linestyle='--')
    plq(plt, mv, 'time', mv, 'ewmhi_thr', color='orange', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ewmlo_thr', color='red', linestyle='--')
    plq(plt, mv, 'time', mv, 'ewmlo_thr', color='orange', linestyle='-.')
    plt.ylim(-0.2, 0.2)
    plt.legend(loc=1)
    plt.subplot(338)
    plq(plt, mr, 'time', mr, 'cc_dif', color='black', linestyle='-')
    plq(plt, mr, 'time', mr, 'ccd_thr', color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'ccd_thr', slr=-1, color='red', linestyle='--')
    # plt.ylim(-.01, .01)
    plt.legend(loc=3)
    plt.subplot(133)
    plq(plt, mr, 'time', mr, 'wrap_hi_fa', add=+24, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_fa', add=+24, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_hi_flt', add=+22, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_flt', add=+22, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_fa', add=+20, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_fa', add=+20, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_flt', add=+18, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_flt', add=+18, color='red', linestyle='--', warn=False)

    plq(plt, mr, 'time', mr, 'wrap_hi_m_fa', add=+16, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_m_fa', add=+16, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_hi_m_flt', add=+14, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_m_flt', add=+14, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_m_fa', add=+12, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_m_fa', add=+12, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_m_flt', add=+10, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_m_flt', add=+10, color='red', linestyle='--', warn=False)

    plq(plt, mr, 'time', mr, 'wrap_hi_n_fa', add=+8, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_n_fa', add=+8, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_hi_n_flt', add=+6, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_hi_n_flt', add=+6, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_n_fa', add=+4, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_n_fa', add=+4, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wrap_lo_n_flt', add=+2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wrap_lo_n_flt', add=+2, color='red', linestyle='--', warn=False)

    plq(plt, mr, 'time', mr, 'red_loss', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'red_loss', color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'wv_fa', add=-2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'wv_fa', add=-2, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ccd_fa', add=-4, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ccd_fa', add=-4, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_diff_fa', add=-6, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_diff_fa', add=-6, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_diff_flt', add=-8, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_diff_flt', add=-8, color='red', linestyle='--', warn=False)
    plq(plt, mr, 'time', mr, 'ib_dec', color='blue', linestyle='-.')
    plq(plt, mv, 'time', mv, 'ib_dec', color='orange', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'time_long', add=-10, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'accy', add=-12, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'off', add=-14, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'SAT', add=-16, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'flt_ekf', add=-18, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'flt_tb', add=-20, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'fail_vb', add=-22, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'fail_ibm', add=-24, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'fail_ib', add=-26, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'red_loss', add=-28, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'diff_ib', add=-30, color='green', linestyle='-')
    plq(plt, mr, 'time', mr, 'conn', add=-32, color='green', linestyle='-')
    # enum  dispw {conn = 0, diff_ib = 1, red_loss = 2, fail_ib = 3, fail_ibm = 4, fail_vb = 5, flt_tb = 6, flt_ekf = 7, SAT = 8, off = 9, accy = 10, time_long = 11, Count};
    plt.legend(loc=1)
    plt.subplot(335)
    plq(plt, mr, 'time', mr, 'bms_off', add=+4, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'bms_off', add=+4, color='red', linestyle='--')
    if sr is not None:
        plq(plt, sr, 'time', sr, 'bms_off_s', add=+4, color='blue', linestyle='-.')
    if hasattr(mr, 'mod_data'):
        mod_min = min(min(mr.mod_data), min(mv.mod_data))
    else:
        mod_min = min(mv.mod_data)
    plq(plt, mr, 'time', mr, 'mod_data', add=-mod_min, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'mod_data', add=-mod_min, color='red', linestyle='--')
    if smv is not None:
        if hasattr(smv, 'bmso_s'):
            plq(plt, smv, 'time', smv, 'bmso_s', add=+4, color='orange', linestyle=':')
        elif hasattr(smv, 'bms_off_s'):
            plq(plt, smv, 'time', smv, 'bms_off_s', add=+4, color='orange', linestyle=':')
    plq(plt, mr, 'time', mr, 'sat', add=+2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'sat', add=+2, color='red', linestyle='--')
    plq(plt, mr, 'time', mr, 'sel', color='black', linestyle='-.')
    plq(plt, mv, 'time', mv, 'sel', color='blue', linestyle=':')
    plq(plt, mr, 'time', mr, 'ib_choice', add=-2, color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_choice', add=-2, color='red', linestyle=':', warn=False)
    plq(plt, mr, 'time', mr, 'vb_sel', add=-2, color='black', linestyle='--')
    plq(plt, mv, 'time', mv, 'vb_sel', add=-2, color='orange', linestyle='-.', warn=False)
    plq(plt, mr, 'time', mr, 'preserving', add=-2, color='blue', linestyle='-.')
    plt.legend(loc=1)
    plt.rcParams['legend.fontsize'] = 'small'

    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
