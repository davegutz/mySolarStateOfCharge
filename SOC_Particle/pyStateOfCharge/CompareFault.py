# MonSim:  Monitor and Simulator replication of Particle Photon Application
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

""" Slice and dice the history dumps."""

import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
from Battery import Battery, BatteryMonitor, is_sat
from Battery import calculate_capacity, Retained
from Colors import Colors
from plot.plq import plq as plq
from Chemistry_BMS import ib_lag
from filter.myFilters import LagExp

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#  For this battery Battleborn 100 Ah with 1.084 x capacity
IB_BAND = 1.  # Threshold to declare charging or discharging
TB_BAND = 25.  # Band around temperature to group data and correct


# Add ib_lag = ib lagged by time constant
def add_ib(data):
    if hasattr(data, 'ib_amp_hdwe_f'):
        data = rf.rec_append_fields(data, 'ib_amp_hdwe', np.array(data.ib_amp_hdwe_f, dtype=float))
    if hasattr(data, 'ib_noa_hdwe_f'):
        data = rf.rec_append_fields(data, 'ib_noa_hdwe', np.array(data.ib_noa_hdwe_f, dtype=float))
    return data


# Add ib_lag = ib lagged by time constant
# noinspection PyPep8Naming
def add_ib_lag(data, mon):
    lag_tau = ib_lag(mon.chemistry.mod_code)
    IbLag = LagExp(1., lag_tau, -100., 100.)
    n = len(data.time)
    if n < 2:
        return data
    if not hasattr(data, 'ib_lag'):
        data = rf.rec_append_fields(data, 'ib_lag', np.array(data.time, dtype=float))
        data.ib_lag = np.zeros(n)
    dt = data.time[1] - data.time[0]
    for i in range(n):
        if i > 0:
            dt = data.time[i] - data.time[i-1]
        # noinspection PyUnresolvedReferences
        data.ib_lag[i] = IbLag.calculate_tau(float(data.ib_f[i]), i == 0, dt, lag_tau)
    return data


# Add schedule lookups and do some rack and stack
# noinspection PyPep8Naming,PyTypeChecker
def add_stuff_f(d_ra, mon, ib_band=0.5, rated_batt_cap=100., Dw=0., time_sync=None, ap_ib_diff_slr=1.,
                ap_ib_quiet_slr=1.):
    voc_soc = []
    soc_min = []
    vsat = []
    time_sec = []
    cc_diff_thr = []
    cc_dif = []
    ewhi_thr = []
    ewlo_thr = []
    ib_diff_thr = []
    ib_quiet_thr = []
    ib_diff = []
    dt = []
    ib_charge_f = []
    dv_dyn_f = []
    ib_dyn = []
    Tb_model_f_fut = []
    bms_off_init = False
    bms_off = False
    rp = Retained()
    if time_sync is None:
        time_sync = d_ra.time_ux[0]
        n = len(d_ra.time_ux)
    for i in range(n):
        soc = d_ra.soc[i]
        voc_stat_f = d_ra.voc_stat_f[i]
        Tb_f = d_ra.Tb_f[i]
        ib_diff_ = d_ra.ib_amp_hdwe_f[i] - d_ra.ib_noa_hdwe_f[i]
        cc_dif_ = d_ra.soc[i] - d_ra.soc_ekf[i]
        ib_diff.append(ib_diff_)
        C_rate = d_ra.ib_f[i] / rated_batt_cap
        Tb_model_f_fut.append(d_ra.Tb_f[min(i+1,n-1)])
        voc_soc.append(mon.chemistry.lookup_voc(d_ra.soc[i], d_ra.Tb_f[i]) + Dw)
        BB = BatteryMonitor(OPT=None)
        cc_diff_thr_, ewhi_thr_, ewlo_thr_, ib_diff_thr_, ib_quiet_thr_ = \
            fault_thr_bb(Tb_f, soc, voc_soc[i], voc_stat_f, C_rate, BB, ap_ib_diff_slr=ap_ib_diff_slr,
                         ap_ib_quiet_slr=ap_ib_quiet_slr)
        ib_f_ = d_ra.ib_f[i]
        tb_f_ = d_ra.Tb_f[i]
        vb_f_ = d_ra.vb_f[i]
        voc_f_ = d_ra.voc_f[i]
        ib_dyn_ = d_ra.ib_f[i]
        reset = True  # Always initializing in history mode - times spread out
        # Battery management system model (uses past value bms_off and voc_stat)
        if not bms_off:
            voltage_low = voc_stat_f < mon.chemistry.vb_down
        else:
            voltage_low = voc_stat_f < mon.chemistry.vb_rising
        bms_charging = ib_f_ > Battery.IB_MIN_UP
        if reset and bms_off_init is not None:
            bms_off = bms_off_init
        else:
            bms_off = (tb_f_ <= mon.chemistry.low_t) or (voltage_low and not rp.tweak_test)  # KISS
        ib_charge_f_ = ib_f_
        if bms_off and not bms_charging:
            ib_charge_f_ = 0.
        cc_dif.append(cc_dif_)
        cc_diff_thr.append(cc_diff_thr_)
        ewhi_thr.append(ewhi_thr_)
        ewlo_thr.append(ewlo_thr_)
        ib_diff_thr.append(ib_diff_thr_)
        ib_quiet_thr.append(ib_quiet_thr_)
        soc_min.append((BB.chemistry.lut_min_soc.interp(d_ra.Tb_f[i])))
        ib_dyn.append(ib_dyn_)
        vsat.append(mon.chemistry.nom_vsat + (d_ra.Tb_f[i] - mon.chemistry.rated_temp) * mon.chemistry.dvoc_dt)
        time_sec.append(float(d_ra.time_ux[i] - time_sync))
        if i > 0:
            dt.append(float(d_ra.time_ux[i] - d_ra.time_ux[i - 1]))
        elif len(d_ra.time_ux) > 1:
            dt.append(float(d_ra.time_ux[1] - d_ra.time_ux[0]))
        else:
            pass
        dv_dyn_f_ = vb_f_ - voc_f_
        ib_charge_f.append(ib_charge_f_)
        dv_dyn_f.append(dv_dyn_f_)
    time_min = (d_ra.time_ux - time_sync)/60.
    time_day = (d_ra.time_ux - time_sync)/3600./24.
    d_mod = rf.rec_append_fields(d_ra, 'time_sec', np.array(time_sec, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'Tb_model_f_fut', np.array(time_sec, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'time', np.array(time_sec, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'time_min', np.array(time_min, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'time_day', np.array(time_day, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'voc_soc', np.array(voc_soc, dtype=float))
    if not hasattr(d_mod, 'soc_min'):
        d_mod = rf.rec_append_fields(d_mod, 'soc_min', np.array(soc_min, dtype=float))
    Tb_model_f_rate = np.zeros(len(d_mod.time_ux))
    d_mod = rf.rec_append_fields(d_mod, 'Tb_model_f_rate', Tb_model_f_rate)
    Tb_model_f_rate_fut = np.zeros(len(d_mod.time_ux))
    d_mod = rf.rec_append_fields(d_mod, 'Tb_model_f_rate_fut', Tb_model_f_rate_fut)
    d_mod = rf.rec_append_fields(d_mod, 'vsat', np.array(vsat, dtype=float))
    # Tb_model_f_fut = d_mod.Tb_h_f.copy()
    # d_mod = rf.rec_append_fields(d_mod, 'Tb_model_f_fut', np.array(Tb_model_f_fut, dtype=float))
    # d_mod = rf.rec_append_fields(d_mod, 'Tb_f', np.array(Tb_f, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_diff', np.array(ib_diff, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_dyn', np.array(ib_dyn, dtype=float))
    # d_mod = rf.rec_append_fields(d_mod, 'time_sec', np.array(time_sec, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'cc_diff_thr', np.array(cc_diff_thr, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'cc_dif', np.array(cc_dif, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ewhi_thr', np.array(ewhi_thr, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ewlo_thr', np.array(ewlo_thr, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_diff_thr', np.array(ib_diff_thr, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_quiet_thr', np.array(ib_quiet_thr, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'dt', np.array(dt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_charge_f', np.array(ib_charge_f, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'dv_dyn_f', np.array(dv_dyn_f, dtype=float))
    d_mod = add_ib_lag(d_mod, mon)
    d_mod = add_ib(d_mod)
    d_mod = calc_fault(d_ra, d_mod)
    voc_stat_chg = np.copy(d_mod.voc_stat_f)
    voc_stat_dis = np.copy(d_mod.voc_stat_f)
    for i in range(len(voc_stat_chg)):
        if d_mod.ib_f[i] > -ib_band:
            voc_stat_dis[i] = None
        elif d_mod.ib_f[i] < ib_band:
            voc_stat_chg[i] = None
    d_mod = rf.rec_append_fields(d_mod, 'voc_stat_chg', np.array(voc_stat_chg, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'voc_stat_dis', np.array(voc_stat_dis, dtype=float))
    dv_hys = d_mod.voc_f - d_mod.voc_stat_f
    d_mod = rf.rec_append_fields(d_mod, 'dv_hys', np.array(dv_hys, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'dV_hys', np.array(dv_hys, dtype=float))
    dv_hys_required = d_mod.voc_f - voc_soc + dv_hys
    d_mod = rf.rec_append_fields(d_mod, 'dv_hys_required', np.array(dv_hys_required, dtype=float))

    voc_dyn = d_mod.voc_f.copy()
    d_mod = rf.rec_append_fields(d_mod, 'voc_dyn', np.array(voc_dyn, dtype=float))
    ib_f = d_mod.ib_f.copy()
    d_mod = rf.rec_append_fields(d_mod, 'ib_sel', np.array(ib_f, dtype=float))
    d_zero = d_mod.ib_f.copy()*0.
    d_mod = rf.rec_append_fields(d_mod, 'tweak_sclr_amp', np.array(d_zero, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'tweak_sclr_noa', np.array(d_zero, dtype=float))

    time_e = d_mod.time.copy()
    d_mod = rf.rec_append_fields(d_mod, 'time_e', np.array(time_e, dtype=float))
    dt_ekf = d_mod.time_ux.copy()*0.
    for i in range(len(time_e)-1, -1, -1):
        # print(i)
        if i > 0:
            dt_ekf[i] = time_e[i] - time_e[i-1]
        else:
            dt_ekf[i] = dt_ekf[i+1]
    d_mod = rf.rec_append_fields(d_mod, 'dt_ekf', np.array(dt_ekf, dtype=float))
    P = d_mod.time_ux.copy()*0.
    d_mod = rf.rec_append_fields(d_mod, 'P', np.array(P, dtype=float))
    z = d_mod.voc_stat_f.copy()
    d_mod = rf.rec_append_fields(d_mod, 'z', np.array(z, dtype=float))
    x = d_mod.soc_ekf.copy()
    d_mod = rf.rec_append_fields(d_mod, 'x', np.array(x, dtype=float))
    x_prior = d_mod.soc_ekf.copy()
    d_mod = rf.rec_append_fields(d_mod, 'x_prior', np.array(x_prior, dtype=float))
    x_post = d_mod.soc_ekf.copy()
    d_mod = rf.rec_append_fields(d_mod, 'x_post', np.array(x_post, dtype=float))
    voc_stat_f_lstate = d_mod.voc_stat_f.copy()
    d_mod = rf.rec_append_fields(d_mod, 'voc_stat_f_lstate', np.array(voc_stat_f_lstate, dtype=float))

    return d_mod


# Calculate thresholds from global input values listed above (review these)
# noinspection PyPep8Naming
def fault_thr_bb(Tb, soc, voc_soc, voc_stat, C_rate, bb, ap_ib_diff_slr=1., ap_ib_quiet_slr=1.):
    # There is no fault logic in the Python code, so hard code it here
    WRAP_HI_A = 32.  # Wrap high voltage threshold, A (32 after testing; 16=0.2v)
    WRAP_LO_A = -32.  # Wrap high voltage threshold, A (-32, -20 too small on truck -16=-0.2v)
    WRAP_HI_SETAT_MARG = 0.2  # Wrap voltage margin to saturation, V (0.2)
    WRAP_HI_SETAT_SCLR = 2.0  # Wrap voltage margin scalar when saturated (2.0)
    IBATT_DISAGREE_THRESH = 10.  # Signal selection threshold for current disagree test, A (10.)
    QUIET_A = 0.005  # Quiet set threshold, sec (0.005, 0.01 too large in truck)
    WRAP_SOC_HI_OFF = 0.97  # Disable e_wrap_hi when saturated
    WRAP_SOC_HI_SCLR = 1000.  # Huge to disable e_wrap
    WRAP_SOC_LO_OFF_ABS = 0.35  # Disable e_wrap when near empty (soc lo any Tb)
    WRAP_SOC_LO_OFF_REL = 0.2  # Disable e_wrap when near empty (soc lo for high Tb where soc_min=.2, voltage cutback)
    WRAP_SOC_LO_SCLR = 60.  # Large to disable e_wrap (60. for startup)
    CC_DIFF_SOC_DIS_THRESH = 0.2  # Signal selection threshold for Coulomb counter EKF disagree test (0.2)
    CC_DIFF_LO_SOC_SCLR = 4.  # Large to disable cc_ctf
    WRAP_MOD_C_RATE = .05  # Moderate charge rate threshold to engage wrap threshold 0
    WRAP_SOC_MOD_OFF = 0.85  # Disable e_wrap_lo when nearing saturated and moderate C_rate (0.85)

    vsat_ = bb.chemistry.nom_vsat + (Tb-25.)*bb.chemistry.dvoc_dt

    # cc_diff
    soc_min = bb.chemistry.lut_min_soc.interp(Tb)
    if soc <= max(soc_min+WRAP_SOC_LO_OFF_REL, WRAP_SOC_LO_OFF_ABS):
        cc_diff_empty_sclr_ = CC_DIFF_LO_SOC_SCLR
    else:
        cc_diff_empty_sclr_ = 1.
    cc_diff_sclr_ = 1.  # ram adjusts during data collection
    cc_diff_thr = CC_DIFF_SOC_DIS_THRESH * cc_diff_sclr_ * cc_diff_empty_sclr_

    # wrap
    if soc >= WRAP_SOC_HI_OFF:
        ewsat_sclr_ = WRAP_SOC_HI_SCLR
        ewmin_sclr_ = 1.
    elif soc <= max(soc_min+WRAP_SOC_LO_OFF_REL, WRAP_SOC_LO_OFF_ABS):
        ewsat_sclr_ = 1.
        ewmin_sclr_ = WRAP_SOC_LO_SCLR
    elif voc_soc > (vsat_ - WRAP_HI_SETAT_MARG) or \
            (voc_stat > (vsat_ - WRAP_HI_SETAT_MARG) and C_rate > WRAP_MOD_C_RATE and soc > WRAP_SOC_MOD_OFF):
        ewsat_sclr_ = WRAP_HI_SETAT_SCLR
        ewmin_sclr_ = 1.
    else:
        ewsat_sclr_ = 1.
        ewmin_sclr_ = 1.
    ewhi_sclr_ = 1.  # ram adjusts during data collection
    ewhi_thr = bb.chemistry.r_ss * WRAP_HI_A * ewhi_sclr_ * ewsat_sclr_ * ewmin_sclr_
    ewlo_sclr_ = 1.  # ram adjusts during data collection
    ewlo_thr = bb.chemistry.r_ss * WRAP_LO_A * ewlo_sclr_ * ewsat_sclr_ * ewmin_sclr_

    # ib_diff
    ib_diff_thr = IBATT_DISAGREE_THRESH * ap_ib_diff_slr

    # ib_quiet
    ib_quiet_thr = QUIET_A * ap_ib_quiet_slr

    return cc_diff_thr, ewhi_thr, ewlo_thr, ib_diff_thr, ib_quiet_thr


def over_fault(hi, filename, fig_files=None, plot_title=None, fig_list=None, subtitle=None, long_term=True,
               cc_dif_tol=0.2, time_units='days', timestr='time_ux', save_plots=False):

    print('over_fault', end=':  ')
    if fig_files is None:
        fig_files = []

    if long_term:
        fig_list.append(plt.figure())  # 1
        plt.subplot(331)
        plt.suptitle(plot_title + ' f1')
        print('f1', end=':  ')
        plt.suptitle(subtitle)
        plq(plt, hi, timestr, hi, 'soc', color='black', linestyle='-', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'soc_ekf', color='blue', linestyle='--', marker='+', markersize='3')
        plt.legend(loc=1)
        plt.subplot(332)
        plq(plt, hi, timestr, hi, 'Tb_f', color='black', linestyle='-', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'Tb', color='black', linestyle='-', marker='.', markersize='3', warn=False)
        plt.legend(loc=1)
        plt.subplot(333)
        plq(plt, hi, timestr, hi, 'ib_f', color='green', linestyle='-', marker='+', markersize='3')
        plq(plt, hi, timestr, hi, 'ib', color='black', linestyle='-', marker='.', markersize='3', warn=False)
        plt.legend(loc=1)
        plt.subplot(334)
        plq(plt, hi, timestr, hi, 'tweak_sclr_amp', color='orange', linestyle='None', marker='+', markersize='3')
        plq(plt, hi, timestr, hi, 'tweak_sclr_noa', color='green', linestyle='None', marker='^', markersize='3')
        plt.ylim(-6, 6)
        plt.legend(loc=1)
        plt.subplot(335)
        plq(plt, hi, timestr, hi, 'falw', color='magenta', linestyle='None', marker='+', markersize='3')
        plt.legend(loc=0)
        plt.subplot(336)
        plq(plt, hi, 'soc', hi, 'soc_ekf', color='blue', linestyle='-', marker='+', markersize='3')

        class Temp:
            def __init__(self):
                self.str = ''
                self.zero_one = [0, 1]
                self.ptol = [cc_dif_tol, 1. + cc_dif_tol]
                self.ntol = [-cc_dif_tol, 1. - cc_dif_tol]

        tmp = Temp()
        plq(plt, tmp, 'zero_one', tmp, 'ptol', color='red', linestyle=':')
        plq(plt, tmp, 'zero_one', tmp, 'ntol', color='red', linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('soc')
        plt.legend(loc=4)
        plt.subplot(337)
        plq(plt, hi, timestr, hi, 'dv_hys', color='blue', linestyle='-', marker='o', markersize='3')
        plq(plt, hi, timestr, hi, 'dv_hys_required', color='black', linestyle='--')
        plq(plt, hi, timestr, hi, 'e_wrap_filt', slr=-1, color='red', linestyle='None', marker='o', markersize='3',
            warn=False)
        plq(plt, hi, timestr, hi, 'e_wrap_filt', slr=-1, color='red', linestyle='None', marker='o', markersize='3')
        plt.xlabel(time_units)
        plt.legend(loc=1)
        plt.subplot(338)
        plq(plt, hi, timestr, hi, 'e_wrap_filt', color='black', linestyle='-', marker='o', markersize='3',
            warn=False)
        plq(plt, hi, timestr, hi, 'wv_fa', color='red', linestyle=':', marker=0, markersize='4')
        plq(plt, hi, timestr, hi, 'wrap_lo_fa', add=-1, color='orange', linestyle=':', marker=2, markersize='4')
        plq(plt, hi, timestr, hi, 'wrap_hi_fa', add=+1, color='green', linestyle=':', marker=3, markersize='4')
        plq(plt, hi, timestr, hi, 'wrap_lo_m_flt', add=-1, color='orange', linestyle=':', marker=2, markersize='4')
        plq(plt, hi, timestr, hi, 'wrap_hi_m_flt', add=+1, color='green', linestyle=':', marker=3, markersize='4')
        plq(plt, hi, timestr, hi, 'wrap_lo_n_flt', add=-1, color='orange', linestyle=':', marker=2, markersize='4')
        plq(plt, hi, timestr, hi, 'wrap_hi_n_flt', add=+1, color='green', linestyle=':', marker=3, markersize='4')
        plt.xlabel(time_units)
        plt.legend(loc=1)
        plt.subplot(339)
        plq(plt, hi, timestr, hi, 'vb_f', color='red', linestyle='None', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'vb', color='orange', linestyle='None', marker='.', markersize='3', warn=False)
        plq(plt, hi, timestr, hi, 'voc_f', color='black', linestyle='None', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'voc', color='blue', linestyle='None', marker='.', markersize='3', warn=False)
        plq(plt, hi, timestr, hi, 'voc_stat_chg', color='green', linestyle='None', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'voc_stat_dis', color='red', linestyle='None', marker='.', markersize='3')
        plt.xlabel(time_units)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if save_plots:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # 2
        plt.subplot(221)
        plt.suptitle(plot_title + ' f2')
        print('f2', end=':  ')
        plt.suptitle(subtitle)
        plq(plt, hi, timestr, hi, 'vsat', color='orange', linewidth='1', marker='.', markersize='1')
        plq(plt, hi, timestr, hi, 'vb_f', color='red', linestyle='None', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'vb', color='orange', linestyle='None', marker='.', markersize='3', warn=False)
        plq(plt, hi, timestr, hi, 'voc_f', color='black', linestyle='None', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'voc', color='blue', linestyle='None', marker='.', markersize='3', warn=False)
        plq(plt, hi, timestr, hi, 'voc_stat_chg', color='green', linestyle='-', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'voc_stat_dis', color='red', linestyle='-', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'voc_soc', color='cyan', linestyle=':', marker='2', markersize='3')
        plt.xlabel(time_units)
        plt.legend(loc=1)
        plt.subplot(122)
        plq(plt, hi, timestr, hi, 'bms_off', add=22, color='blue', linestyle='-', marker='h', markersize='3')
        plq(plt, hi, timestr, hi, 'saturated', add=20, color='red', linestyle='-', marker='s', markersize='3')
        plq(plt, hi, timestr, hi, 'ib_dscn_fa', add=18, color='black', linestyle='-', marker='o', markersize='3')
        plq(plt, hi, timestr, hi, 'ib_diff_fa', add=16, color='blue', linestyle='-', marker='^', markersize='3')
        plq(plt, hi, timestr, hi, 'wv_fa', add=14, color='cyan', linestyle='-', marker='s', markersize='3')
        plq(plt, hi, timestr, hi, 'wrap_lo_fa', add=12, color='blue', linestyle='-', marker='p', markersize='3')
        plq(plt, hi, timestr, hi, 'wrap_lo_m_fa', add=12, color='red', linestyle='--', marker='p', markersize='3')
        plq(plt, hi, timestr, hi, 'wrap_lo_n_fa', add=12, color='orange', linestyle='-.', marker='p', markersize='3')
        plq(plt, hi, timestr, hi, 'wrap_hi_fa', add=10, color='blue', linestyle='-', marker='h', markersize='3')
        plq(plt, hi, timestr, hi, 'wrap_hi_m_fa', add=10, color='red', linestyle='--', marker='h', markersize='3')
        plq(plt, hi, timestr, hi, 'wrap_hi_n_fa', add=10, color='orange', linestyle='-.', marker='h', markersize='3')
        plq(plt, hi, timestr, hi, 'ccd_fa', add=8, color='blue', linestyle='-', marker='H', markersize='3')
        plq(plt, hi, timestr, hi, 'ib_noa_fa', add=6, color='red', linestyle='-', marker='+', markersize='3')
        plq(plt, hi, timestr, hi, 'ib_amp_fa', add=4, color='magenta', linestyle='-', marker='_', markersize='3')
        plq(plt, hi, timestr, hi, 'vb_fa_lt', add=2, color='cyan', linestyle='-', marker='1', markersize='3')
        plq(plt, hi, timestr, hi, 'Tb_fa', color='orange', linestyle='-', marker='2', markersize='3')
        plq(plt, hi, timestr, hi, 'Tb_fa', color='orange', linestyle='-', marker='2', markersize='3')
        plt.ylim(-1, 24)
        plt.xlabel(time_units)
        plt.legend(loc=1)
        plt.subplot(223)
        plq(plt, hi, timestr, hi, 'ib_f', color='red', linestyle='-', marker='.', markersize='3')
        plq(plt, hi, timestr, hi, 'ib', color='orange', linestyle='-', marker='.', markersize='3', warn=False)
        plt.xlabel(time_units)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if save_plots:
            plt.savefig(fig_file_name, format="png")

        fig_list.append(plt.figure())  # 3
        plt.subplot(221)
        plt.suptitle(plot_title + ' f3')
        print('f3', end=':  ')
        plt.suptitle(subtitle)
        plq(plt, hi, timestr, hi, 'dv_hys', color='blue', linestyle='-', marker='o', markersize='3')
        plq(plt, hi, timestr, hi, 'dv_hys_required', color='black', linestyle='--')
        plq(plt, hi, timestr, hi, 'e_wrap_filt', slr=-1, color='red', linestyle='None', marker='o', markersize='3',
            warn=False)
        plt.xlabel(time_units)
        plt.legend(loc=4)
        # plt.ylim(-0.7, 0.7)
        plt.ylim(bottom=-0.7)
        plt.subplot(223)
        plq(plt, hi, timestr, hi, 'ib_f', color='black')
        plq(plt, hi, timestr, hi, 'ib', color='blue', warn=False)
        plq(plt, hi, timestr, hi, 'soc', slr=10, color='green')
        plt.xlabel(time_units)
        plt.legend(loc=4)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        if save_plots:
            plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # 4
    plt.subplot(331)
    plt.suptitle(plot_title + ' f4')
    print('f4', end=':  ')
    plq(plt, hi, timestr, hi, 'ib_f', color='green')
    plq(plt, hi, timestr, hi, 'ib', color='cyan', warn=False)
    plq(plt, hi, timestr, hi, 'ib_diff', color='black', linestyle='-.')
    plq(plt, hi, timestr, hi, 'ib_diff_thr', color='red', linestyle='-.')
    plq(plt, hi, timestr, hi, 'ib_diff_thr', slr=-1, color='red', linestyle='-.')
    plt.legend(loc=1)
    plt.subplot(332)
    plq(plt, hi, timestr, hi, 'sat', add=2, color='cyan', linestyle='-')
    plq(plt, hi, timestr, hi, 'saturated', add=2, color='magenta', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(333)
    plq(plt, hi, timestr, hi, 'vb_f', color='green', linestyle='-')
    plq(plt, hi, timestr, hi, 'vb', color='cyan', linestyle='-', warn=False)
    plt.legend(loc=1)
    plt.subplot(334)
    plq(plt, hi, timestr, hi, 'voc_stat_f', color='green', linestyle='-')
    plq(plt, hi, timestr, hi, 'voc_stat', color='cyan', linestyle='-', warn=False)
    plq(plt, hi, timestr, hi, 'vsat', color='blue', linestyle='-')
    plq(plt, hi, timestr, hi, 'voc_soc', add=+0.1, color='black', linestyle='-.')
    plq(plt, hi, timestr, hi, 'voc_f', add=0.1, color='green', linestyle=':', warn=False)
    plq(plt, hi, timestr, hi, 'voc', add=0.1, color='cyan', linestyle=':', warn=False)
    plt.legend(loc=1)
    plt.subplot(335)
    plq(plt, hi, timestr, hi, 'e_wrap_filt', color='black', linestyle='--', warn=False)
    plq(plt, hi, timestr, hi, 'e_wrap', color='black', linestyle='--', warn=False)
    plq(plt, hi, timestr, hi, 'ewhi_thr', color='red', linestyle='-.')
    plq(plt, hi, timestr, hi, 'ewlo_thr', color='red', linestyle='-.')
    plt.ylim(-1, 1)
    plt.legend(loc=1)
    plt.subplot(336)
    plq(plt, hi, timestr, hi, 'wrap_hi_flt', add=6, color='blue', linestyle='-')
    plq(plt, hi, timestr, hi, 'wrap_hi_m_flt', add=6, color='red', linestyle='--')
    plq(plt, hi, timestr, hi, 'wrap_hi_n_flt', add=6, color='orange', linestyle='-.')
    plq(plt, hi, timestr, hi, 'wrap_lo_flt', add=4, color='blue', linestyle='-')
    plq(plt, hi, timestr, hi, 'wrap_lo_m_flt', add=4, color='red', linestyle='--')
    plq(plt, hi, timestr, hi, 'wrap_lo_n_flt', add=4, color='orange', linestyle='-.')
    plt.legend(loc=1)
    plt.subplot(339)
    plq(plt, hi, timestr, hi, 'wrap_hi_fa', add=6, color='blue', linestyle='-')
    plq(plt, hi, timestr, hi, 'wrap_hi_m_fa', add=6, color='red', linestyle='--')
    plq(plt, hi, timestr, hi, 'wrap_hi_n_fa', add=6, color='orange', linestyle='-.')
    plq(plt, hi, timestr, hi, 'wrap_lo_fa', add=4, color='blue', linestyle='-')
    plq(plt, hi, timestr, hi, 'wrap_lo_m_fa', add=4, color='red', linestyle='--')
    plq(plt, hi, timestr, hi, 'wrap_lo_n_fa', add=4, color='orange', linestyle='-.')
    plq(plt, hi, timestr, hi, 'wv_fa', add=2, color='orange', linestyle='-.')
    plq(plt, hi, timestr, hi, 'ccd_fa', color='green', linestyle='-')
    plq(plt, hi, timestr, hi, 'red_loss', color='blue', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(337)
    plq(plt, hi, timestr, hi, 'cc_dif', color='black', linestyle='-')
    plq(plt, hi, timestr, hi, 'cc_diff_thr', color='red', linestyle='--')
    plq(plt, hi, timestr, hi, 'cc_diff_thr', slr=-1, color='red', linestyle='--')
    # plt.ylim(-.01, .01)
    plt.legend(loc=1)
    plt.subplot(338)
    plq(plt, hi, timestr, hi, 'ib_diff_flt', add=2, color='cyan', linestyle='-')
    plq(plt, hi, timestr, hi, 'ib_diff_fa', add=2, color='magenta', linestyle='--')
    plq(plt, hi, timestr, hi, 'vb_flt', color='blue', linestyle='-.')
    plq(plt, hi, timestr, hi, 'vb_fa_lt', color='black', linestyle=':')
    plq(plt, hi, timestr, hi, 'Tb_flt', color='red', linestyle='-')
    plq(plt, hi, timestr, hi, 'Tb_fa', color='cyan', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if save_plots:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


# noinspection PyUnusedLocal
def overall_fault(mr, mv, sr, sv, smr, smv, filename, fig_files=None, plot_title=None, fig_list=None, run_type=None,
                  save_plots=False):
    print('overall_fault', end=':  ')
    if fig_files is None:
        fig_files = []

    fig_list.append(plt.figure())  # of 1
    plt.subplot(331)
    plt.suptitle(plot_title + ' O_F 1')
    print('0_F 1', end=':  ')
    plq(plt, mr, 'time', mr, 'ib_sel', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_sel', color='cyan', linestyle='--', warn=not run_type=='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'ib_in', color='cyan', linestyle='--', warn=not run_type=='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'ib_in_s', color='orange', linestyle='-.', warn=not run_type=='HistHist' and not run_type=='HistSim')
    if not run_type=='HistHist':
        plq(plt, mr, 'time', mr, 'Tb', color='red', linestyle='-', warn=not run_type=='HistSim')
        plq(plt, mv, 'time', mv, 'Tb', color='blue', linestyle='--', warn=not run_type=='HistSim')
        plq(plt, smv, 'time', smv, 'Tb', color='green', linestyle='-.', warn=not run_type=='HistSim')
    else:
        plq(plt, mr, 'time', mr, 'Tb_f', color='red', linestyle='-')
        plq(plt, mv, 'time', mv, 'Tb_f', color='blue', linestyle='--')
    plt.legend(loc=1)
    if not run_type=='HistHist':
        plt.subplot(332)
        plq(plt, mr, 'time', mr, 'ioc', color='black', linestyle='-', warn=not run_type=='HistSim')
        plq(plt, mv, 'time', mv, 'ioc', color='cyan', linestyle='--', warn=not run_type=='HistSim')
        plq(plt, sv, 'time', sv, 'ioc', color='orange', linestyle=':', warn=not run_type=='HistSim')
        plt.legend(loc=1)
    plt.subplot(333)
    plq(plt, mr, 'time', mr, 'dV_hys', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_hys', color='cyan', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, sv, 'time', sv, 'dv_hys', color='orange', linestyle='-.', warn=not run_type=='HistHist' and not run_type=='HistSim')
    plt.legend(loc=1)
    plt.subplot(334)
    plq(plt, mr, 'soc', mr, 'vb', color='black', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'soc', mr, 'vb_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'vb_f', color='cyan', linestyle='--', warn= not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'vb_s', color='orange', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.legend(loc=1)
    plt.subplot(335)
    plq(plt, mr, 'time', mr, 'voc', color='black', linestyle='-', warn=run_type!='HistHist'and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'voc_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc', color='cyan', linestyle='--', warn=run_type!='HistHist'and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'voc_f', color='cyan', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'voc_s', color='orange', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.legend(loc=1)
    plt.subplot(336)
    plq(plt, mr, 'time', mr, 'soc', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='cyan', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'soc_s', color='orange', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'soc_ekf', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc_ekf', color='red', linestyle='--', warn=not run_type=='HistSim')
    plt.legend(loc=1)
    plt.subplot(337)
    plq(plt, mr, 'time', mr, 'voc_stat', color='black', linestyle='-', warn=run_type!='HistHist'and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'voc_stat_f', color='black', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_stat', color='cyan', linestyle='--', warn=run_type!='HistHist'and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'voc_stat_f', color='cyan', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'voc_stat_s', color='orange', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.legend(loc=1)
    plt.subplot(338)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'e_wrap_filt', color='black', linestyle='-')
    if hasattr(mv, 'voc_stat'):
        mv.e_wrap = np.array(mv.voc_soc) - np.array(mv.voc_stat)
    elif hasattr(mv, 'voc_soc') and hasattr(mv, 'voc_stat_f'):
        mv.e_wrap = np.array(mv.voc_soc) - np.array(mv.voc_stat_f)
    mr.cc_dif = np.array(mr.soc_ekf) - np.array(mr.soc)
    if hasattr(mv, 'soc_ekf'):
        mv.cc_dif = np.array(mv.soc_ekf) - np.array(mv.soc)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-', warn=run_type!='HistHist'and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'e_wrap', color='cyan', linestyle='--', warn=run_type!='HistHist'and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'e_wrap_filt', color='black', linestyle='--', warn=run_type!='HistHist')
    plq(plt, mv, 'time', mv, 'e_wrap_filt', color='cyan', linestyle='--', warn=run_type!='HistHist'and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'cc_dif', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'cc_dif', color='red', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'cc_diff_thr', color='cyan', linestyle='--')
    plq(plt, mr, 'time', mr, 'cc_diff_thr', slr=-1, color='cyan', linestyle='--')
    plq(plt, mr, 'time', mr, 'ewhi_thr', color='red', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ewlo_thr', color='red', linestyle='-.')
    plt.ylim(-.5, .5)
    plt.legend(loc=1)
    if run_type != 'HistHist' and run_type != 'HistSim':
        plt.subplot(339)
        plq(plt, mr, 'time', mr, 'dv_dyn', color='black', linestyle='-')
        plq(plt, mv, 'time', mv, 'dv_dyn', color='cyan', linestyle='--')
        plq(plt, smv, 'time', smv, 'dv_dyn_s', color='orange', linestyle='-.')
        plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if save_plots:
        plt.savefig(fig_file_name, format="png")

    fig_list.append(plt.figure())  # O_F 2
    plt.subplot(331)
    plt.suptitle(plot_title + ' O_F 2')
    print('0_F 2', end=':  ')
    if hasattr(mr, 'vb'):
        mr.dv_dyn = mr.vb - mr.voc
    else:
        mr.dv_dyn_f = mr.vb_f - mr.voc_f
    plq(plt, mr, 'time', mr, 'dv_dyn', color='blue', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'dv_dyn_f', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'dv_dyn', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'dv_dyn_f', color='cyan', linestyle='--', warn=not run_type=='HistSim' and not run_type=='HistSim')
    plq(plt, smr, 'time', smr, 'dv_dyn_s', color='black', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'dv_dyn_s', color='magenta', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(332)
    plq(plt, mr, 'time', mr, 'soc', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='cyan', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, smr, 'time', smr, 'soc_s', color='black', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'soc_s', color='magenta', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'soc_ekf', color='green', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc_ekf', color='red', linestyle='--', warn=not run_type=='HistSim')
    plt.xlabel('sec')
    plt.legend(loc=4)
    plt.subplot(333)
    plq(plt, mr, 'time', mr, 'ib_sel', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'ib_sel', color='cyan', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'ib_in', color='blue', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'ib_in', color='magenta', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smr, 'time', smr, 'ib_in_s', color='black', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'ib_in_s', color='red', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(334)
    plq(plt, mr, 'time', mr, 'voc', color='blue', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'voc_f', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'voc_f', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'voc_stat', color='orange', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'voc_stat_f', color='orange', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_stat', color='red', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'voc_stat_f', color='red', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smr, 'time', smr, 'voc_stat_s', color='black', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'voc_stat_s', color='red', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smr, 'time', smr, 'vb_s', color='purple', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'vb_s', color='pink', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(335)
    plq(plt, mr, 'time', mr, 'e_wrap', color='black', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'e_wrap_filt', color='black', linestyle='-', warn=run_type!='HistHist')
    plq(plt, mv, 'time', mv, 'e_wrap', color='orange', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'e_wrap_filt', color='orange', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(336)
    plq(plt, mr, 'time', mr, 'vb', color='blue', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'vb_f', color='blue', linestyle='-')
    plq(plt, mv, 'soc', mv, 'vb', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'soc', mv, 'vb_f', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smr, 'soc_s', smr, 'vb_s', color='black', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='magenta', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'voc_stat', color='orange', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'voc_stat_f', color='orange', linestyle='-')
    plq(plt, mv, 'time', mv, 'voc_stat', color='blue', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'voc_stat_f', color='blue', linestyle='--', warn=not run_type=='HistSim')
    plt.xlabel('state-of-charge')
    plt.legend(loc=2)
    plt.subplot(337)
    plq(plt, mr, 'time', mr, 'vb', color='blue', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'vb_f', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'vb', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'vb_f', color='cyan', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, smr, 'time', smr, 'vb_s', color='black', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'vb_s', color='magenta', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.xlabel('sec')
    plt.legend(loc=2)
    plt.subplot(338)
    plq(plt, mr, 'time', mr, 'dv_hys', color='blue', linestyle='-')
    plq(plt, mr, 'time', mr, 'dv_hys_f', color='blue', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'dv_hys', color='cyan', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, smr, 'time', smr, 'dv_hys_s', color='black', linestyle='-.', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, smv, 'time', smv, 'dv_hys_s', color='magenta', linestyle=':', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.xlabel('sec')
    plt.legend(loc=3)
    plt.subplot(339)
    plq(plt, mr, 'time', mr, 'Tb', color='blue', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'Tb_f', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'Tb', color='red', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'Tb_f', color='red', linestyle='--', warn=not run_type=='HistSim')
    plq(plt, mr, 'time', mr, 'tau_hys', color='black', linestyle='-', warn=run_type!='HistHist' and not run_type=='HistSim')
    plq(plt, mv, 'time', mv, 'tau_hys', color='cyan', linestyle='--', warn=run_type!='HistHist' and not run_type=='HistSim')
    plt.legend(loc=3)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    if save_plots:
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


# noinspection PyShadowingNames
def calc_fault(d_ra, d_mod):
    falw = d_ra.falw.astype(int)
    fltw = d_ra.fltw.astype(int)
    ib_noa_bare_flt =np.bool_(falw & 2 ** 12)
    ib_amp_bare_flt =np.bool_(falw & 2 ** 11)
    ib_dscn_fa = np.bool_(falw & 2 ** 10)
    ib_diff_fa = np.bool_((falw & 2 ** 8) | (falw & 2 ** 9))
    wv_fa = np.bool_(falw & 2 ** 7)
    wrap_lo_fa = np.bool_(falw & 2 ** 6)
    wrap_hi_fa = np.bool_(falw & 2 ** 5)
    vc_flt = np.bool_(np.array(fltw) & 2 ** 13)
    wrap_hi_m_flt = np.bool_(np.array(fltw) & 2 ** 14)
    wrap_lo_m_flt = np.bool_(np.array(fltw) & 2 ** 15)
    wrap_hi_n_flt = np.bool_(np.array(fltw) & 2 ** 16)
    wrap_lo_n_flt = np.bool_(np.array(fltw) & 2 ** 17)
    vc_fa = np.bool_(np.array(falw) & 2 ** 13)
    wrap_hi_m_fa = np.bool_(np.array(falw) & 2 ** 14)
    wrap_lo_m_fa = np.bool_(np.array(falw) & 2 ** 15)
    wrap_hi_n_fa = np.bool_(np.array(falw) & 2 ** 16)
    wrap_lo_n_fa = np.bool_(np.array(falw) & 2 ** 17)
    ccd_fa = np.bool_(falw & 2 ** 4)
    ib_noa_flt = np.bool_(fltw & 2 ** 3)
    ib_noa_fa = np.bool_(falw & 2 ** 3)
    ib_amp_flt = np.bool_(fltw & 2 ** 2)
    ib_amp_fa = np.bool_(falw & 2 ** 2)
    vb_fa_lt = np.bool_(falw & 2 ** 1)
    Tb_fa = np.bool_(falw & 2 ** 0)
    Tb_fa = np.bool_(falw & 2 ** 19)
    d_mod = rf.rec_append_fields(d_mod, 'ib_noa_bare_flt', np.array(ib_noa_bare_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_amp_bare_flt', np.array(ib_amp_bare_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_dscn_fa', np.array(ib_dscn_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_diff_fa', np.array(ib_diff_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wv_fa', np.array(wv_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_lo_fa', np.array(wrap_lo_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_hi_fa', np.array(wrap_hi_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_hi_m_flt', np.array(wrap_hi_m_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_hi_n_flt', np.array(wrap_hi_n_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_lo_m_flt', np.array(wrap_lo_m_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_lo_n_flt', np.array(wrap_lo_n_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_hi_m_fa', np.array(wrap_hi_m_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_hi_n_fa', np.array(wrap_hi_n_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_lo_m_fa', np.array(wrap_lo_m_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'wrap_lo_n_fa', np.array(wrap_lo_n_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'vc_fa', np.array(vc_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'vc_flt', np.array(vc_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ccd_fa', np.array(ccd_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_noa_flt', np.array(ib_noa_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_amp_flt', np.array(ib_amp_flt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_noa_fa', np.array(ib_noa_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'ib_amp_fa', np.array(ib_amp_fa, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'vb_fa_lt', np.array(vb_fa_lt, dtype=float))
    d_mod = rf.rec_append_fields(d_mod, 'Tb_fa', np.array(Tb_fa, dtype=float))

    try:
        ib_diff_flt = np.bool_((fltw & 2 ** 8) | (fltw & 2 ** 9))
        wrap_hi_flt = np.bool_(fltw & 2 ** 5)
        wrap_lo_flt = np.bool_(fltw & 2 ** 6)
        red_loss = np.bool_(fltw & 2 ** 7)
        ib_dscn_flt = np.bool_(fltw & 2 ** 10)
        vb_flt = np.bool_(fltw & 2 ** 1)
        Tb_flt = np.bool_(fltw & 2 ** 0)
        d_mod = rf.rec_append_fields(d_mod, 'ib_diff_flt', np.array(ib_diff_flt, dtype=float))
        d_mod = rf.rec_append_fields(d_mod, 'wrap_hi_flt', np.array(wrap_hi_flt, dtype=float))
        d_mod = rf.rec_append_fields(d_mod, 'wrap_lo_flt', np.array(wrap_lo_flt, dtype=float))
        d_mod = rf.rec_append_fields(d_mod, 'red_loss', np.array(red_loss, dtype=float))
        d_mod = rf.rec_append_fields(d_mod, 'ib_dscn_flt', np.array(ib_dscn_flt, dtype=float))
        d_mod = rf.rec_append_fields(d_mod, 'vb_flt', np.array(vb_flt, dtype=float))
        d_mod = rf.rec_append_fields(d_mod, 'Tb_flt', np.array(Tb_flt, dtype=float))
    except IOError:
        pass

    return d_mod


# Fake stuff to get replicate to accept inputs and run
def bandaid(h, chm_in=0):
    res = np.zeros(len(h.time_ux))
    res[0:10] = 1
    mod = np.zeros(len(h.time_ux))
    ib_sel = h['ib'].copy()
    vb_sel = h['vb'].copy()
    tb_sel = h['Tb'].copy()
    voc = h['voc'].copy()
    ib_in_s = h['ib'].copy()
    soc_s = h['soc'].copy()
    bms_off_s = h['bms_off'].copy()
    sat_s = h['saturated'].copy()
    chm = np.ones(len(h.time_ux))*chm_in
    sel = np.zeros(len(h.time_ux))
    preserving = np.ones(len(h.time_ux))
    chm_s = np.ones(len(h.time_ux))*chm_in
    mon_run = rf.rec_append_fields(h, 'res', res)
    mon_run = rf.rec_append_fields(mon_run, 'mod_data', mod)
    mon_run = rf.rec_append_fields(mon_run, 'ib_past', ib_in_s)
    if not hasattr(mon_run, 'ib_sel'):
        mon_run = rf.rec_append_fields(mon_run, 'ib_sel', ib_sel)
    mon_run = rf.rec_append_fields(mon_run, 'tb_sel', tb_sel)
    if not hasattr(mon_run, 'voc'):
        mon_run = rf.rec_append_fields(mon_run, 'voc', voc)
    mon_run = rf.rec_append_fields(mon_run, 'preserving', preserving)
    mon_run = rf.rec_append_fields(mon_run, 'vb_sel', vb_sel)
    mon_run = rf.rec_append_fields(mon_run, 'soc_s', soc_s)
    mon_run = rf.rec_append_fields(mon_run, 'chm', chm)
    mon_run = rf.rec_append_fields(mon_run, 'sel', sel)
    mon_run = rf.rec_append_fields(mon_run, 'ewhi_thr', sel)
    mon_run = rf.rec_append_fields(mon_run, 'ewlo_thr', sel)
    mon_run = rf.rec_append_fields(mon_run, 'ccd_thr', sel)
    mon_run = rf.rec_append_fields(mon_run, 'voc_ekf', sel)
    mon_run = rf.rec_append_fields(mon_run, 'y', sel)
    sim_run = np.array(np.zeros(len(h.time_ux), dtype=[('time_ux', '<i4')])).view(np.recarray)
    sim_run.time_ux = mon_run.time_ux.copy()
    sim_run = rf.rec_append_fields(sim_run, 'chm_s', chm_s)
    sim_run = rf.rec_append_fields(sim_run, 'sat_s', sat_s)
    sim_run = rf.rec_append_fields(sim_run, 'ib_in_s', ib_in_s)
    sim_run = rf.rec_append_fields(sim_run, 'bms_off_s', bms_off_s)
    sim_run = rf.rec_append_fields(sim_run, 'dv_dyn_s', bms_off_s)
    sim_run = rf.rec_append_fields(sim_run, 'dv_hys_s', bms_off_s)
    sim_run = rf.rec_append_fields(sim_run, 'voc_stat_s', bms_off_s)
    return mon_run, sim_run

# Make an array useful for analysis (around temp) and add some metrics
def filter_Tb(raw, temp_corr, mon, tb_band=5., rated_batt_cap=100.):
    h = raw[abs(raw.Tb_f - temp_corr) < tb_band]

    saturated_ = np.copy(h.Tb_f)
    bms_off_ = np.copy(h.Tb_f)
    for i in range(len(h.Tb_f)):
        saturated_[i] = is_sat(h.Tb_f[i], mon.chemistry.rated_temp, h.voc_f[i], h.soc[i], mon.chemistry.nom_vsat, mon.chemistry.dvoc_dt,
                         mon.chemistry.low_t, vsat_add=mon.sp_vsat_add)
        bms_off_[i] = (h.Tb_f[i] < mon.chemistry.low_t) or ((h.voc_stat_f[i] < 10.5) and (h.ib_f[i] < Battery.IB_MIN_UP))

    # Correct for temp
    q_cap = calculate_capacity(q_cap_rated_scaled=rated_batt_cap * 3600., dqdt=mon.chemistry.dqdt, tb_f=h.Tb_f,
                               t_rated=mon.chemistry.rated_temp)
    dq = (h.soc - 1.) * q_cap
    dq -= mon.chemistry.dqdt * q_cap * (temp_corr - h.Tb_f)
    q_cap_r = calculate_capacity(q_cap_rated_scaled=rated_batt_cap * 3600., dqdt=mon.chemistry.dqdt, tb_f=temp_corr,
                                 t_rated=mon.chemistry.rated_temp)
    soc_r = 1. + dq / q_cap_r
    h = rf.rec_append_fields(h, 'soc_r', soc_r)
    h.voc_stat_r = h.voc_stat_f - (h.Tb_f - temp_corr) * mon.chemistry.dvoc_dt

    # delineate charging and discharging
    voc_stat_r_chg = np.copy(h.voc_stat_f)
    voc_stat_r_dis = np.copy(h.voc_stat_f)
    for i in range(len(voc_stat_r_chg)):
        if h.ib_f[i] > -0.5:
            voc_stat_r_dis[i] = None
        elif h.ib_f[i] < 0.5:
            voc_stat_r_chg[i] = None

    # Hysteresis
    if len(h.time_ux) > 1:
        t_s_min = h.time_min[0]
        t_e_min = h.time_min[-1]
        dt_hys_min = 1.
        hys_time_min = np.arange(t_s_min, t_e_min, dt_hys_min, dtype=float)
        min_per_month = 30*24*60
        if len(hys_time_min) > 2 * min_per_month:
            print(Colors.fg.red)
            print("HUGE time range.  Something is wrong with time")
            print(Colors.reset)
            return None
        h = rf.rec_append_fields(h, 'sat', saturated_)
        h = rf.rec_append_fields(h, 'saturated', saturated_)
        h = rf.rec_append_fields(h, 'bms_off', bms_off_)
        h = rf.rec_append_fields(h, 'voc_stat_r_dis', voc_stat_r_dis)
        h = rf.rec_append_fields(h, 'voc_stat_r_chg', voc_stat_r_chg)
    return h


# Table lookup vector
def look_it(x, tab, temp):
    voc = x.copy()
    for i in range(len(x)):
        voc[i] = tab.interp(x[i], temp)
    return voc
