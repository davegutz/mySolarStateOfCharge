# CompareHistSim.py:  load fault, hist, summ data and compare to simulation.
# Copyright (C) 2026 Dave Gutz
#
# noinspection PyPep8Naming,PyUnboundLocalVariable,PyShadowingNames,PyShadowingBuiltins,PyUnresolvedReferences,PyAttributeOutsideInit
# type: ignore
# pylint: disable=invalid-name, used-before-assignment, redefined-outer-name, redefined-builtin, no-member, attribute-defined-outside-init
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
from battery_constants import load_off_nominal_battery, apply_off_nominal_battery
from Battery import Battery, BatteryMonitor, is_sat, calculate_capacity
from MonSim import replicate, save_clean_file, save_fault_coverage, UserOptions
from resample import resample
from PlotKiller import show_killer
from DataOverModel import dom_plot
from DataOverModel import write_clean_file
from CompareFault import overall_fault, over_fault
from unite_pictures import cleanup_fig_files, precleanup_fig_files, pngs_to_pdf
from datetime import datetime
from load_data import load_data, remove_nan, remove_0T
from local_paths import version_from_data_file, local_paths
from CompareFault import add_stuff_f
from Util import rename_all
from pathlib import Path, PurePosixPath
from plot.PlotOptions import PlotOptions

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#  For this battery Battleborn 100 Ah with 1.084 x capacity
IB_BAND = 1.  # Threshold to declare charging or discharging
TB_BAND = 25.  # Band around temperature to group data and correct.  Large value means no banding, effectively

# Calculate thresholds from global input values listed above (review these)
# noinspection PyPep8Naming
def fault_thr_bb(Tb, soc, voc_soc, voc_stat, C_rate, bb):
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
    cc_diff_thr = CC_DIFF_SOC_DIS_THRESH * Battery.ap_cc_diff_slr * cc_diff_empty_sclr_

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
    ib_diff_sclr_ = 1.  # ram adjusts during data collection
    ib_diff_thr = IBATT_DISAGREE_THRESH * ib_diff_sclr_

    # ib_quiet
    ib_quiet_sclr_ = 1.  # ram adjusts during data collection
    ib_quiet_thr = QUIET_A * ib_quiet_sclr_

    return cc_diff_thr, ewhi_thr, ewlo_thr, ib_diff_thr, ib_quiet_thr


def hstack2(arrays):
    if arrays[0] is None and arrays[1] is None:
        return None
    elif arrays[1] is None:
        return arrays[0]
    elif arrays[0] is None:
        return arrays[1]

    return arrays[0].__array_wrap__(np.hstack(arrays))


# Fake stuff to get replicate to accept inputs and run
def bandaid(h):
    res = np.zeros(len(h.time_ux))
    res[0:1] = 1
    time_t = h['time'].copy()
    ib_sel = h['ib_f'].copy()  # TODO: fix selection logic
    ib_in_s = h['ib_f'].copy()
    vb_sel = h['vb_f'].copy()
    tb_sel = h['Tb_f'].copy()
    voc_f = h['voc_f'].copy()
    soc_s = h['soc'].copy()
    bms_off_s = h['bms_off'].copy()
    sat_s = h['saturated'].copy()
    chm_s = h['chm_s'].copy()
    dt_s = h['dt'].copy()
    dv_dyn_s = h['dt'].copy()*0.
    dv_hys_s = h['dt'].copy()*0.
    deltaq_s = h['delta_q'].copy()
    voc_stat_s = h['voc_stat_f'].copy()
    sel = np.zeros(len(h.time_ux))
    preserving = np.ones(len(h.time_ux))
    mon_run = rf.rec_append_fields(h, 'res', res)
    mon_run = rf.rec_append_fields(mon_run, 'ib_past', ib_in_s)
    if not hasattr(mon_run, 'ib_sel'):
        mon_run = rf.rec_append_fields(mon_run, 'ib_sel', ib_sel)
    else:
        mon_run.ib_sel = ib_sel.copy()
    mon_run = rf.rec_append_fields(mon_run, 'tb_sel', tb_sel)
    if not hasattr(mon_run, 'voc_f'):
        mon_run = rf.rec_append_fields(mon_run, 'voc_f', voc_f)
    mon_run = rf.rec_append_fields(mon_run, 'time_t', time_t)
    mon_run = rf.rec_append_fields(mon_run, 'preserving', preserving)
    mon_run = rf.rec_append_fields(mon_run, 'vb_sel', vb_sel)
    mon_run = rf.rec_append_fields(mon_run, 'soc_s', soc_s)
    mon_run = rf.rec_append_fields(mon_run, 'sel', sel)
    mon_run = rf.rec_append_fields(mon_run, 'ccd_thr', sel)
    mon_run = rf.rec_append_fields(mon_run, 'voc_ekf', sel)
    mon_run = rf.rec_append_fields(mon_run, 'y', sel)
    sim_run = np.array(np.zeros(len(h.time), dtype=[('time', '<f8')])).view(np.recarray)
    sim_run.time = mon_run.time.copy()
    sim_run = rf.rec_append_fields(sim_run, 'dt_s', dt_s)
    sim_run = rf.rec_append_fields(sim_run, 'chm_s', chm_s)
    sim_run = rf.rec_append_fields(sim_run, 'sat_s', sat_s)
    sim_run = rf.rec_append_fields(sim_run, 'ib_in_s', ib_in_s)
    sim_run = rf.rec_append_fields(sim_run, 'bms_off_s', bms_off_s)
    sim_run = rf.rec_append_fields(sim_run, 'dv_dyn_s', dv_dyn_s)
    sim_run = rf.rec_append_fields(sim_run, 'dv_hys_s', dv_hys_s)
    sim_run = rf.rec_append_fields(sim_run, 'voc_stat_s', voc_stat_s)
    sim_run = rf.rec_append_fields(sim_run, 'delta_q_s', deltaq_s)
    return mon_run, sim_run


# Make an array useful for analysis (around temp) and add some metrics
def filter_Tb(raw, tb_forr, mon, tb_band=5., rated_batt_cap=None):
    h = raw[abs(raw.Tb_f - tb_forr) < tb_band]

    saturated_ = np.copy(h.Tb_f)
    bms_off_ = np.copy(h.Tb_f)
    for i in range(len(h.Tb_f)):
        saturated_[i] = is_sat(h.Tb_f[i], mon.chemistry.rated_temp, h.voc_f[i], h.soc[i], mon.chemistry.nom_vsat,
                         mon.chemistry.dvoc_dt, mon.chemistry.low_t)
        bms_off_[i] = (h.Tb_f[i] < mon.chemistry.low_t) or ((h.voc_stat_f[i] < 10.5) and (h.ib_f[i] < Battery.IB_MIN_UP))

    # Correct for temp
    q_cap = calculate_capacity(q_cap_rated_scaled=rated_batt_cap * 3600., dqdt=mon.chemistry.dqdt, tb_f=h.Tb_f,
                               t_rated=mon.chemistry.rated_temp)
    dq = (h.soc - 1.) * q_cap
    dq -= mon.chemistry.dqdt * q_cap * (tb_forr - h.Tb_f)
    q_cap_r = calculate_capacity(q_cap_rated_scaled=rated_batt_cap * 3600., dqdt=mon.chemistry.dqdt, tb_f=tb_forr,
                                 t_rated=mon.chemistry.rated_temp)
    soc_r = 1. + dq / q_cap_r
    h = rf.rec_append_fields(h, 'soc_r', soc_r)
    h.voc_stat_r = h.voc_stat_f - (h.Tb_f - tb_forr) * mon.chemistry.dvoc_dt

    # delineate charging and discharging
    voc_stat_r_chg = np.copy(h.voc_stat_f)
    voc_stat_r_dis = np.copy(h.voc_stat_f)
    for i in range(len(voc_stat_r_chg)):
        if h.ib_f[i] > -0.5:
            voc_stat_r_dis[i] = None
        elif h.ib_f[i] < 0.5:
            voc_stat_r_chg[i] = None
    if len(h.time_ux) > 1:
        t_s_min = h.time_min[0]
        t_e_min = h.time_min[-1]
        dt_hys_min = 1.  # ??????????????????????????????
        print(f" {t_s_min=} {t_e_min=} {dt_hys_min=}  days of data = {(t_e_min-t_s_min)/(24.*60)} ", end='')
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


def shift_time(mr, extra_shift=0.):
    # Shift time
    first_non_zero = 0
    n = len(mr.time)
    while abs(mr.ib_f[first_non_zero]) < 0.02 and first_non_zero < n-1:
        first_non_zero += 1
    if first_non_zero < n:  # success
        if first_non_zero > 0:
            shift = (mr.time[first_non_zero] + mr.time[first_non_zero-1]) / 2.
        else:
            shift = mr.time[first_non_zero]
        print('shift time by', shift)
        mr.time = mr.time - shift + extra_shift
        mr.time_e = mr.time_e - shift + extra_shift
    return mr


def add_chm(hist, mon_t_=False, mon=None, chm=None):
    if not mon_t_ or mon is None:
        print("add_chm:  not executing")
        if chm is not None:
            chm_s = []
            for i in range(len(hist.time)):
                chm_s.append(chm)
            hist = rf.rec_append_fields(hist, 'chm_s', np.array(chm_s, dtype=int))
            hist = rf.rec_append_fields(hist, 'chm', np.array(chm_s, dtype=int))
        return hist
    else:
        chm = []
        for i in range(len(hist.time_ux)):
            t_sec = float(hist.time_ux[i] - hist.time_ux[0]) + mon.time[0]
            chm.append(np.interp(t_sec, mon.time, mon.chm))
        hist = rf.rec_append_fields(hist, 'chm_s', np.array(chm, dtype=int))
        hist = rf.rec_append_fields(hist, 'chm', np.array(chm, dtype=int))
    return hist

def add_delta_q(hist):
    delta_q = []
    for i in range(len(hist.time_ux)):
        q_capacity = hist.qcrs[i] * (1. + hist.dqdt[i]*(hist.Tb_f[i] - Battery.RATED_TEMP))
        delta_q.append(-q_capacity * (1. - hist.soc[i]))
    hist = rf.rec_append_fields(hist, 'delta_q', np.array(delta_q, dtype=float))
    return hist

def add_mod(hist, mon_t_=False, mon=None):
    if not mon_t_ or mon is None:
        print("add_mod:  not executing")
        return hist
    else:
        mod_data = []
        for i in range(len(hist.time_ux)):
            t_sec = float(hist.time_ux[i]) - float(hist.time_ux[0]) + mon.time[0]
            mod_data.append(np.interp(t_sec, mon.time, mon.mod_data))
        return rf.rec_append_fields(hist, 'mod_data', np.array(mod_data, dtype=int))


def add_qcrs(hist, mon_t_=False, mon=None, qcrs=None, t_rated=None, dqdt=None):
    if not mon_t_ or mon is None:
        print("add_qcrs:  not executing")
        if qcrs is not None:
            qcrs_m = []
            t_rated_m = []
            dqdt_m = []
            for i in range(len(hist.time)):
                qcrs_m.append(qcrs)
                t_rated_m.append(t_rated)
                dqdt_m.append(dqdt)
            hist = rf.rec_append_fields(hist, 'qcrs_s', np.array(qcrs_m, dtype=float))
            hist = rf.rec_append_fields(hist, 'qcrs', np.array(qcrs_m, dtype=float))
            hist = rf.rec_append_fields(hist, 't_rated', np.array(t_rated_m, dtype=float))
            hist = rf.rec_append_fields(hist, 'dqdt', np.array(dqdt_m, dtype=float))
        return hist
    else:
        qcrs_m = []
        t_rated_m = []
        dqdt_m = []
        for i in range(len(hist.time_ux)):
            t_sec = float(hist.time_ux[i] - hist.time_ux[0]) + mon.time[0]
            qcrs_m.append(np.interp(t_sec, mon.time, mon.chm))
            t_rated_m.append(t_rated)
            dqdt_m.append(dqdt)
        hist = rf.rec_append_fields(hist, 'qcrs_s', np.array(qcrs_m, dtype=float))
        hist = rf.rec_append_fields(hist, 'qcrs', np.array(qcrs_m, dtype=float))
        hist = rf.rec_append_fields(hist, 't_rated', np.array(t_rated_m, dtype=float))
        hist = rf.rec_append_fields(hist, 'dqdt', np.array(dqdt_m, dtype=float))
    return hist


# noinspection PyPep8Naming
def scale_large_time(D):
    if D is None:
        return D
    time_ux = np.atleast_1d(D.time_ux)
    if np.any(time_ux > 1e12):
        D.time_ux /= 1000.
    return D


# noinspection PyPep8Naming
def load_hist_and_prep(data_file=None, time_end=None, plots=True, use_mon_csv=False, unit_key=None,
                       sync_time=None, dt_resample=10, Tb_force=None, skip=1):
    """Load history, reconstruct samples by linear interpolation and normalize all soc and Tb to 20C"""

    print(f"\nload_hist_and_prep:\n{data_file=}\n{plots=}\n{use_mon_csv=}\n{unit_key=}\n{dt_resample=}\n{Tb_force=}\n{skip=}\n")

    # Load battery (ref)
    battery_hdr = "Battery_hdr"
    battery_val = "Battery_val"
    battery_file_clean = write_clean_file(data_file, type_='_battery', hdr_key=battery_hdr,
                                          unit_key=battery_val, skip=skip)
    if battery_file_clean:
        battery_raw = np.genfromtxt(battery_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        battery_raw = None
        print(f"load_hist_and_prep: returning battery_raw=None")
    # Override Battery with loaded values Battery_hdr/Battery_val in battery_raw
    Battery_off_dict = load_off_nominal_battery(Battery_to_add=battery_raw)
    apply_off_nominal_battery(Battery, Battery_off_dict)

    rated_batt_cap = Battery.NOM_UNIT_CAP * Battery.sp_s_cap_mon
    qcrs = rated_batt_cap * 3600.


    # Save these
    rated_batt_cap = Battery.NOM_UNIT_CAP * Battery.sp_s_cap_mon
    # Reconstruction of soc using subsampled data is poor.  Drive everything with soc from Monitor
    dvoc_mon = 0.

    # Load mon to extract mod information
    # # Load mon v4 (old)
    if use_mon_csv:
        mon, sim, fault, mon_t_file_clean, temp_mont_t_file_clean, _ = \
            load_data(data_file, 1, unit_key=unit_key, time_end=time_end, zero_zero=False, mon_str='hist')
        mon = rename_all(mon)
    else:
        mon = None

    # Load summaries
    s_raw = None
    temp_sum_file_clean = write_clean_file(data_file, type_='_flt', hdr_key='fltb', unit_key='unit_u',
                                           skip=1, comment_str='---')
    if temp_sum_file_clean:
        s_raw = np.genfromtxt(temp_sum_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        print("data from", temp_sum_file_clean, "empty after loading")

    s_raw = scale_large_time(s_raw)

    # Load history
    h_raw = None
    temp_hist_file_clean = write_clean_file(data_file, type_='_flt', hdr_key='fltb', unit_key='unit_h',
                                            skip=1, comment_str='---')
    if temp_hist_file_clean:
        h_raw = np.genfromtxt(temp_hist_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        print("data from", temp_hist_file_clean, "empty after loading")
    h_raw = scale_large_time(h_raw)

    # Load fault
    temp_flt_file_clean = write_clean_file(data_file, type_='_flt', hdr_key='fltb', unit_key='unit_f',
                                           skip=1, comment_str='---')
    if temp_flt_file_clean:
        f_raw = np.genfromtxt(temp_flt_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        f_raw = None
        print("data from", temp_flt_file_clean, "empty after loading")
    f_raw = scale_large_time(f_raw)

    # Save files
    filename = None
    if temp_flt_file_clean is not None:
        filename = (PurePosixPath(temp_flt_file_clean).name.replace('.csv', '_')
                    + PurePosixPath( Path(__file__).as_posix()).stem)
    elif temp_hist_file_clean is not None:
        filename = (PurePosixPath(temp_hist_file_clean).name.replace('.csv', '_')
                    + PurePosixPath( Path(__file__).as_posix()).stem)
    elif temp_sum_file_clean is not None:
        filename = (PurePosixPath(temp_sum_file_clean).name.replace('.csv', '_')
                    + PurePosixPath( Path(__file__).as_posix()).stem)

    unit = ''
    t_rated = None
    if use_mon_csv is True and mon is not None:
        if Battery.CHEM != int(mon.chm[0]):
            print(f"CompareHistSim::WARNING  Battery.CHEM {Battery.CHEM} not equal to transient input chm {int(mon.chm[0])}")
        Battery.CHEM = int(mon.chm[0])
        unit = unit_key.split('_')[-2]
    batt = BatteryMonitor(mod_code=Battery.CHEM)


    # Force Tb.  This is useful for verifying calibration runs where voc(soc) schedule extracted from the run
    # with slightly varying Tb but assumed constant when making new schedule
    if Tb_force is not None:
        if f_raw is not None:
            f_raw.Tb = Tb_force
        if h_raw is not None:
            h_raw.Tb = Tb_force
        if s_raw is not None:
            s_raw.Tb = Tb_force

    # Sort and augment data
    fault = None
    if f_raw is not None:
        f_raw = np.unique(f_raw)
        f_raw = remove_nan(f_raw)
        f_raw = remove_0T(f_raw, 'FAULTS in f_raw')
        if len(f_raw) > 0:
            # noinspection PyTypeChecker
            # Rename
            f_raw = rename_all(f_raw)
            fault = add_stuff_f(f_raw, batt, ib_band=IB_BAND, rated_batt_cap=rated_batt_cap, Dw=dvoc_mon,
                                time_sync=sync_time, ap_ib_diff_slr=Battery_off_dict['ap_ib_diff_slr'],
                                ap_ib_quiet_slr=Battery_off_dict['ap_ib_quiet_slr'])
            print("\nfault after add_stuff_f:\n", fault.dtype.names, fault, "\n")
            fault = filter_Tb(fault, 20., batt, tb_band=100., rated_batt_cap=rated_batt_cap)  # tb_band=100 disables banding
        else:
            fault = None

    # sums and history
    h_combo_raw = hstack2((h_raw, s_raw))
    if h_combo_raw is None or np.atleast_1d(h_combo_raw.time_ux).size < 2:
        return None, None, unit, fault, None, filename, Battery
    else:
        h_combo_raw = np.unique(h_combo_raw)
        h_combo_raw = remove_nan(h_combo_raw)
        h_combo_raw = remove_0T(h_combo_raw, 'HISTORY u and h in h_combo_raw')
        # print("\nhist raw:\n", h_combo_raw.dtype.names, "\n", h_combo_raw, "\n", h_combo_raw.dtype.names, "\n")
        # noinspection PyTypeChecker
        h_combo_raw = rename_all(h_combo_raw)
        hist = add_stuff_f(h_combo_raw, batt, ib_band=IB_BAND, rated_batt_cap=rated_batt_cap, Dw=dvoc_mon,
                           time_sync=sync_time, ap_ib_diff_slr=Battery_off_dict['ap_ib_diff_slr'],
                           ap_ib_quiet_slr=Battery_off_dict['ap_ib_quiet_slr'])

        hist = add_mod(hist, use_mon_csv, mon)
        hist = add_chm(hist, use_mon_csv, mon, Battery.CHEM)
        hist = add_qcrs(hist, mon_t_=use_mon_csv, mon=mon, qcrs=qcrs, t_rated=t_rated, dqdt=batt.chemistry.dqdt)
        hist = add_delta_q(hist)
        print("\nhist after adding stuff:\n", hist.dtype.names, "\n", hist, "\n", hist.dtype.names, "\n :hist after adding stuff\n")
        print("\nhist convert to 20C...:", end='')

        # Convert all the long time readings (history) to same arbitrary (20 deg C) temperature
        hist_20C = filter_Tb(hist, 20., batt, tb_band=TB_BAND, rated_batt_cap=rated_batt_cap)
        print("done")

        # Shift time by detecting when ib changes
        sim = None
        if sync_time is None:
            print("\nShifting time by detecting when ib changes...", end='')
            hist_20C = shift_time(hist_20C)
            print("done.\nShifting time by detecting when ib changes...", end='"')

        # Covert to fast update rate
            print("\nresampling ...", end='')
            h_20C_resamp = resample(data=hist_20C, dt_resamp=dt_resample, time_var='time',
                                specials=[
                                    ('ib_amp_bare_flt', 0),
                                    ('ib_noa_bare_flt', 0),
                                    ('falw', 0),
                                    ('ib_dscn_fa', 0),
                                    ('ib_diff_fa', 0),
                                    ('ib_diff_flt', 0),
                                    ('wv_fa', 0),
                                    ('vc_flt', 0),
                                    ('vc_fa', 0),
                                    ('wrap_lo_fa', 0),
                                    ('wrap_lo_flt', 0),
                                    ('wrap_hi_fa', 0),
                                    ('wrap_hi_flt', 0),
                                    ('ccd_fa', 0),
                                    ('ib_noa_flt', 0),
                                    ('ib_amp_flt', 0),
                                    ('ib_noa_fa', 0),
                                    ('ib_amp_fa', 0),
                                    ('vb_fa_lt', 0),
                                    ('Tb_fa', 0),
                                    ('wrap_lo_m_fa', 0),
                                    ('wrap_lo_m_flt', 0),
                                    ('wrap_hi_m_fa', 0),
                                    ('wrap_hi_m_flt', 0),
                                    ('wrap_lo_n_flt', 0),
                                    ('wrap_lo_n_fa', 0),
                                    ('wrap_hi_n_fa', 0),
                                    ('wrap_hi_n_flt', 0),
                                ])
            h_20C_resamp.dt = h_20C_resamp.time.copy()
            for i in range(len(h_20C_resamp.time)):
                h_20C_resamp.dt[i] = h_20C_resamp.time[i] - h_20C_resamp.time[max(i-1, 0)]

            # Hand fix oddities`
            mon, sim = bandaid(h_20C_resamp)
            mon = rename_all(mon)
            sim = rename_all(sim)

        return mon, sim, unit, fault, hist_20C, filename, Battery


# noinspection PyPep8Naming
def compare_hist_sim(data_file=None, time_end=None, plots=True, use_mon_csv=False, unit_key=None,
                     sync_time=None, dt_resample=10, Tb_force=None, request_history=None, strict_overplot=False,
                     terse=False, fig_list=None, fig_files=None, show_killer_=True, hardcopy=False):

    print(f"\ncompare_hist_sim: \
    \n{data_file=} \
    \n{time_end=} \
    \n{plots=} \
    \n{use_mon_csv=} \
    \n{unit_key=} \
    \n{sync_time=} \
    \n{dt_resample=} \
    \n{Tb_force=} \
    \n{request_history=} \
    \n{strict_overplot=} \
    \n{terse=} \
    \n{fig_files=} \
    \n{fig_list=} \
    \n{show_killer_=} \
    \n{hardcopy=} \
    \n")

    if fig_files is None:
        fig_files = []
    if fig_list is None:
        fig_list = []

    date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    date_ = datetime.now().strftime("%y%m%d")

    # Save these
    scale_batt = 1
    cc_dif_tol = 0.2
    use_mon_soc = True
    # Reconstruction of soc using subsampled data is poor.  Drive everything with soc from Monitor
    dvoc_mon = 0.
    dvoc_sim = 0.
    mon_ver = None
    sim_ver = None
    sim_s_ver = None
    use_sat_mon = True

    # Load history, normalizing all soc and Tb to 20C
    # noinspection PyShadowingNames
    mon_run, sim_run, unit, fault, hist_20C, load_filename, Battery = \
        load_hist_and_prep(data_file=data_file, time_end=time_end, plots=plots, use_mon_csv=use_mon_csv,
                           unit_key=unit_key, sync_time=sync_time, dt_resample=dt_resample, Tb_force=Tb_force)
    sim_s_run = None

    # File path operations
    data_file_txt = PurePosixPath(data_file).name
    version = version_from_data_file(data_file)
    path_to_temp, save_pdf_path, _ = local_paths(version)

    if mon_run is not None and sim_run is not None:
        # Replicate
        data_file_clean = path_to_temp + '/' + data_file_txt.replace('.csv', '_hist' + '.csv', 1)
        mon_file_save = data_file_clean.replace(".csv", "_rep_hist.csv")
        fault_coverage_file_save = data_file_clean.replace(".csv", "_fault_coverage_hist.csv")
        replicateOptions = UserOptions(mon_run=mon_run, sim_run=sim_run, run_type='HistSim', init_time=1.,
                                       verbose=False, max_time=time_end, use_vb_sim=False, scale_batt=scale_batt,
                                       use_mon_soc=use_mon_soc, add_voc_mon=dvoc_mon, add_voc_sim=dvoc_sim,
                                       unit=unit, use_ib_mon=True, request_history=request_history, mod_force=0,
                                       use_sat_mon=use_sat_mon)
        mon_ver, sim_ver, sim_s_ver, mon_r, sim_r, battery = replicate(replicateOptions)
        save_clean_file(mon_ver, mon_file_save, 'mon_rep_hist' + date_)
        save_fault_coverage(mon_run, fault_coverage_file_save, 'fault_coverage_hist' + date_)
        
        # Check if replicate broke early due to skip
        if mon_ver is None:
            print("\nCompareHistSim: Replication broke early due to data skip. Aborting without plots.")
            import tkinter.messagebox
            tkinter.messagebox.showerror(title="Data Integrity Error",
                                         message="CompareHistSim: Replication broke early due to data skip.\n\nAborting without plots.")
            return fig_list, fig_files

    # Plots
    if plots:
        plot_title = load_filename + '   ' + date_time
        filename = str(PurePosixPath(save_pdf_path) / load_filename)

        S = PlotOptions(terse=terse, save_plots=hardcopy)

        if fault is not None and len(fault.time) > 1:
            fig_list, fig_files = over_fault(fault, filename, fig_files=fig_files, plot_title=plot_title,
                                             subtitle='faults', fig_list=fig_list, cc_dif_tol=cc_dif_tol,
                                             time_units='sec', save_plots=S.save_plots)

        if hist_20C is not None and len(hist_20C.time) > 1:
            sim_run = None
            mon_tst = None
            sim_tst = None
            sim_s_tst = None
            if not S.terse:
                fig_list, fig_files = overall_fault(mon_run, mon_tst, sim_run, sim_tst, sim_s_run, sim_s_tst, filename,
                                                    fig_files, plot_title=plot_title, fig_list=fig_list,
                                                    run_type='HistSim', save_plots=S.save_plots)

            fig_list, fig_files = dom_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_run, sim_s_ver, filename, fig_files,
                                           plot_title=plot_title, fig_list=fig_list, strict_overplot=strict_overplot,
                                           terse=S.terse, run_type='HistSim', save_plots=S.save_plots)

        print('showing plots...')
        plt.ion()
        plt.show(block=False)

        # Copies — batch/AUTO mode only; show_killer's do_hardcopy handles the interactive case
        if S.save_plots and not show_killer_:
            import threading
            def _assemble(base=filename, path=save_pdf_path, dt=date_time):
                try:
                    precleanup_fig_files(output_pdf_name=base, path_to_pdfs=path)
                    print('\ncreating pdf...')
                    pngs_to_pdf(png_folder=path, output_pdf=base + '_' + dt + '.pdf')
                except Exception as e:
                    print(f"pdf assembly ERROR: {e}")
            threading.Thread(target=_assemble, daemon=True).start()

        if not fig_list:
            string = 'none plots kill'
        else:
            string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
        if show_killer_:
            show_killer(string, 'CompareFault', fig_list=fig_list, fig_files=fig_files, pdf_path=save_pdf_path, pdf_base=filename, hardcopy=hardcopy)
        cleanup_fig_files(fig_files)
    print('DONE')

    return fig_list, fig_files


# noinspection PyUnusedLocal,PyPep8Naming
def main():  # Sample usage. OK on 20260217

    import sys
    if sys.platform == 'linux':
        gdrive = '/home/daveg/gdrive/'
    else:
        gdrive = 'G:/My Drive/'

    # User inputs (multiple input_files allowed
    # Cut-pasted from GUI_TestSOC Run window
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction/g20250612a/truckHist_20260302.csv'

    data_file = '/home/daveg/.local/SOC_Particle/plink/dataReduction/g20250612a/tLoFailModel_soc3p2_hi_lo_bb.csv'
    time_end = None
    plots = False
    use_mon_csv = True
    unit_key = 'g20250612a_soc3p2_hi_lo_bb'
    sync_time = None
    dt_resample = 10
    Tb_force = None
    request_history = 5
    strict_overplot = True
    terse = True
    fig_files = None
    fig_list = None
    show_killer_ = True
    hardcopy = True

    compare_hist_sim(data_file=data_file, use_mon_csv=use_mon_csv, unit_key=unit_key, dt_resample=dt_resample,
                     plots=plots, Tb_force=Tb_force, request_history=request_history,
                     strict_overplot=strict_overplot, terse=terse, hardcopy=hardcopy)


if __name__ == '__main__':  # Example usage.  Ran ok 20260217
    main()
