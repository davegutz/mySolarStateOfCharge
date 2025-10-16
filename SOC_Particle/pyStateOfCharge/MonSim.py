# MonSim:  Monitor and Simulator replication of Particle Photon Application
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

""" Python model of what's installed on the Particle Photon.  Includes
a monitor object (MON) and a simulation object (SIM).   The monitor is
the EKF and Coulomb Counter.   The SIM is a battery model, that also has a
Coulomb Counter built in."""

from Battery import BatteryMonitor, BatterySim, is_sat, Retained
from Battery import overall_batt
from TFDelay import TFDelay
from MonSimNomConfig import *  # Global config parameters.   Overwrite in your own calls for studies
from numpy import random as random
from MonSimPrint import *
from MonSimClasses import *
from dataclasses import dataclass
from typing import Optional

def battery_size(mo, so, scale_in_, unit_cap_rated_):
    if hasattr(mo, 'qcrs'):
        scale_mon_ = mo.qcrs[0] / (Battery.UNIT_CAP_RATED*3600)
    else:
        scale_mon_ = unit_cap_rated / Battery.UNIT_CAP_RATED
        if scale_in_:
            scale_mon_ *= scale_in
    if so is not None and hasattr(so, 'qcrs_s'):
        scale_sim_ = so.qcrs_s[0] / (Battery.UNIT_CAP_RATED*3600)
    else:
        scale_sim_ = unit_cap_rated / Battery.UNIT_CAP_RATED
        if scale_in_:
            scale_sim_ *= scale_in_
    return scale_mon_, scale_sim_


def chm_from_mon_or_sim(mo, so):
    chem_m = mo.chm
    if so is not None:
        chem_s = so.chm_s
    else:
        chem_s = chm_m
    return chem_m, chem_s

def get_modeling(mo):
    if hasattr(mo, 'mod_data'):
        modeling_ = mo.mod_data
    else:
        modeling_ = 255 * np.ones(len(mo.time))
    return modeling_

def sync_to_mon_or_sim(mo, so, t_mx=None):
    if so is not None and len(so.time) < len(mo.time):
        time = so.time
    else:
        time = mo.time
    if t_mx is not None:
        t_delt = time - time[0]
        time = time[np.where(t_delt <= t_mx)]
    return time

def vb_from_raw_or_selected(use_raw, mo):
    if use_raw:
        vb_ = mo.vb_h
    else:
        if hasattr(mo, 'vb_f'):
            vb_ = mo.vb_f
        else:
            vb_ = mo.vb
    return vb_

@dataclass
class UserOptions:
    mon_ref: 'DataOverModel.SavedData'  # Mandatory reference data to be replicated
    run_type: Optional[str] = None  # Either "RunSim" or "HistSim" depending on caller
    sim_ref: Optional['DataOverModel.SavedDataSim'] = None  # Embedded model data
    unit: Optional[str] = None  # Name of the battery instance derived from 'HDWE_UNIT' of configuration include .h file
    Bsim: Optional[int] = None  # sim model code BB=0 (Battleborn), CH=1 (Chins), CHG=2 (Chins in Garage)
    Bmon: Optional[int] = None  # mon model code BB=0 (Battleborn), CH=1 (Chins), CHG=2 (Chins in Garage)
    init_time: Optional[float] = -4.  # The process tries to determine mon_ref.init_time when data is loaded by finding
    # when Ib changes. This input helps out to over-ride those results when they don't work as desired. It shouldn't
    # be needed often.
    max_time: Optional[float] = None  # Limit the simultation run, s

    # Model scalar / adders
    scale_in: Optional[float] = None  # Battery size scalar applied to the nominal battery unit of 100 A-h
    slr_cap_chg: Optional[float] = 1.  # Scalar on ideal capacitor model for hysteresis charging model only
    slr_cap_dis: Optional[float] = 1.  # Scalar on ideal capacitor model for hysteresis discharging model only
    slr_coul_eff: Optional[float] = 1.  # Scalar on Coulombic Efficiency of battery model, both for the BatterySim model
    # and the BatteryMonitor Coulomb counter
    slr_cutback_gain: Optional[float] = 1.  # Scalar on the automatic BatterySim model of saturation effects
    slr_hys_cap_sim: Optional[float] = 1.  # Scalar on the battery size effect on hysteresis
    slr_hys_chg: Optional[float] = 1.  # Direct scalar on the magnitude of hysteresis during charging
    slr_hys_dis: Optional[float] = 1.  # Direct scalar on the magnitude of hysteresis during charging
    slr_hys_mon: Optional[float] = 1.  # Overall scalar on the magnitude of hysteresis in BatteryMonitor
    slr_hys_sim: Optional[float] = 1.  # Overall scalar on the magnitude of hysteresis in BatterySim
    slr_res_0: Optional[float] = 1.  # Scalar on Randles static resistance model
    slr_res_ct: Optional[float] = 1.  # Scalar on Randles charge transfer function resistance
    slr_r_ss: Optional[float] = 1.  # Scalar on equivalent battery resistance state-space charge transfer
    # TODO: when is ss used versus ct
    slr_tauct_sim: Optional[float] = 1.  # Scalar on Randles charge transfer function time constant in ModelSim
    add_s_voc_soc: Optional[float] = 0.  # Adder to SOC input of voc_soc table lookup of voc from soc
    add_voc_sim: Optional[float] = 0.  # Adder to BatterySim voc table outputs (should match dvoc of Chemistry_BMS.cpp)
    add_voc_mon: Optional[float] = 0.  # Adder to BatteryMonitor voc table outputs (should match dvoc of
    # Chemistry_BMS.cpp)
    add_Tb_in: Optional[float] = None  # Adder on sensed Tb, deg C

    # Failure injection
    ib_fail_t: Optional[float] = None  # Time to inject a failure into the Ib input signal
    ib_fail: Optional[float] = 0.  # The fixed Ib value to fail to, A
    vb_fail_t: Optional[float] = None  # Time to inject a failure into the Vb input signal
    vb_fail: Optional[float] = 13.2  # The fixed Vb value to fail to, V

    # Configuration changes
    eframe_mult: Optional[int] = Battery.cp_eframe_mult

    stauct_mon: Optional[float] = 1.
    use_vb_sim: Optional[bool] = False
    request_history: Optional[int] = None  # Print simulation history (0 - 5) to check overplot using data in addition
    use_ib_mon: Optional[bool] = False  # Drive BatterySim directly with the BatteryMonitor input, useful when raw sim data not available
    use_mon_soc: Optional[bool] = False  # Drive SOC of the model directly with data to focus on modeling that is downstream of SOC
    use_vb_raw: Optional[bool] = False  # Force usage of raw Vb bypassing the signal selection logic
    verbose: Optional[bool] = True  # Lots of 'helpful' information used to provide some quick clues about whatever
    # to or instead of plots

#  Replicate the application in its entirety here.
#  There are no 'bank' parameters anywhere in this model.   It is assumed that all inputs from the application have
#  been converted to the single battery unit 12v form, S1P1, lower-case nomenclature.
def replicate(OPT: UserOptions):

    """TODO:
    1. *** Current sense class. Done
    2. *** Fig. 9 EKF 2a:  dt_eframe at 0 ****->i_ekf = -1 --> i_ekf=max(i_ekf, 0) in MonSimPrint.py
    2. *** Fig. 9 EKF 2a:  dt_eframe at i_ekf=0 is = 4.882 while ref is 5.255. *** dt_eframe[0] = dt_ekf[0]
    3. *** Fig. 5 Dom 4a:  TB ver needs past value. *** Plotted wrong thing.  Tb-->Tb_rap and Tb_rap_ver
    4. *** Fig. 7 EKF 1:  Bu_ver at 0.  Fixed by dt_eframe fix
    5. *** Fig. 2 Dom 2: dv_hys plots not intelligible.  Not sure what these mean
    6. *** Fig. 1 1a: e_wrap filter initialization.   Need filt_rates from Noa and Amp filters.  LoopIbNoa and Amp e_wrap_rate().  Fixed by incorporating the scale logic from app
    7. Fig. 9 EKF 2a: hx(soc) negative slope?  This needs to be run just below saturation
    8. *** Fig. 10 EKF 3:  voc_ekv (hx) not equal at 0.  Fixed by modifying reset_ekf logic
    9. Run CompareHistSim etc.
    10. *** Fig 18 dv_hys jitter around 0 in reference data  ***->0*dv_hys before return instead of in return Fixed by sending dv_hys over data stream instead of calculating in load.
    11. *** Fig. 23 voltage resolutions off/on mon 1.  Resolutions of data corrected.
    12. *** Can reorder execution so header printed before any relevant data?  Deleting first two rows of data before headers causes data mismatch in sim.  No change.  Not problem
    13. *** Discarded sim data ('vv0') causes issue.  Don't record/save local entered data.  Fixed by 'skip' logic
    14. *** Fig. 6 Ult 1:  e_wrap_m_filt_ver off (  e_wrap_m_filt = 1.5 ?).  Fixed by multitude of other fixes.
    15. *** Fig. 7 EKF 1: S initialization.  Fixed by putting S calc inside reset
    16. *** Fig 10  EKF 3:  z=voc_stat_f and ver not equal.  Fixed by updating variable names
    17. *** Fig 12 Hyst 1: e_wrap_ver not equal.  Fixed by adding scale logic from app
    18. *** Fig 13 sim_s 1:  ib_in_ver  Not fixed: OK because have manually over-ridden ib selection to force ib_noa use in logic but not selection
    19. Fig 15 sim_s 2a:  vb?   Keep looking for this when run at other op conditions.  Shutdown problem.
    20. **** Fig. many:  delta_q_s_ver != delta_q_s   Fixed by changing Sen->T to t_ in Sim::count_coulombs
    21. **** _s values in print are off.  Where those there 9/29?  Yes.  Continue to debug...delays in ib_s model BatterySim
    22. **** Fig. 21 GP 3 Tune (3,3,6):  vb?  OK.  The battery is near saturation and the voc(soc) curve is slightly innacurate
    23. skip_* being set properly?
    24. HistSim:  dv_dyn_ver looks wrong.
    """

    # time
    t = sync_to_mon_or_sim(OPT.mon_ref, OPT.sim_ref, t_mx=OPT.max_time)

    # vb
    vb = vb_from_raw_or_selected(OPT.use_vb_raw, OPT.mon_ref)

    # chem
    chm_m, chm_s = chm_from_mon_or_sim(OPT.mon_ref, OPT.sim_ref)

    t_len = len(t)
    rp = Retained()

    # modeling
    modeling = get_modeling(OPT.mon_ref)

    # tweaking
    tweak_test = rp.tweak_test()

    SN = Sensors(mon_ref=OPT.mon_ref, sim_ref=OPT.sim_ref, add_Tb_in=OPT.add_Tb_in, run_type=OPT.run_type)

    # Battery sizing
    scale_mon, scale_sim = battery_size(OPT.mon_ref, OPT.sim_ref, OPT.scale_in, unit_cap_rated)

    # Make batteries
    sim = BatterySim(SN=SN, mod_code=chm_s[0], tb_f=SN.Tb0_s, scale=scale_sim, tweak_test=tweak_test,
                     dv_hys=OPT.mon_ref.dv_hys[0], slr_res_0=OPT.slr_res_0, slr_res_ct=OPT.slr_res_ct, stauct=OPT.slr_tauct_sim, slr_r_ss=OPT.slr_r_ss,
                     s_hys=OPT.slr_hys_sim, dvoc=OPT.add_voc_sim, scale_hys_cap=OPT.slr_hys_cap_sim, slr_coul_eff=OPT.slr_coul_eff,
                     slr_cap_chg=OPT.slr_cap_chg, slr_cap_dis=OPT.slr_cap_dis, slr_hys_chg=OPT.slr_hys_chg, slr_hys_dis=OPT.slr_hys_dis,
                     slr_cutback_gain=OPT.slr_cutback_gain, add_s_voc_soc=OPT.add_s_voc_soc, unit=OPT.unit, mon_ref=OPT.mon_ref,
                     sim_ref=OPT.sim_ref)
    mon = BatteryMonitor(SN=SN, mod_code=chm_m[0], tb_f=SN.Tb0, scale=scale_mon, tweak_test=tweak_test,
                         slr_res_0=OPT.slr_res_0, slr_res_ct=OPT.slr_res_ct, stauct=OPT.stauct_mon,
                         slr_r_ss=OPT.slr_r_ss, s_hys=OPT.slr_hys_mon, dvoc=OPT.add_voc_mon, eframe_mult=OPT.eframe_mult,
                         slr_coul_eff=OPT.slr_coul_eff, unit=OPT.unit, ref=OPT.mon_ref, dTb=SN.dTb, run_type=OPT.run_type)
    Is_sat_delay = TFDelay(in_=OPT.mon_ref.soc[0] > 0.97, t_true=T_SAT, t_false=T_DESAT, dt=0.1)  # later, dt is changed

    # Time sync
    if hasattr(OPT.mon_ref, 'time_ref'):
        mon.saved.time_ref = OPT.mon_ref.time_ref
        sim.saved_s.time_ref = OPT.mon_ref.time_ref
    else:
        mon.saved.time_ref = 0.
        sim.saved_s.time_ref = 0.

    # time loop initialization
    now = t[0]
    reset_ekf = True
    i_ekf = -1
    i_temp = -1
    T = OPT.mon_ref.dt[0]
    i = None
    hdr = None
    sat_s_init = None
    e_wrap_trim_amp_init = None
    e_wrap_trim_noa_init = None

    # Print debug information
    if OPT.request_history is not None and OPT.request_history > 0:
        hdr = print_hist(OPT.request_history, OPT.run_type, 0, i_temp, i_ekf, t, OPT.mon_ref, mon, True, True,
                         SN.Tb, SN.Tb_past, OPT.sim_ref, sim, SN)

    # Top of time loop
    for i in range(t_len):

        if i >= 206:
            pass  # used for debug breakpoint at i >= <val>

        # Time
        now = t[i]
        T_ekf = None
        if i != 0:
            candidate_dt = t[i] - t[i-1]  # update
            if candidate_dt > 1e-6:
                T = candidate_dt

        # Get temperature data
        if hasattr(OPT.mon_ref, 'time_t'):
            calc_temp = (i_temp+1 < len(OPT.mon_ref.time_t)) and (OPT.mon_ref.time_t[i_temp+1] <= OPT.mon_ref.time[i])
        else:
            calc_temp = True
        if calc_temp:
            i_temp += 1
            mon.Tb = mon.Tb_hdwe  # past value
            mon.reset_temp = (i_temp < 2)  # make sure temp init is longer than reset
            if hasattr(OPT.mon_ref, 'Tt'):
                mon.dt_temp = OPT.mon_ref.Tt[i_temp]
            else:
                mon.dt_temp = mon.dt
            if hasattr(OPT.mon_ref, 'Tb_hdwe'):
                mon.Tb_hdwe = OPT.mon_ref.Tb_hdwe[i_temp]
            else:
                mon.Tb_hdwe = OPT.mon_ref.Tb_f[i_temp]
            if OPT.run_type == 'RunSim':
                sim.Tb = OPT.mon_ref.Tb[i_temp]
                mon.Tb = OPT.mon_ref.Tb[i_temp]
                mon.Tb_s = OPT.mon_ref.Tb[i_temp]
            else:
                sim.Tb = OPT.mon_ref.Tb_f[i_temp]
                mon.Tb = OPT.mon_ref.Tb_f[i_temp]
                mon.Tb_s = OPT.mon_ref.Tb_f[i_temp]
            if i_temp > 0:
                SN.update_tb()
                mon.Tb_rap = SN.Tb_past
                mon.Tb_f_rap = SN.Tb_f_past
                mon.Tb_f_rate_rap = SN.Tb_f_rate_past
            if hasattr(OPT.mon_ref, 'Tb_mod'):
                sim.Tb_f = OPT.mon_ref.Tb_mod[i_temp]
            else:
                sim.Tb_f = sim.Tb
            monx, simx = SN.calc_tb(mon, sim, i_temp, OPT)

        # Input
        dc_dc_on = False
        rp.modeling = modeling[i]

        # Basic reset model verification is to init to the input data
        # Tried hard not to re-implement solvers in the Python verification  tool
        # Also, BTW, did not implement signal selection or tweak logic
        reset = bool((t[i] <= OPT.init_time) or (t[i] < 0. and t[0] > OPT.init_time))
        if OPT.mon_ref.res is not None:
            reset = reset or bool(OPT.mon_ref.res[i] > 0.)
        prn_soc_debug(time=now, leader="before sim init:     ", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)

        if reset:
            sim.apply_soc(OPT.mon_ref.soc_s[i], SN.Tb_f_past)  # calculates delta_q
            sim.load(sim.delta_q)
            sim.assign_tb(sim.Tb)
            sim.assign_tb_f(sim.Tb_f)
            sim.apply_delta_q_t(sim.delta_q, SN.Tb_f_past)
            sat_s_init = SN.voc_stat_init > OPT.mon_ref.vsat[0]
            if OPT.sim_ref is not None:
                sat_s_init = OPT.sim_ref.sat_s[0]
            sim.sat = sat_s_init
            mon.sat = OPT.mon_ref.sat[0]

        if calc_temp:
            prn_soc_debug(time=now, leader="b temp filtr:    ", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)

            mon = SN.temp_calc(OPT.mon_ref, mon, Battery, i_temp)

            prn_soc_debug(time=now, leader="a temp filtr:    ", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)

        # Models
        if rp.modeling == 0:
            SN.update_ib_vb(i)

        if OPT.sim_ref is not None and not OPT.use_ib_mon:
            ib_in_s = OPT.sim_ref.ib_in_s[i]
        else:
            if OPT.run_type == 'RunSim':
                ib_in_s = OPT.mon_ref.ib[i]
            else:
                ib_in_s = OPT.mon_ref.ib_f[i]

        if OPT.Bsim is None:
            _chm_s = chm_s[i]
        else:
            _chm_s = OPT.Bsim

        prn_soc_debug(time=now, leader="befor sim.calculater:    ", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)
        sim.calculate(_chm_s, None, ib_in_s, SN.dt_s[i], reset, None, None, SN,
                      soc=sim.soc, q_capacity=sim.q_capacity, dc_dc_on=dc_dc_on, rp=rp, sat_init=sat_s_init,
                      bms_off_init=OPT.sim_ref.bms_off_s[0])
        prn_soc_debug(time=now, leader="after sim.calculater:    ", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)
        sim.count_coulombs(chem=_chm_s, dt=SN.dt_s[i], reset_temp=reset, tb_f=sim.Tb_f, tb_f_rate=SN.Tb_f_rate_past,
                           charge_curr=sim.ib_charge, sat=False, soc_s_init=SN.soc_s[i], mon_sat=mon.sat,
                           sim_delta_q=SN.dq_s[i], use_soc_in=OPT.use_mon_soc, soc_in=SN.soc_s[i])
        prn_soc_debug(time=now, leader="after sim.count_cou:    ", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)

        # EKF
        if reset:
            mon.apply_delta_q_t(SN.delta_q[i], SN.Tb_f_rap[i])
            rp.delta_q = mon.delta_q
            mon.load(rp.delta_q)

        prn_soc_debug(time=now, leader="after mon_soc_apply  ", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)

        # Chemistry
        if OPT.Bmon is None:
            _chm_m = chm_m[i]
        else:
            _chm_m = OPT.Bmon

        if OPT.ib_fail_t and t[i] > OPT.ib_fail_t:
            ib_ = OPT.ib_fail
        else:
            if OPT.mon_ref.ib_sel is not None:
                ib_ = OPT.mon_ref.ib_sel[i]
            else:
                ib_ = OPT.mon_ref.ib[i]

        if OPT.use_vb_sim:
            vb_ = sim.vb
        elif OPT.vb_fail_t and t[i] >= OPT.vb_fail_t:
            vb_ = OPT.vb_fail
        else:
            vb_ = vb[i]

        # Monitor EKF sequencing logic
        if (i_ekf+1 < len(OPT.mon_ref.time_e)) and (OPT.mon_ref.time_e[i_ekf+1] <= OPT.mon_ref.time[i]):
            i_ekf += 1
            reset_ekf = i_ekf == 0
            if i_ekf < 1:
                T_ekf = OPT.mon_ref.dt_ekf[i_ekf]
            else:
                T_ekf = OPT.mon_ref.time_e[i_ekf] - OPT.mon_ref.time_e[i_ekf-1]  # update
            calc_ekf = True
        else:
            calc_ekf = False
        SN.update_ekf(i_ekf)

        if reset_ekf and calc_ekf:
            mon.init_soc_ekf(OPT.mon_ref, i, i_ekf, run_type=OPT.run_type)  # when modeling (assumed in python) ekf wants to equal model

        # Monitor calculate
        if i == 2:
            pass
        if rp.modeling == 0:
            mon.calculate(_chm_m, vb_, ib_, T, reset, calc_ekf, T_ekf, SN,
                          rp=rp, bms_off_init=OPT.mon_ref.bms_off[0], ib_amp=OPT.mon_ref.ibmh[i], ib_noa=OPT.mon_ref.ibnh[i],
                          reset_ekf=reset_ekf)
        else:
            mon.calculate(_chm_m, vb_ + random.randn() * v_std + dv_sense, ib_ + random.randn() * i_std + di_sense, T,
                          reset, calc_ekf, T_ekf, SN,
                          rp=rp, bms_off_init=OPT.mon_ref.bms_off[0], ib_amp=SN.ibmm[i], ib_noa=SN.ibnm[i],
                          reset_ekf=reset_ekf)
        ib_charge = mon.ib_charge
        sat = is_sat(SN.Tb_f_past, mon.chemistry.rated_temp, mon.voc_filt, mon.soc, mon.chemistry.nom_vsat,
                     mon.chemistry.dvoc_dt, mon.chemistry.low_t)
        saturated = Is_sat_delay.calculate(sat, T_SAT, T_DESAT, min(T, T_SAT / 2.), reset)

        # Monitor count Coulumbs
        if rp.modeling == 0:
            mon.count_coulombs(chem=_chm_m, dt=T, reset=reset, tb_f=SN.Tb_f_past, charge_curr=ib_charge,
                               sat=saturated, use_soc_in=OPT.use_mon_soc, soc_in=OPT.mon_ref.soc[i])
        else:
            mon.count_coulombs(chem=_chm_m, dt=T, reset=reset, tb_f=SN.Tb_f_past, charge_curr=ib_charge,
                               sat=saturated, use_soc_in=OPT.use_mon_soc, soc_in=OPT.mon_ref.soc[i])
        mon.calc_charge_time(mon.q, mon.q_capacity, ib_charge, mon.soc)
        mon.assign_soc_s(sim.soc)

        # Break if data integrity questionable
        if SN.skip_e[i_ekf] or SN.skip_t[i_temp] or SN.skip_sel[i] or SN.skip_rap[i] or SN.skip_s[i]:
            break

        # Save plot info
        mon.save(t[i], T, mon.soc, sim.voc)
        sim.save(t[i], T)
        sim.save_s(t[i])

        # Print initial
        if i == 0 and OPT.verbose:
            print('time=', t[i])
            print('mon:  ', str(mon))
            print('time=', t[i])
            print('sim:  ', str(sim))

        # History print
        if OPT.request_history is not None and OPT.request_history > 0:
            hdr = print_hist(OPT.request_history, OPT.run_type, i, i_temp, i_ekf, t, OPT.mon_ref, mon, calc_temp, calc_ekf,
                             SN.Tb, SN.Tb_past, OPT.sim_ref, sim, SN)

        prn_soc_debug(time=now, leader="end loop:", i=i, i_temp=i_temp, mon_old=OPT.mon_ref, mon=mon)

        # pick a pass to run debugger to a time
        if i >= 206:
            pass  # used for debug breakpoint at i >= <val>
        if now > 2:
            pass  # used for debug breakpoint at now > <val>
        else:
            pass

        # Finish loop
        if calc_ekf:
            reset_ekf = False

    # Final hdr print
    if OPT.request_history is not None and OPT.request_history > 0:
        print(hdr)
    if SN.skip_e[i_ekf] or SN.skip_t[i_temp] or SN.skip_sel[i] or SN.skip_rap[i] or SN.skip_s[i]:
        print(f"\n\n************** Data integrity degraded by skip.  A digit could have been inserted anywhere in data.  Break.")
        print("   now {:5.3f}".format(now),
              "   time_end {:5.3f}\n\n".format(t[-1]),
              )

    # Data
    if OPT.verbose:
        print('   time mo.chm so.chm so.ib_in_s so.dv_hys  mo.ib mo.soc mo.dv_hys   smv.ib_in_s sim.ibs sim.ioc sim.sat sim.dis sim.dv_dot smv.dv_hys  mv.ib  mv.soc mon.ibs  mon.ioc   mon.sat   mon.dis    mon.dv_dot  mv.dv_hys')
        print('time=', now)
        print('mon:  ', str(mon))
        print('sim:  ', str(sim))

    return mon.saved, sim.saved, sim.saved_s, mon, sim


if __name__ == '__main__':
    import sys
    from DataOverModel import SavedData, SavedDataSim, write_clean_file
    from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files
    import matplotlib.pyplot as plt
    if sys.platform == 'darwin':
        import matplotlib
        matplotlib.use('tkagg')
    plt.rcParams['axes.grid'] = True

    def main():
        date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        date_ = datetime.now().strftime("%y%m%d")

        # Transient  inputs
        time_end = None
        # time_end = 35200
        ib_fail_t = None
        init_time_in = None
        use_ib_mon_in = False
        scale_in = None
        use_vb_raw = False
        # unit_key = None
        # data_file_old_txt = None
        scale_r_ss_in = 1.
        scale_hys_sim_in = 1.
        dvoc_sim_in = 0.
        dvoc_mon_in = 0.
        Bmon_in = None
        Bsim_in = None
        skip = 1
        zero_zero_in = False
        zero_thr_in = 0.02
        data_file_old_txt = 'EKF_Track Dr2000 v20220917.txt'; unit_key = 'pro_2022'
        hdr_key = "unit,"  # Find one instance of title
        hdr_key_sel = "unit_s,"  # Find one instance of title
        unit_key_sel = "unit_sel"
        hdr_key_sim = "unit_m,"  # Find one instance of title
        unit_key_sim = "unit_sim"
        save_pdf_path = '../dataReduction/figures'
        path_to_temp = '../dataReduction/temp'
        import os
        if not os.path.isdir(path_to_temp):
            os.mkdir(path_to_temp)

        # Load mon v4 (old)
        data_file_clean = write_clean_file(data_file_old_txt, type_='_mon', hdr_key=hdr_key, unit_key=unit_key,
                                           skip=skip)
        mon_old_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)

        # Load sel (old)
        sel_file_clean = write_clean_file(data_file_old_txt, type_='_sel', hdr_key=hdr_key_sel,
                                          unit_key=unit_key_sel, skip=skip)
        sel_old_raw = None
        if sel_file_clean:
            sel_old_raw = np.genfromtxt(sel_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
        mon_old = SavedData(rap=mon_old_raw, sel=sel_old_raw, time_end=time_end, zero_zero=zero_zero_in,
                            zero_thr=zero_thr_in, init_time_in=init_time_in)
        # SavedData determines when to initialize
        init_time = mon_old.init_time

        # Load _m v24 portion of real-time run (old)
        data_file_sim_clean = write_clean_file(data_file_old_txt, type_='_sim', hdr_key=hdr_key_sim,
                                               unit_key=unit_key_sim, skip=skip)
        if data_file_sim_clean:
            sim_old_raw = np.genfromtxt(data_file_sim_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
            sim_old = SavedDataSim(time_ref=mon_old.time_ref, data=sim_old_raw, time_end=time_end)
        else:
            sim_old = None

        # New run
        mon_file_save = data_file_clean.replace(".csv", "_rep.csv")

        replicateOptions = UserOptions(mon_ref=mon_old, sim_ref=sim_old, Bmon=Bmon_in, Bsim=Bsim_in,
                                       init_time=mon_old.init_time, use_ib_mon=use_ib_mon_in,
                                       use_mon_soc=use_mon_soc_in, use_vb_raw=use_vb_raw, add_voc_sim=dvoc_sim_in,
                                       add_voc_mon=dvoc_mon_in, use_vb_sim=use_vb_sim_in,
                                       add_s_voc_soc=add_s_voc_soc_in, verbose=verbose, slr_r_ss=scale_r_ss_in,
                                       scale_in=scale_in, slr_hys_sim=s_hys_sim_in, request_history=request_history,
                                       ib_fail_t=ib_fail_t)
        mon_ver, sim_ver, sim_s_ver, mon, sim = replicate(replicateOptions)
        save_clean_file(mon_ver, mon_file_save, 'mon_rep' + date_)

        # Plots
        fig_list = []
        fig_files = []
        data_root = data_file_clean.split('/')[-1].replace('.csv', '-')
        filename = data_root + os.path.split(__file__)[1].split('.')[0]
        plot_title = filename + '   ' + date_time
        fig_list, fig_files = overall_batt(mon_ver, sim_ver, filename, fig_files, plot_title=plot_title,
                                           fig_list=fig_list, suffix='_ver')  # sim over mon verify
        unite_pictures_into_pdf(outputPdfName=filename+'_'+date_time+'.pdf', save_pdf_path=save_pdf_path)
        cleanup_fig_files(fig_files)

        plt.show()

    main()
