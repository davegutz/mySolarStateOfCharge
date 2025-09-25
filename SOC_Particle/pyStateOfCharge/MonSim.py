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
from Scale import Scale
from MonSimPrint import *
from MonSimClasses import *

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
        print(f"what do we do now?  {rp.modeling=}")
        # exit(1)
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
        vb_ = mo.vb
    return vb_

#  Replicate the application in its entirety here.
#  There are no 'bank' parameters anywhere in this model.   It is assumed that all inputs from the application have
#  been converted to the single battery unit 12v form, S1P1, lower-case nomenclature.
def replicate(mon_old, sim_old=None, init_time=-4., t_vb_fail=None, vb_fail=13.2,
              t_ib_fail=None, ib_fail=0., use_ib_mon=False, scale_in=None, Bsim=None, Bmon=None, use_vb_raw=False,
              scale_r_ss=1., s_hys_sim=1., s_hys_mon=1., dvoc_sim=0., dvoc_mon=0., dTb_in=None,
              verbose=True, t_max=None, eframe_mult=Battery.cp_eframe_mult, sres0=1., sresct=1., stauct_sim=1.,
              stauct_mon=1, use_vb_sim=False, scale_hys_cap_sim=1., s_cap_chg=1., s_cap_dis=1.,
              s_hys_chg=1., s_hys_dis=1., s_coul_eff=1., use_mon_soc=False, cutback_gain_sclr=1., ds_voc_soc=0.,
              unit=None, request_history=None):

    """TODO:
    1. Current sense class
    2. *** Fig. 9 EKF 2a:  dt_eframe at 0 ****->i_ekf = -1 --> i_ekf=max(i_ekf, 0) in MonSimPrint.py
    2. *** Fig. 9 EKF 2a:  dt_eframe at i_ekf=0 is = 4.882 while ref is 5.255. *** dt_eframe[0] = dt_ekf[0]
    3. *** Fig. 5 Dom 4a:  TB ver needs past value. *** Plotted wrong thing.  Tb-->Tb_rap and Tb_rap_ver
    4. *** Fig. 7 EKF 1:  Bu_ver at 0.  Fixed by dt_eframe fix
    5. *** Fig. 2 Dom 2: dv_hys plots not intelligible.  Not sure what these mean
    6. Fig. 1 1a: e_wrap filter initialization.   Need filt_rates from Noa and Amp filters.  LoopIbNoa and Amp e_wrap_rate()
    7. Fig. 9 EKF 2a: hx(soc) negative slope?
    8. *** Fig. 10 EKF 3:  voc_ekv (hx) not equal at 0.  Fixed by modifying reset_ekf logic
    9. Run CompareHistSim etc.
    10. *** Fig 18 dv_hys jitter around 0 in reference data  ***->0*dv_hys before return instead of in return Fixed by sending dv_hys over data stream instead of calculating in load.
    11. voltage resolutions off/on mon 1
    12. *** Can reorder execution so header printed before any relevant data?  Deleting first two rows of data before headers causes data mismatch in sim.  No change.  Not problem
    13. *** Discarded sim data ('vv0') causes issue.  Don't record/save local entered data.  Fixed by 'skip' logic
    14. Fig. 6 Ult 1:  e_wrap_m_filt_ver off (  e_wrap_m_filt = 1.5 ?)
    15. *** Fig. 7 EKF 1: S initialization.  Fixed by putting S calc inside reset
    16. *** Fig 10  EKF 3:  z=voc_stat_f and ver not equal.  Fixed by updating variable names
    17. Fig 12 Hyst 1: e_wrap_ver not equal.
    18. *** Fig 13 sim_s 1:  ib_in_ver  Not fixed: OK because have manually over-ridden ib selection to force ib_noa use in logic but not selection
    19. Fig 15 sim_s 2a:  vb?
    """

    # time
    t = sync_to_mon_or_sim(mon_old, sim_old, t_mx=t_max)

    # vb
    vb = vb_from_raw_or_selected(use_vb_raw, mon_old)

    # chem
    chm_m, chm_s = chm_from_mon_or_sim(mon_old, sim_old)

    t_len = len(t)
    rp = Retained()

    # modeling
    modeling = get_modeling(mon_old)

    # tweaking
    tweak_test = rp.tweak_test()

    ST = TbSense(mon_ref=mon_old, dTb_in=dTb_in)

    # Battery sizing
    scale_mon, scale_sim = battery_size(mon_old, sim_old, scale_in, unit_cap_rated)

    # Make batteries
    sim = BatterySim(mod_code=chm_s[0], tb_f=ST.Tb0_s, scale=scale_sim, tweak_test=tweak_test,
                     dv_hys=mon_old.dv_hys[0], sres0=sres0, sresct=sresct, stauct=stauct_sim, scale_r_ss=scale_r_ss,
                     s_hys=s_hys_sim, dvoc=dvoc_sim, scale_hys_cap=scale_hys_cap_sim, s_coul_eff=s_coul_eff,
                     s_cap_chg=s_cap_chg, s_cap_dis=s_cap_dis, s_hys_chg=s_hys_chg, s_hys_dis=s_hys_dis,
                     cutback_gain_sclr=cutback_gain_sclr, ds_voc_soc=ds_voc_soc, unit=unit, mon_ref=mon_old,
                     sim_ref=sim_old)
    mon = BatteryMonitor(mod_code=chm_m[0], tb_f=ST.Tb0, scale=scale_mon, tweak_test=tweak_test,
                         sres0=sres0, sresct=sresct, stauct=stauct_mon,
                         scale_r_ss=scale_r_ss, s_hys=s_hys_mon, dvoc=dvoc_mon, eframe_mult=eframe_mult,
                         s_coul_eff=s_coul_eff, unit=unit, ref=mon_old, dTb=ST.dTb)
    Is_sat_delay = TFDelay(in_=mon_old.soc[0] > 0.97, t_true=T_SAT, t_false=T_DESAT, dt=0.1)  # later, dt is changed

    # Time sync
    mon.saved.time_ref = mon_old.time_ref
    sim.saved_s.time_ref = mon_old.time_ref

    # time loop initialization
    now = t[0]
    reset_ekf = True
    i_ekf = -1
    i_temp = -1
    T = mon_old.dt[0]
    i = None
    hdr = None
    sat_s_init = None
    ib_amp_init = None
    ib_dyn_amp_init = None
    e_wrap_trim_amp_init = None
    ib_noa_init = None
    ib_dyn_noa_init = None
    e_wrap_trim_noa_init = None

    # Print debug information
    if request_history is not None and request_history > 0:
        hdr = print_hist(request_history, 0, i_temp, i_ekf, t, mon_old, mon, True, True,
                         ST.Tb, ST.Tb_past, sim_old, sim, ST)

    # Top of time loop
    for i in range(t_len):
        if i >= 206:
            pass  # used for debug breakpoint at i >= <val>
        now = t[i]
        mon_old.i = i
        T_ekf = None
        if i != 0:
            candidate_dt = t[i] - t[i-1]  # update
            if candidate_dt > 1e-6:
                T = candidate_dt

        # Get temperature data
        calc_temp = (i_temp+1 < len(mon_old.time_t)) and (mon_old.time_t[i_temp+1] <= mon_old.time[i])
        if calc_temp:
            i_temp += 1
            mon.Tb = mon.Tb_hdwe  # past value
            mon.reset_temp = (i_temp < 2)  # make sure temp init is longer than reset
            mon.dt_temp = mon_old.Tt[i_temp]
            mon.Tb_hdwe = mon_old.Tb_hdwe[i_temp]
            sim.Tb = mon_old.Tb[i_temp]
            mon.Tb = mon_old.Tb[i_temp]
            mon.Tb_s = mon_old.Tb[i_temp]
            if i_temp > 0:
                ST.update()
            mon.Tb_f_rap = ST.Tb_f_past
            mon.Tb_f_rate_rap = ST.Tb_f_rate_past
            sim.Tb_f = mon_old.Tb_mod[i_temp]

        # Input
        dc_dc_on = False
        rp.modeling = modeling[i]

        # Basic reset model verification is to init to the input data
        # Tried hard not to re-implement solvers in the Python verification  tool
        # Also, BTW, did not implement signal selection or tweak logic
        reset = bool((t[i] <= init_time) or (t[i] < 0. and t[0] > init_time))
        if mon_old.res is not None:
            reset = reset or bool(mon_old.res[i] > 0.)
        prn_soc_debug(time=now, leader="before sim init:     ", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)

        if reset:
            sim.apply_soc(mon_old.soc_s[i], ST.Tb_f_past)  # calculates delta_q
            sim.load(sim.delta_q)
            sim.assign_tb(sim.Tb)
            sim.assign_tb_f(sim.Tb_f)
            sim.apply_delta_q_t(sim.delta_q, ST.Tb_f_past)
            sat_s_init = mon_old.voc_stat[0] > mon_old.vsat[0]
            if sim_old is not None:
                sat_s_init = sim_old.sat_s[0]
            sim.sat = sat_s_init
            mon.sat = mon_old.sat[0]

        if calc_temp:
            prn_soc_debug(time=now, leader="b temp filtr:    ", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)

            mon = ST.temp_calc(mon_old, mon, Battery, i_temp)

            prn_soc_debug(time=now, leader="a temp filtr:    ", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)

        # Models
        if sim_old is not None and not use_ib_mon:
            ib_in_s = sim_old.ib_in_s[max(i, 1)]
        else:
            ib_in_s = mon_old.ib[max(i, 1)]

        if Bsim is None:
            _chm_s = chm_s[i]
        else:
            _chm_s = Bsim

        prn_soc_debug(time=now, leader="befor sim.calculater:    ", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)
        ib_dyn_rate_init = sim_old.ib_dyn_rate_s[i]
        sim.calculate(_chm_s, None, ib_in_s, sim_old.dt_s[i], reset, None, None,
                      ib_dyn_init=sim_old.ib_dyn_s[i], ib_dyn_rate_init=ib_dyn_rate_init,
                      soc=sim.soc, q_capacity=sim.q_capacity, dc_dc_on=dc_dc_on, rp=rp, sat_init=sat_s_init,
                      bms_off_init=mon_old.bms_off[0], dv_dyn_0=sim_old.dv_dyn_s[i])
        prn_soc_debug(time=now, leader="after sim.calculater:    ", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)
        sim.count_coulombs(chem=_chm_s, dt=sim_old.dt_s[i], reset_temp=reset, tb_f=sim.Tb_f, tb_f_rate=ST.Tb_f_rate_past,
                           charge_curr=sim.ib_charge, sat=False, soc_s_init=sim_old.soc_s[i], mon_sat=mon.sat,
                           sim_delta_q=sim_old.dq_s[i], use_soc_in=use_mon_soc, soc_in=sim_old.soc_s[i])
        prn_soc_debug(time=now, leader="after sim.count_cou:    ", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)

        # EKF
        z_init = None
        if reset:
            mon.apply_delta_q_t(mon_old.delta_q[i], mon_old.Tb_f_rap[i])
            rp.delta_q = mon.delta_q
            mon.load(rp.delta_q)

        prn_soc_debug(time=now, leader="after mon_soc_apply  ", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)

        # Chemistry
        if Bmon is None:
            _chm_m = chm_m[i]
        else:
            _chm_m = Bmon

        if t_ib_fail and t[i] > t_ib_fail:
            ib_ = ib_fail
        else:
            if mon_old.ib_sel is not None:
                ib_ = mon_old.ib_sel[i]
            else:
                ib_ = mon_old.ib[i]

        if use_vb_sim:
            vb_ = sim.vb
        elif t_vb_fail and t[i] >= t_vb_fail:
            vb_ = vb_fail
        else:
            vb_ = vb[i]

        # Monitor EKF sequencing logic
        if (i_ekf+1 < len(mon_old.time_e)) and (mon_old.time_e[i_ekf+1] <= mon_old.time[i]):
            i_ekf += 1
            if i_ekf < 1:
                T_ekf = mon_old.dt_ekf[i_ekf]
            else:
                T_ekf = mon_old.time_e[i_ekf] - mon_old.time_e[i_ekf-1]  # update
            calc_ekf = True
        else:
            calc_ekf = False
        if i_ekf == 0:
            reset_ekf = True

        if reset_ekf and calc_ekf:
            mon.init_soc_ekf(mon_old, i, i_ekf)  # when modeling (assumed in python) ekf wants to equal model

        # Monitor calculate
        if rp.modeling == 0:
            if reset_ekf:
                z_init = mon_old.z[i_ekf]

            # print(f"{i=}   {i_ekf=}   t {mon_old.time[i]}   te {mon_old.time_e[i_ekf]}    dt {mon_old.dt_ekf[i_ekf]}     calc {calc_ekf}      res_ekf {reset_ekf}      z_init {z_init}")
            if reset:
                ib_dyn_rate_init = mon_old.ib_dyn_rate[i]
                ib_amp_init = mon_old.ibmh[max(i-1, 0)]
                ib_dyn_amp_init = mon_old.ib_dyn_m[i]
                ib_noa_init = mon_old.ibnh[max(i-1, 0)]
                ib_dyn_noa_init = mon_old.ib_dyn_n[i]
                e_wrap_trim_amp_init = mon_old.e_wrap_m_trim[i]
                e_wrap_trim_noa_init = 0.
            mon.calculate(_chm_m, vb_, ib_, T, reset, calc_ekf, T_ekf, z_init, ST.Tb_f_rate_past,
                          rp=rp, bms_off_init=mon_old.bms_off[0], ib_amp=mon_old.ibmh[i], ib_noa=mon_old.ibnh[i],
                          e_w_amp_r=mon_old.e_wrap_m[i], e_w_amp_filt_r=mon_old.e_wrap_m_filt[i],
                          e_w_noa_r=mon_old.e_wrap_n[i], e_w_noa_filt_r=mon_old.e_wrap_n_filt[i],
                          reset_ekf=reset_ekf, ib_dyn_init=mon_old.ib_dyn[i], ib_dyn_rate_init=ib_dyn_rate_init,
                          ib_amp_init=ib_amp_init, ib_dyn_amp_init=ib_dyn_amp_init,
                          ib_noa_init=ib_noa_init, ib_dyn_noa_init=ib_dyn_noa_init,
                          e_wrap_trim_amp_init=e_wrap_trim_amp_init, e_wrap_trim_noa_init=e_wrap_trim_noa_init)
        else:
            mon.calculate(_chm_m, vb_ + randn() * v_std + dv_sense, ib_ + randn() * i_std + di_sense, T,
                          reset, calc_ekf, T_ekf, mon_old.z[0], ST.Tb_f_rate_past,
                          rp=rp, bms_off_init=mon_old.bms_off[0], ib_amp=mon_old.ibmm[i], ib_noa=mon_old.ibnm[i],
                          e_w_amp_r=mon_old.e_wrap_m[i],
                          e_w_amp_filt_r=mon_old.e_wrap_m_filt[i], e_w_noa_r=mon_old.e_wrap_n[i],
                          e_w_noa_filt_r=mon_old.e_wrap_n_filt[i],
                          reset_ekf=reset_ekf)
        ib_charge = mon.ib_charge
        sat = is_sat(ST.Tb_f_past, mon.chemistry.rated_temp, mon.voc_filt, mon.soc, mon.chemistry.nom_vsat,
                     mon.chemistry.dvoc_dt, mon.chemistry.low_t)
        saturated = Is_sat_delay.calculate(sat, T_SAT, T_DESAT, min(T, T_SAT / 2.), reset)

        # Monitor count Coulumbs
        if rp.modeling == 0:
            mon.count_coulombs(chem=_chm_m, dt=T, reset=reset, tb_f=ST.Tb_f_past, charge_curr=ib_charge,
                               sat=saturated, use_soc_in=use_mon_soc, soc_in=mon_old.soc[i])
        else:
            mon.count_coulombs(chem=_chm_m, dt=T, reset=reset, tb_f=ST.Tb_f_past, charge_curr=ib_charge,
                               sat=saturated, use_soc_in=use_mon_soc, soc_in=mon_old.soc[i])
        mon.calc_charge_time(mon.q, mon.q_capacity, ib_charge, mon.soc)
        mon.assign_soc_s(sim.soc)

        # Break is data integrity questionable
        if mon_old.skip_e[i_ekf] or mon_old.skip_t[i_temp] or mon_old.skip_sel[i] or mon_old.skip_rap[i] or sim_old.skip_s[i]:
            break

        # Save plot info
        mon.save(t[i], T, mon.soc, sim.voc)
        sim.save(t[i], T)
        sim.save_s(t[i])

        # Print initial
        if i == 0 and verbose:
            print('time=', t[i])
            print('mon:  ', str(mon))
            print('time=', t[i])
            print('sim:  ', str(sim))

        # History print
        if request_history is not None and request_history > 0:
            hdr = print_hist(request_history, i, i_temp, i_ekf, t, mon_old, mon, calc_temp, calc_ekf,
                             ST.Tb, ST.Tb_past, sim_old, sim, ST)

        prn_soc_debug(time=now, leader="end loop:", i=i, i_temp=i_temp, mon_old=mon_old, mon=mon)

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
    if request_history is not None and request_history > 0:
        print(hdr)
    if mon_old.skip_e[i_ekf] or mon_old.skip_t[i_temp] or mon_old.skip_sel[i] or mon_old.skip_rap[i] or sim_old.skip_s[i]:
        print(f"\n\n************** Data integrity degraded by skip.  A digit could have been inserted anywhere in data.  Break.")
        print("   now {:5.3f}".format(now),
              "   time_end {:5.3f}\n\n".format(t[-1]),
              )

    # Data
    if verbose:
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
        t_ib_fail = None
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
        mon_ver, sim_ver, sim_s_ver, _mon, _sim =\
            replicate(mon_old, sim_old=sim_old, init_time=init_time, sres0=1.0, sresct=1.0, t_ib_fail=t_ib_fail,
                      use_ib_mon=use_ib_mon_in, scale_in=scale_in, use_vb_raw=use_vb_raw, scale_r_ss=scale_r_ss_in,
                      s_hys_sim=scale_hys_sim_in, dvoc_sim=dvoc_sim_in, dvoc_mon=dvoc_mon_in,
                      Bmon=Bmon_in, Bsim=Bsim_in)
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
