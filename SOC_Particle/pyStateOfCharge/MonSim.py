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

import numpy as np
from numpy.random import randn
import Battery
from Battery import Battery, BatteryMonitor, BatterySim, is_sat, Retained
from Battery import overall_batt
from TFDelay import TFDelay
from MonSimNomConfig import *  # Global config parameters.   Overwrite in your own calls for studies
from datetime import datetime, timedelta
from Scale import Scale
from myFilters import LagExp
from pyDAGx import myTables


def print_soc_debug(leader="", reset=None, mo_soc=None, mv_soc=None, mv_Tb_f=None, mv_q=None, mv_q_capacity=None):
    print(leader, end='')
    print("reset {:2.0f}     mo.soc {:10.8f}    mon.soc {:10.8f}    mon.Tb_f_past {:10.8f}    mon.q {:10.3f}    mon.q_cap {:10.3f}".
          format(reset, mo_soc, mv_soc, mv_Tb_f, mv_q, mv_q_capacity))

def print_soc_hist(i, i_temp, t, mon_old, mon, calc_temp):
    hdr = "  i  time   r r_t sa sa_v  ib_c   ib_c_v   soc      soc_v        dt    dt_v   delq     delq_v      qcrs   qcrs_v    q_cap  q_cap_v    Tb         Tb_v           Tb_f       Tb_f_v       Tb_f_rap   Tb_f_rap_v   Tb_f_rate  Tb_f_rate_v"
    if calc_temp:
        print(hdr)
    print("{:3d}".format(i), "{:6.3f}".format(t[i]), "{:2.0f}".format(mon.reset), "{:2.0f}".format(mon.reset_temp),
          "{:2.0f}".format(mon_old.sat[i]), "{:2.0f}".format(mon.sat),
          "{:9.3f}".format(mon_old.ib_charge[i]), "{:6.3f}".format(mon.ib_charge),
          "{:11.6f}".format(mon_old.soc[i]), "{:8.6f}".format(mon.soc),
          "{:9.3f}".format(mon_old.dt[i]), "{:5.3f}".format(mon.dt),
          "{:9.1f}".format(mon_old.delta_q[i]), "{:5.1f}".format(mon.delta_q),
          "{:9.0f}".format(mon_old.qcrs[i]), "{:6.0f}".format(mon.q_cap_rated_scaled),
          "{:9.0f}".format(mon_old.q_capacity[i]), "{:6.0f}".format(mon.q_capacity),
          "{:14.7f}".format(mon_old.Tb[i_temp]), "{:10.7f}".format(mon.Tb),
          "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:12.7f}".format(mon_old.Tb_f_rate[i_temp]), "{:10.7f}".format(mon.Tb_f_rate),
         )
    return hdr

def print_temp_hist(i, i_temp, t, mon_old, mon, calc_temp, Tb_, Tb_past_):
    hdr = "  i  time  r  r_t i_t  calc   Tt      Tb_hdwe     Tb_hdwe_v         Tb   Tb_v                 Tb_        Tb_past_   Tb_hdwe_filt  Tb_hdwe_filt_v     Tb_rap  Tb_rap_v         Tb_f      Tb_f_v          Tb_f_rap  Tb_f_rap_v        Tb_h_f_r  Tb_h_f_r_v     Tb_f_rate Tb_f_rate_v      Tb_f_rate_rap Tb_f_rate_rap_v"
    if calc_temp:
        print(hdr)
    print("{:3d}".format(i), "{:6.3f}".format(t[i]), "{:2.0f}".format(mon.reset),
          "{:2d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7.3f}".format(mon_old.Tt[i_temp]),
          "{:13.7f}".format(mon_old.Tb_hdwe[i_temp]), "{:11.7f}".format(mon.Tb_hdwe),
          "{:14.7f}".format(mon_old.Tb[i_temp]), "{:11.7f}".format(mon.Tb),
          "{:14.7f}".format(Tb_), "{:11.7f}".format(Tb_past_),
          "{:14.7f}".format(mon_old.Tb_hdwe_filt[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt),
          "{:14.7f}".format(mon_old.Tb_rap[i]), "{:11.7f}".format(mon.Tb_rap),
          "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:11.7f}".format(mon.Tb_f),
          "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:11.7f}".format(mon.Tb_f_rap),
          "{:14.7f}".format(mon_old.Tb_hdwe_filt_rate[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt_rate),
          "{:14.7f}".format(mon_old.Tb_f_rate[i_temp]), "{:11.7f}".format(mon.Tb_f_rate),
          "{:14.7f}".format(mon_old.Tb_f_rate_rap[i]), "{:11.7f}".format(mon.Tb_f_rate_rap),
          )
    return hdr

def save_clean_file(mon_ver, csv_file, unit_key):
    default_header_str = "unit,               hm,                  cTime,        dt,       sat,sel,mod,\
      Tb,Tb_rap,Tb_f,Tb_f_rap,Tb_f_rate,Tb_f_rate_rap, vb,  ib,  ioc,  voc_soc,    vsat,dv_dyn,voc_stat,voc_stat_f,voc_ekf,     y_ekf,    soc_s,soc_ekf,soc,ib_lag,voc_soc_new,"
    n = len(mon_ver.time)
    date_time_start = datetime.now()
    with open(csv_file, "w") as output:
        output.write(default_header_str + "\n")
        for i in range(n):
            s = unit_key + ','
            dt_dt = timedelta(seconds=mon_ver.time[i]-mon_ver.time[0])
            time_stamp = date_time_start + dt_dt
            s += time_stamp.strftime("%Y-%m-%dT%H:%M:%S,")
            s += "{:7.3f},".format(mon_ver.time[i] + mon_ver.time_ref)
            s += "{:7.3f},".format(mon_ver.dt[i])
            s += "{:1.0f},".format(mon_ver.sat[i])
            s += "{:1.0f},".format(mon_ver.sel[i])
            s += "{:1.0f},".format(mon_ver.mod_data[i])
            s += "{:7.6f},".format(mon_ver.Tb[i])
            s += "{:7.6f},".format(mon_ver.Tb_rap[i])
            s += "{:7.6f},".format(mon_ver.Tb_f[i])
            s += "{:7.6f},".format(mon_ver.Tb_f_rap[i])
            s += "{:7.6f},".format(mon_ver.Tb_f_rate[i])
            s += "{:7.6f},".format(mon_ver.Tb_f_rate_rap[i])
            s += "{:7.3f},".format(mon_ver.vb[i])
            s += "{:7.3f},".format(mon_ver.ib[i])
            s += "{:7.3f},".format(mon_ver.ioc[i])
            s += "{:7.3f},".format(mon_ver.voc_soc[i])
            s += "{:7.3f},".format(mon_ver.vsat[i])
            s += "{:7.3f},".format(mon_ver.dv_dyn[i])
            s += "{:7.3f},".format(mon_ver.voc_stat[i])
            s += "{:7.3f},".format(mon_ver.voc_ekf[i])
            s += "{:7.3f},".format(mon_ver.y_ekf[i])
            s += "{:7.3f},".format(mon_ver.soc_s[i])
            s += "{:7.3f},".format(mon_ver.soc_ekf[i])
            s += "{:7.3f},".format(mon_ver.soc[i])
            s += "{:7.5f},".format(mon_ver.ib_lag[i])
            s += "{:7.3f},".format(mon_ver.voc_soc_new[i])
            s += "\n"
            output.write(s)
        print("Wrote(save_clean_file):", csv_file)


def save_clean_file_sim(sim_ver, csv_file, unit_key):
    header_str = "unit_m,c_time,Tb_s,vsat_s,voc_stat_s,dv_dyn_s,vb_s,ib_s,sat_s,dq_s,\
    soc_s,reset_s,"
    n = len(sim_ver.time)
    with open(csv_file, "w") as output:
        output.write(header_str + "\n")
        for i in range(n):
            s = unit_key + ','
            s += "{:13.3f},".format(sim_ver.time[i])
            s += "{:5.2f},".format(sim_ver.Tb_s[i])
            s += "{:8.3f},".format(sim_ver.vsat_s[i])
            s += "{:5.2f},".format(sim_ver.voc_stat_s[i])
            s += "{:5.2f},".format(sim_ver.dv_dyn_s[i])
            s += "{:5.2f},".format(sim_ver.vb_s[i])
            s += "{:8.3f},".format(sim_ver.ib_s[i])
            s += "{:7.3f},".format(sim_ver.sat_s[i])
            s += "{:5.3f},".format(sim_ver.dq_s[i])
            s += "{:7.3f},".format(sim_ver.soc_s[i])
            s += "{:7.3f},".format(sim_ver.reset_s[i])
            s += "\n"
            output.write(s)
        print("Wrote(save_clean_file_sim):", csv_file)

#  Replicate the application in its entirety here.
#  There are no 'bank' parameters anywhere in this model.   It is assumed that all inputs from the application have
#  been converted to the single battery unit 12v form, S1P1, lower-case nomenclature.
def replicate(mon_old, sim_old=None, init_time=-4., t_vb_fail=None, vb_fail=13.2,
              t_ib_fail=None, ib_fail=0., use_ib_mon=False, scale_in=None, Bsim=None, Bmon=None, use_vb_raw=False,
              scale_r_ss=1., s_hys_sim=1., s_hys_mon=1., dvoc_sim=0., dvoc_mon=0., drive_ekf=False, dTb_in=None,
              verbose=True, t_max=None, eframe_mult=Battery.cp_eframe_mult, sres0=1., sresct=1., stauct_sim=1.,
              stauct_mon=1, use_vb_sim=False, scale_hys_cap_sim=1., s_cap_chg=1., s_cap_dis=1.,
              s_hys_chg=1., s_hys_dis=1., s_coul_eff=1., use_mon_soc=False, cutback_gain_sclr=1., ds_voc_soc=0.,
              unit=None, request_temp_history=False, request_soc_history=False):
    if sim_old is not None and len(sim_old.time) < len(mon_old.time):
        t = sim_old.time
    else:
        t = mon_old.time
    if t_max is not None:
        t_delt = t - t[0]
        t = t[np.where(t_delt <= t_max)]
    reset_sel = mon_old.res
    if use_vb_raw:
        vb = mon_old.vb_h
    else:
        vb = mon_old.vb
    chm_m = mon_old.chm
    if sim_old is not None:
        chm_s = sim_old.chm_s
        # sat_s_init = sim_old.sat_s[0]
    else:
        chm_s = chm_m
        # sat_s_init = mon_old.voc_stat[0] > mon_old.vsat[0]
    t_len = len(t)
    rp = Retained()
    if hasattr(mon_old, 'mod_data'):
        modeling = mon_old.mod_data
    else:
        modeling = 255 * np.ones(len(mon_old.time))
        print(f"what do we do now?  {rp.modeling=}")
        # exit(1)
    print("use_mon_soc is", use_mon_soc, "use_ib_mon is", use_ib_mon)
    tweak_test = rp.tweak_test()
    Tb0 = mon_old.Tb_f[0]
    lut_dTb = None
    if dTb_in is not None:
        dTb_in = np.array(dTb_in)
        Tb0 += dTb_in[1, 0]
        lut_dTb = myTables.TableInterp1D(np.array(dTb_in[0, :]), np.array(dTb_in[1, :]))

    # Setup
    TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)
    if hasattr(mon_old, 'qcrs'):
        scale_mon = mon_old.qcrs[0] / (Battery.UNIT_CAP_RATED*3600)
    else:
        scale_mon = unit_cap_rated / Battery.UNIT_CAP_RATED
        if scale_in:
            scale_mon *= scale_in
    if sim_old is not None and hasattr(sim_old, 'qcrs_s'):
        scale_sim = sim_old.qcrs_s[0] / (Battery.UNIT_CAP_RATED*3600)
    else:
        scale_sim = unit_cap_rated / Battery.UNIT_CAP_RATED
        if scale_in:
            scale_sim *= scale_in
    s_q = Scale(1., 3., 0.000005, 0.00005)
    s_r = Scale(1., 3., 0.001, 1.)   # t_ib_fail = 1000
    sim = BatterySim(mod_code=chm_s[0], tb_f=Tb0, scale=scale_sim, tweak_test=tweak_test,
                     dv_hys=mon_old.dv_hys[0], sres0=sres0, sresct=sresct, stauct=stauct_sim, scale_r_ss=scale_r_ss,
                     s_hys=s_hys_sim, dvoc=dvoc_sim, scale_hys_cap=scale_hys_cap_sim, s_coul_eff=s_coul_eff,
                     s_cap_chg=s_cap_chg, s_cap_dis=s_cap_dis, s_hys_chg=s_hys_chg, s_hys_dis=s_hys_dis,
                     cutback_gain_sclr=cutback_gain_sclr, ds_voc_soc=ds_voc_soc, unit=unit)
    mon = BatteryMonitor(mod_code=chm_m[0], tb_f=Tb0, scale=scale_mon, tweak_test=tweak_test,
                         sres0=sres0, sresct=sresct, stauct=stauct_mon, scaler_q=s_q, scaler_r=s_r,
                         scale_r_ss=scale_r_ss, s_hys=s_hys_mon, dvoc=dvoc_mon, eframe_mult=eframe_mult,
                         s_coul_eff=s_coul_eff, unit=unit)
    mon.saved.time_ref = mon_old.time_ref
    sim.saved_s.time_ref = mon_old.time_ref
    # need Tb input.   perhaps need higher order to enforce basic type 1 response
    Is_sat_delay = TFDelay(in_=mon_old.soc[0] > 0.97, t_true=T_SAT, t_false=T_DESAT, dt=0.1)  # later, dt is changed
    bms_off_init = mon_old.bms_off[0]
    e_w_amp_0 = None
    e_w_amp_filt_0 = None
    e_w_noa_0 = None
    e_w_noa_filt_0 = None
    temp_hdr = None
    soc_hdr = None

    # time loop initialization
    now = t[0]
    i_ekf = None
    i_temp = None
    hdr = None
    i_ekf = -1
    i_temp = -1
    mon.dt_temp = 0.
    T = mon_old.dt[0]
    if dTb_in is not None:
        dTb = lut_dTb.interp(t[0])
    else:
        dTb = 0.
    mon.Tb_hdwe = mon_old.Tb_hdwe[0]
    # mon.Tb_s = mon_old.Tb_s[0]
    mon.Tb_hdwe_filt = mon_old.Tb_hdwe_filt[0]
    mon.Tb_hdwe_filt_rate = mon_old.Tb_hdwe_filt_rate[0]
    Tb_ = mon_old.Tb_rap[0] + dTb
    Tb_f_ = mon_old.Tb_f_rap[0] + dTb
    Tb_f_rate_past_ = mon_old.Tb_f_rate[0]
    sim.Tb = mon_old.Tb[0]
    mon.Tb = mon_old.Tb_rap[0]
    mon.Tb_f = mon_old.Tb_f_rap[0]
    mon.Tb_f_rate = mon_old.Tb_f_rate[0]
    Tb_past_ = Tb_
    Tb_f_past_ = Tb_f_
    reset = True
    # Top of time loop
    for i in range(t_len):
        print_soc_debug(leader="\n\ntop:                 ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc, mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)
        now = t[i]
        mon_old.i = i
        T_ekf = None
        if i != 0:
            candidate_dt = t[i] - t[i-1]  # update
            if candidate_dt > 1e-6:
                T = candidate_dt
        if dTb_in is not None:
            dTb = lut_dTb.interp(t[i])
        else:
            dTb = 0.
        # Get temperature data
        calc_temp = (i_temp+1 < len(mon_old.time_t)) and (mon_old.time_t[i_temp+1] <= mon_old.time[i])
        if calc_temp:
            i_temp += 1
            mon.Tb = mon.Tb_hdwe  # past value
            mon.reset_temp = (i_temp < 2)  # make sure temp init is longer than reset
            mon.dt_temp = mon_old.Tt[i_temp]
            mon.Tb_hdwe = mon_old.Tb_hdwe[i_temp]
            sim.Tb = mon_old.Tb[i_temp]
            Tb_past_ = Tb_
            Tb_f_past_ = Tb_f_
            Tb_f_rate_past_ = mon.Tb_f_rate
            mon.Tb = mon_old.Tb[i_temp]
            mon.Tb_s = mon_old.Tb[i_temp]
            Tb_ = mon.Tb + dTb
            Tb_f_ = mon.Tb_f + dTb
            sim.Tb_f = mon.Tb_f

        # dc_dc_on = bool(lut_dc.interp(t[i]))
        dc_dc_on = False
        rp.modeling = modeling[i]

        # Basic reset model verification is to init to the input data
        # Tried hard not to re-implement solvers in the Python verification  tool
        # Also, BTW, did not implement signal selection or tweak logic
        reset = (t[i] <= init_time) or (t[i] < 0. and t[0] > init_time)
        if reset_sel is not None:
            reset = reset or reset_sel[i]
        # if mon.reset_temp is not None:
        #     reset = reset or mon.reset_temp
        print_soc_debug(leader="before reset:        ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc, mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)

        if reset:
            sim.apply_soc(mon_old.soc_s[i], Tb_f_past_)  # calculates delta_q
            sim.load(sim.delta_q)
            sim.assign_tb(Tb_past_)
            sim.assign_tb_f(Tb_f_past_)
            sim.apply_delta_q_t(sim.delta_q, Tb_f_past_)
            if sim_old is not None:
                sat_s_init = sim_old.sat_s[0]
            else:
                sat_s_init = mon_old.voc_stat[0] > mon_old.vsat[0]
            sim.sat = sat_s_init
            mon.sat = mon_old.sat[0]

        if calc_temp:
            if mon.reset_temp:
                Tb_f_rate_past = mon_old.Tb_hdwe_filt_rate[i_temp]
            else:
                Tb_f_rate_past = mon.Tb_hdwe_filt_rate
            mon.Tb_hdwe_filt = \
                TbSenseFilt.calculate_tau_seeded(mon.Tb_hdwe, mon_old.Tb_hdwe_filt[i_temp],
                                                 mon_old.Tb_hdwe_filt_rate[i_temp], mon.reset_temp,
                                                 mon.dt_temp, Battery.TB_FILT, rmax=Battery.T_RLIM,
                                                 rmin=-Battery.T_RLIM)
            mon.Tb_hdwe_filt_rate = TbSenseFilt.rate
            mon.Tb_f_rate = mon.Tb_hdwe_filt_rate
            mon.Tb_rap = Tb_past_
            mon.Tb_f = mon.Tb_hdwe_filt
            Tb_f_ = mon.Tb_hdwe_filt
            mon.Tb_rstate = TbSenseFilt.rstate
            mon.Tb_state = TbSenseFilt.state

            print_each_update = True
            if print_each_update:
                print("{:6.3f} reset   {:2.0f}  Tt {:9.7f}  Tb_hdwe  {:11.7f}  Tb_hdwe_filt   {:11.7f} rstate   {:11.7f} lstate   {:11.7f} hwfrate   {:11.7f} tbfrate   {:11.7f} Tb_rap  {:8.3f} Tb_f   {:8.3f}".
                      format(now, mon_old.reset_temp[i_temp], mon_old.Tt[i_temp],
                             mon_old.Tb_hdwe[i_temp],
                             mon_old.Tb_hdwe_filt[i_temp], mon_old.Tb_rstate[i_temp], mon_old.Tb_lstate[i_temp],
                             mon_old.Tb_hdwe_filt_rate[i_temp], mon_old.Tb_f_rate[i_temp],
                             mon_old.Tb_rap[i], mon_old.Tb_f[i_temp]))
                print("{:6.3f} reset_v {:2d}  Tt {:9.7f}  Tb_hdwe_v{:11.7f}  Tb_hdwe_filt_v {:11.7f} rstate_v {:11.7f} lstate_v {:11.7f} hwfrate_v {:11.7f} tbfrate_v {:11.7f} Tb_rap_v{:8.3f} Tb_f_v {:8.3f}\n\n".
                      format(now, mon.reset_temp, mon.dt_temp,
                             mon.Tb_hdwe,
                             mon.Tb_hdwe_filt, mon.Tb_rstate, mon.Tb_state,
                             mon.Tb_hdwe_filt_rate, mon.Tb_f_rate,
                             mon.Tb_rap, mon.Tb_f))

        # Models
        if sim_old is not None and not use_ib_mon:
            ib_in_s = sim_old.ib_in_s[i]
        else:
            ib_in_s = mon_old.ib[i]
        if Bsim is None:
            _chm_s = chm_s[i]
        else:
            _chm_s = Bsim
        sim.calculate(_chm_s, None, ib_in_s, T, reset, None, None, None,
                      soc=sim.soc, q_capacity=sim.q_capacity, dc_dc_on=dc_dc_on, rp=rp, sat_init=sat_s_init,
                      bms_off_init=bms_off_init)
        sim.count_coulombs(chem=_chm_s, dt=T, reset=reset, tb_f=Tb_f_, tb_f_rate=Tb_f_rate_past_, charge_curr=sim.ib_charge,
                           sat=False, soc_s_init=mon_old.soc_s[0], mon_sat=mon.sat, mon_delta_q=mon.delta_q,
                           use_soc_in=use_mon_soc, soc_in=mon_old.soc[i])
        print_soc_debug(leader="after sim.calculate: ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc, mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)

        # EKF
        reset_ekf = False
        z_init = None
        if reset:
            mon.apply_soc(mon_old.soc[i], Tb_f_)
            rp.delta_q = mon.delta_q
            mon.load(rp.delta_q)
            mon.assign_tb(Tb_past_)
            if hasattr(mon_old, 'e_wrap_m'):
                e_w_amp_0 = mon_old.e_wrap_m[0]
            if hasattr(mon_old, 'e_wrap_m_filt'):
                e_w_amp_filt_0 = mon_old.e_wrap_m_filt[0]
            if hasattr(mon_old, 'e_wrap_n'):
                e_w_noa_0 = mon_old.e_wrap_n[0]
            if hasattr(mon_old, 'e_wrap_n_filt'):
                e_w_noa_filt_0 = mon_old.e_wrap_n_filt[0]
            reset_ekf = True
        print_soc_debug(leader="after reset:         ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc, mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)

        # Monitor calculations including ekf
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

        # Raw current handling
        ibmm = None
        ibnm = None
        ibmh = None
        ibnh = None
        if hasattr(mon_old, 'ibmm'):
            ibmm = mon_old.ibmm[i]
        if hasattr(mon_old, 'ibnm'):
            ibnm = mon_old.ibnm[i]
        if hasattr(mon_old, 'ibmh'):
            ibmh = mon_old.ibmh[i]
        if hasattr(mon_old, 'ibnm'):
            ibnh = mon_old.ibnh[i]

        if use_vb_sim:
            vb_ = sim.vb
        elif t_vb_fail and t[i] >= t_vb_fail:
            vb_ = vb_fail
        else:
            vb_ = vb[i]

        # EKF sequencing logic
        if (i_ekf+1 < len(mon_old.time_e)) and (mon_old.time_e[i_ekf+1] <= mon_old.time[i]):
            i_ekf += 1
            if i_ekf < 1:
                T_ekf = mon_old.time_e[1] - mon_old.time_e[0]
            else:
                T_ekf = mon_old.time_e[i_ekf] - mon_old.time_e[i_ekf-1]  # update
            calc_ekf = True
        else:
            calc_ekf = False
        if i_ekf < 1:
            reset_ekf = True
            mon.init_soc_ekf(mon_old.x[0], mon_old.P[0])  # when modeling (assumed in python) ekf wants to equal model

        if rp.modeling == 0:
            if reset_ekf:
                z_init = mon_old.z[i_ekf]
            # print(f"{i=} {i_ekf=} {mon_old.time[i]} {mon_old.time_e[i_ekf]} dt {mon_old.dt_ekf[i_ekf]} calc {calc_ekf} res {reset_ekf} z_init {z_init}")
            print_soc_debug(leader="before mon.calculate ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc,
                            mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)
            mon.calculate(_chm_m, vb_, ib_, T, reset, calc_ekf, T_ekf, z_init, Tb_f_rate_past_,
                          rp=rp, bms_off_init=bms_off_init, ib_amp=ibmh, ib_noa=ibnh, e_w_amp_0=e_w_amp_0,
                          e_w_amp_filt_0=e_w_amp_filt_0, e_w_noa_0=e_w_noa_0, e_w_noa_filt_0=e_w_noa_filt_0,
                          reset_ekf=reset_ekf)
        else:
            mon.calculate(_chm_m, vb_ + randn() * v_std + dv_sense, ib_ + randn() * i_std + di_sense, T,
                          reset, calc_ekf, T_ekf, mon_old.z[0], Tb_f_rate_past_,
                          rp=rp, bms_off_init=bms_off_init, ib_amp=ibmm, ib_noa=ibnm, e_w_amp_0=e_w_amp_0,
                          e_w_amp_filt_0=e_w_amp_filt_0, e_w_noa_0=e_w_noa_0, e_w_noa_filt_0=e_w_noa_filt_0,
                          reset_ekf=reset_ekf)
        print_soc_debug(leader="after mon.calculate: ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc,
                        mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)
        ib_charge = mon.ib_charge
        sat = is_sat(Tb_f_past_, mon.voc_filt, mon.soc, mon.chemistry.nom_vsat, mon.chemistry.dvoc_dt, mon.chemistry.low_t)
        saturated = Is_sat_delay.calculate(sat, T_SAT, T_DESAT, min(T, T_SAT / 2.), reset)
        if rp.modeling == 0:
            mon.count_coulombs(chem=_chm_m, dt=T, reset=reset, tb_f=Tb_f_past_, tb_f_rate=Tb_f_rate_past_, charge_curr=ib_charge,
                               sat=saturated, use_soc_in=use_mon_soc, soc_in=mon_old.soc[i])
        else:
            mon.count_coulombs(chem=_chm_m, dt=T, reset=reset, tb_f=Tb_f_past_, tb_f_rate=Tb_f_rate_past_, charge_curr=ib_charge,
                               sat=saturated, use_soc_in=use_mon_soc, soc_in=mon_old.soc[i])
        print_soc_debug(leader="after mon.count_cou: ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc,
                        mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)
        mon.Tb_f_rap = Tb_f_past_
        mon.Tb_f_rate_rap = Tb_f_rate_past_
        mon.calc_charge_time(mon.q, mon.q_capacity, ib_charge, mon.soc)
        mon.assign_soc_s(sim.soc)
        # Plot stuff
        mon.save(t[i], T, mon.soc, sim.voc)
        sim.save(t[i], T)
        sim.save_s(t[i])

        # Print initial
        if i == 0 and verbose:
            print('time=', t[i])
            print('mon:  ', str(mon))
            print('time=', t[i])
            print('sim:  ', str(sim))
        # if t[i]>29495:
        #     print('time=', t[i])
        #     print('mon:  ', str(mon))
        #     print('time=', t[i])
        #     print('sim:  ', str(sim))
        #     print(t[i])
        if verbose:
            if sim_old is not None:
                print("{:9.3f}".format(t[i]), "{:4.0f}".format(mon_old.chm[i]), "{:4.0f}".format(sim_old.chm_s[i]),
                      "{:9.3f}".format(sim_old.ib_in_s[i]), "{:9.3f}".format(sim_old.dv_hys_s[i]),
                      "{:9.3f}".format(mon_old.ib[i]), "{:12.7f}".format(mon_old.soc[i]), "{:9.3f}".format(mon_old.dv_hys[i]),
                      "{:9.3f}".format(sim.saved_s.ib_in_s[i]), "{:9.3f}".format(sim.hys.ibs), "{:9.3f}".format(sim.hys.ioc),
                      "{:4.0f}".format(sim.sat), "{:9.3f}".format(sim.hys.disabled), "{:9.3f}".format(sim.hys.dv_dot),
                      "{:9.3f}".format(sim.saved.dv_hys[i]), "{:9.3f}".format(mon.saved.ib[i]), "{:12.7f}".format(mon.saved.soc[i]),
                      "{:4.0f}".format(mon.sat), "{:9.3f}".format(mon.saved.dv_hys[i]))
            else:
                print("{:9.3f}".format(t[i]), "{:4.0f}".format(mon_old.chm[i]),
                      "{:9.3f}".format(mon_old.ib[i]), "{:12.7f}".format(mon_old.soc[i]), "{:9.3f}".format(mon_old.dv_hys[i]),
                      "{:9.3f}".format(sim.saved_s.ib_in_s[i]), "{:9.3f}".format(sim.hys.ibs), "{:9.3f}".format(sim.hys.ioc),
                      "{:4.0f}".format(sim.sat), "{:9.3f}".format(sim.hys.disabled), "{:9.3f}".format(sim.hys.dv_dot),
                      "{:9.3f}".format(sim.saved.dv_hys[i]), "{:9.3f}".format(mon.saved.ib[i]), "{:12.7f}".format(mon.saved.soc[i]),
                      "{:4.0f}".format(mon.sat), "{:9.3f}".format(mon.saved.dv_hys[i]))

        if request_temp_history:
            temp_hdr = print_temp_hist(i, i_temp, t, mon_old, mon, calc_temp, Tb_, Tb_past_)

        if request_soc_history:
            soc_hdr = print_soc_hist(i, i_temp, t, mon_old, mon, calc_temp)


        print_soc_debug(leader="end loop:            ", reset=reset, mo_soc=mon_old.soc[i], mv_soc=mon.soc,
                        mv_Tb_f=mon.Tb_f, mv_q=mon.q, mv_q_capacity=mon.q_capacity)

    if request_temp_history:
        print(temp_hdr)
    if request_soc_history:
        print(soc_hdr)

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
        # Save these
        # data_file_old_txt = '../dataReduction/real world Xp20 20220902.txt'; unit_key = 'soc0_2022'; use_ib_mon_in=True; scale_in=1.12

        # Regression suite
        # data_file_old_txt = 'ampHiFail20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'ampLoFail20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'ampHiFailNoise20220914.txt'; unit_key = 'pro_2022';
        # data_file_old_txt = 'rapidTweakRegression20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'rapidTweakRegression40C_20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'slowTweakRegression20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'triTweakDisch20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'satSit20220914.txt'; unit_key = 'pro_2022';
        # data_file_old_txt = 'ampHiFailSlow20220914.txt'; unit_key = 'pro_2022';
        # data_file_old_txt = 'vHiFail20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'pulse20220914.txt'; unit_key = 'pro_2022'; init_time_in=-0.001;
        # data_file_old_txt = 'tbFailMod20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'tbFailHdwe20220914.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'real world Xp20 30C 20220914.txt'; unit_key = 'soc0_2022'; scale_in = 1.084; use_vb_raw = False; scale_r_ss_in = 1.; scale_hys_mon_in = 3.33; scale_hys_sim_in = 3.33; dvoc_mon_in = -0.05; dvoc_sim_in = -0.05
        # data_file_old_txt = 'real world Xp20 30C 20220914a+b.txt'; unit_key = 'soc0_2022'; scale_in = 1.084; use_vb_raw = False; scale_r_ss_in = 1.; scale_hys_mon_in = 3.33; scale_hys_sim_in = 3.33; dvoc_mon_in = -0.05; dvoc_sim_in = -0.05
        # data_file_old_txt = 'real world Xp20 30C 20220917.txt'; unit_key = 'soc0_2022'; scale_in = 1.084; init_time_in = -11110
        # data_file_old_txt = 'EKF_Track 20200917.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'EKF_Track Dr100 v20220917.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'EKF_Track Dr200 v20220917.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'EKF_Track Dr400 v20220917.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'EKF_Track Dr800 v20220917.txt'; unit_key = 'pro_2022'
        data_file_old_txt = 'EKF_Track Dr2000 v20220917.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'EKF_Track Dr200 Xf0p04 v20220917.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'EKF_Track Dr400 Xf0p04 v20220917.txt'; unit_key = 'pro_2022'
        # data_file_old_txt = 'EKF_Track Dr800 Xf0p04 v20220917.txt'; unit_key = 'pro_2022'
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
        mon_old = SavedData(data=mon_old_raw, sel=sel_old_raw, time_end=time_end, zero_zero=zero_zero_in,
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
