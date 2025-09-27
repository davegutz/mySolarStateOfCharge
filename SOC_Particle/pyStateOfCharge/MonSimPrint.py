# MonSimPring:  Debug prints for MonSim
# Copyright (C) 2025 Dave Gutz
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

from datetime import datetime, timedelta

def prn_soc_debug(leader="", time=None, i=None, i_temp=None, mon_old=None, mon=None):
    execute = True
    if execute:
        return
    else:
        if time is not None:
            print("\n\ntime {:7.3f}".format(time))
        print("                                                                                              " + leader, end='')
        print(
    "{:14.7f}".format(mon_old.Tb_hdwe_filt[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt),
    "{:14.7f}".format(mon_old.Tb_rap[i]), "{:11.7f}".format(mon.Tb_rap),
    "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:11.7f}".format(mon.Tb_f),
    "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:11.7f}".format(mon.Tb_f_rap),
    "{:14.7f}".format(mon_old.Tb_hdwe_filt_rate[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt_rate),
    "{:14.7f}".format(mon_old.Tb_f_rate[i_temp]), "{:11.7f}".format(mon.Tb_f_rate),
    "{:14.7f}".format(mon_old.Tb_f_rate_rap[i]), "{:11.7f}".format(mon.Tb_f_rate_rap),
        )

def print_hist(request_history, i, i_temp, i_ekf, t, mon_old, mon, calc_temp, calc_ekf, Tb, Tb_past, sim_old, sim, ST):
    hdr = None
    match request_history:
        case 0:
            hdr = ''
        case 1:
            hdr = print_ekf_hist(i, i_temp, i_ekf, t, mon_old, mon, calc_ekf, calc_temp)
        case 2:
            hdr = print_soc_hist(i, i_temp, t, mon_old, mon, calc_temp)
        case 3:
            hdr = print_soc_s_hist(i, i_temp, t, mon_old, mon, calc_temp, sim_old, sim, i_ekf, calc_ekf)
        case 4:
            hdr = print_temp_hist(i, i_temp, t, mon_old, mon, calc_temp, Tb, Tb_past, ST, i_ekf, calc_ekf)
        case 5:
            hdr = print_volt_hist(i, i_temp, i_ekf, t, mon_old, mon, calc_temp, calc_ekf)
    return hdr

def print_ekf_hist(i, i_temp, i_ekf, t, mon_old, mon, calc_ekf, calc_temp):
    hdr = "  i  time   r r_t  i_e  r_e  c_e   dt_ekf        sa      ib_charge             soc                    soc_ekf                 y_ekf                voc_ekf                Tb_f                    x_prior             fr     Tb_f_rap                x                       tb_f_for_hx             x_for_hx                  hx                    z         z_ekf     P                            P_post                       P_prior                       H                      R                     S                    K                          x_post"
    i_ekf = max(i_ekf, 0)
    if calc_temp or calc_ekf:
        print(hdr)
    print("{:3d}".format(i), "{:6.3f}".format(t[i]), "{:2.0f}".format(mon.reset), "{:2.0f}".format(mon.reset_temp),
          "{:3d}".format(i_ekf), "{:4d}".format(mon.reset_ekf), "{:4d}".format(calc_ekf),
          "{:9.3f}".format(mon_old.dt_ekf[i_ekf]), "{:5.3f}".format(mon.dt_eframe),
          "{:4.0f}".format(mon_old.sat[i]), "{:2.0f}".format(mon.sat),
          "{:10.5f}".format(mon_old.ib_charge[i]), "{:9.5f}".format(mon.ib_charge),
          "{:13.7f}".format(mon_old.soc[i]), "{:10.7f}".format(mon.soc),
          "{:11.7f}".format(mon_old.soc_ekf[i]), "{:9.7f}".format(mon.soc_ekf),
          "{:11.5f}".format(mon_old.y_ekf[i]), "{:9.5f}".format(mon.y_ekf),
          "{:11.5f}".format(mon_old.voc_ekf[i]), "{:9.5f}".format(mon.voc_ekf),
          "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:11.7f}".format(mon_old.x_prior[i_ekf]), "{:9.7f}".format(mon.x_prior), "{:2.0f}".format(mon_old.frz[i_ekf]),
          "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:11.7f}".format(mon_old.x[i_ekf]), "{:9.7f}".format(mon.x),
          "{:14.7f}".format(mon_old.tb_f_for_hx[i_ekf]), "{:10.7f}".format(mon.tb_f_for_hx),
          "{:11.7f}".format(mon_old.x_for_hx[i_ekf]), "{:9.7f}".format(mon.x_for_hx),
          "{:14.5f}".format(mon_old.hx[i_ekf]), "{:9.5f}".format(mon.hx),
          "{:11.5f}".format(mon_old.z[i_ekf]), "{:9.5f}".format(mon.z_ekf),
          "{:14.11f}".format(mon_old.P[i_ekf]), "{:12.11f}".format(mon.P),
          "{:14.11f}".format(mon_old.P_post[i_ekf]), "{:12.11f}".format(mon.P_post),
          "{:14.11f}".format(mon_old.P_prior[i_ekf]), "{:12.11f}".format(mon.P_prior),
          "{:11.7f}".format(mon_old.H[i_ekf]), "{:9.7f}".format(mon.H),
          "{:11.6f}".format(mon_old.R[i_ekf]), "{:9.6f}".format(mon.R),
          "{:11.6f}".format(mon_old.S[i_ekf]), "{:9.6f}".format(mon.S),
          "{:13.9f}".format(mon_old.K[i_ekf]), "{:10.9f}".format(mon.K),
          "{:12.7f}".format(mon_old.x_post[i_ekf]), "{:9.7f}".format(mon.x_post),
          )
    return hdr

def print_soc_hist(i, i_temp, t, mon_old, mon, calc_temp):
    hdr = "  i  time   r r_t sa      ib_charge             soc                  dt                i * dt * coul_eff    Tb_f                      Tb_f_rap                    ddq                  delq                   qcrs                   q_cap                  Tb                        Tb_f_rate"
    if calc_temp:
        print(hdr)
    if i > 0:
        d_dq = mon_old.delta_q[i]-mon_old.delta_q[i-1]
    else:
        d_dq = mon_old.delta_q[i+1]-mon_old.delta_q[i]
    i_dt_old = mon_old.dt[i] * mon_old.ib_charge[i]
    i_dt_new = mon.dt * mon.ib_charge
    coul_eff = 0.9985
    if mon.ib_charge > 0:
        i_dt_old *= coul_eff
        i_dt_new *= coul_eff
    print("{:3d}".format(i), "{:6.3f}".format(t[i]), "{:2.0f}".format(mon.reset), "{:2.0f}".format(mon.reset_temp),
          "{:2.0f}".format(mon_old.sat[i]), "{:2.0f}".format(mon.sat),
          "{:10.5f}".format(mon_old.ib_charge[i]), "{:9.5f}".format(mon.ib_charge),
          "{:11.5f}".format(mon_old.soc[i]), "{:8.5f}".format(mon.soc),
          "{:9.3f}".format(mon_old.dt[i]), "{:5.3f}".format(mon.dt),
          "{:12.4f}".format(i_dt_old), "{:9.4f}".format(i_dt_new),
          "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:12.4f}".format(d_dq), "{:9.4f}".format(mon.d_delta_q),
          "{:12.4f}".format(mon_old.delta_q[i]), "{:9.4f}".format(mon.delta_q),
          "{:12.1f}".format(mon_old.qcrs[i]), "{:9.1f}".format(mon.q_cap_rated_scaled),
          "{:12.1f}".format(mon_old.q_capacity[i]), "{:9.1f}".format(mon.q_capacity),
          "{:14.7f}".format(mon_old.Tb[i_temp]), "{:10.7f}".format(mon.Tb),
          "{:12.7f}".format(mon_old.Tb_f_rate[i_temp]), "{:10.7f}".format(mon.Tb_f_rate),
         )
    return hdr

def print_soc_s_hist(i, i_temp, t, mon_old, mon, calc_temp, sim_old, sim, i_ekf, calc_ekf):
    hdr = "  i  time   r       rt   it   ct      re   ie  ce    sa       sa_s     dt              dt_s             ib_in_s                ib_s              ib_fut         ib_dyn_s               dv_hys_s               ib_charge_s            ioc_s                 soc                  soc_s               delq                           i * dt_s * coul_eff      d_delq_s            delq_s                      qcrs                    q_cap                  q_cap_s                Tb_f_s                    Tb_f                      Tb_f_rap                 Tb_f_rate               vb                     vb_s                 voc_stat               voc_stat_s            voc_s                 dv_dyn_s             vsat                 "
    if calc_temp:
        print(hdr)
    if i > 0:
        d_dq_s = sim_old.dq_s[i]-sim_old.dq_s[i-1]
    else:
        d_dq_s = sim_old.dq_s[i+1]-sim_old.dq_s[i]
    i_dt_old = sim_old.dt_s[i] * sim_old.ib_charge_s[i]
    i_dt_new = sim.dt * sim.ib_charge
    coul_eff = 0.9985
    if sim.ib_charge > 0:
        i_dt_old *= coul_eff
        i_dt_new *= coul_eff
    print("{:3d}".format(i), "{:6.3f}".format(t[i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(mon_old.sat[i]), "{:2.0f}".format(mon.sat),
          "{:5.0f}".format(sim_old.sat_s[i]), "{:2.0f}".format(sim.sat),
          "{:9.3f}".format(mon_old.dt[i]), "{:5.3f}".format(mon.dt),
          "{:9.3f}".format(sim_old.dt_s[i]), "{:5.3f}".format(sim.dt),
          "{:12.5f}".format(sim_old.ib_in_s[i]), "{:9.5f}".format(sim.ib_in),
          "{:12.5f}".format(sim_old.ib_s[i]), "{:9.5f}".format(sim.ib), "{:9.5f}".format(sim.ib_fut),
          "{:12.5f}".format(sim_old.ib_dyn_s[i]), "{:9.5f}".format(sim.ib_dyn),
          "{:12.5f}".format(sim_old.dv_hys_s[i]), "{:9.5f}".format(sim.dv_hys),
          "{:12.5f}".format(sim_old.ib_charge_s[i]), "{:9.5f}".format(sim.ib_charge),
          "{:12.5f}".format(sim_old.ioc_s[i]), "{:9.5f}".format(sim.ioc),
          "{:11.5f}".format(mon_old.soc[i]), "{:8.5f}".format(mon.soc),
          "{:11.5f}".format(mon_old.soc_s[i]), "{:8.5f}".format(sim.soc),
          "{:14.4f}".format(mon_old.delta_q[i]), "{:9.4f}".format(mon.delta_q),
          "{:12.4f}".format(i_dt_old), "{:9.4f}".format(i_dt_new),
          "{:14.4f}".format(d_dq_s), "{:9.4f}".format(sim.d_delta_q),
          "{:14.4f}".format(sim_old.dq_s[i]), "{:9.4f}".format(sim.delta_q),
          "{:12.1f}".format(mon_old.qcrs[i]), "{:9.1f}".format(mon.q_cap_rated_scaled),
          "{:12.1f}".format(mon_old.q_capacity[i]), "{:9.1f}".format(mon.q_capacity),
          "{:12.1f}".format(sim_old.qcap_s[i]), "{:9.1f}".format(sim.q_capacity),
          "{:14.7f}".format(sim_old.Tb_f_s[i]), "{:10.7f}".format(sim.Tb_f),
          "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:12.7f}".format(mon_old.Tb_f_rate[i_temp]), "{:10.7f}".format(mon.Tb_f_rate),
          "{:11.5f}".format(mon_old.vb[i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(sim_old.vb_s[i]), "{:9.5f}".format(sim.vb),
          "{:11.5f}".format(mon_old.voc_stat[i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(sim_old.voc_stat_s[i]), "{:9.5f}".format(sim.voc_stat),
          "{:11.5f}".format(sim_old.voc_s[i]), "{:9.5f}".format(sim.voc),
          "{:11.5f}".format(sim_old.dv_dyn_s[i]), "{:9.5f}".format(sim.dv_dyn),
          "{:11.5f}".format(mon_old.vsat[i]), "{:9.5f}".format(mon.vsat),
          )
    return hdr

def print_temp_hist(i, i_temp, t, mon_old, mon, calc_temp, Tb_, Tb_past_, ST, i_ekf, calc_ekf):
    hdr = "  i  time   r       rt   it   ct      re   ie  ce     Tt      Tb_hdwe                    Tb                         Tb_                        Tb_past_  Tb_hdwe_filt     Tb_rap                     Tb_f                       Tb_f_rap                    Tb_h_f_r                   Tb_f_rate                              Tb_f_rate_rap              tb_f_for_hx"
    if calc_temp:
        print(hdr)
    print("{:3d}".format(i), "{:6.3f}".format(t[i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:9.3f}".format(mon_old.Tt[i_temp]),
          "{:13.7f}".format(mon_old.Tb_hdwe[i_temp]), "{:11.7f}".format(mon.Tb_hdwe),
          "{:14.7f}".format(mon_old.Tb[i_temp]), "{:11.7f}".format(mon.Tb),
          "{:14.7f}".format(Tb_), "{:11.7f}".format(Tb_past_),
          "{:14.7f}".format(mon_old.Tb_hdwe_filt[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt),
          "{:14.7f}".format(mon_old.Tb_rap[i]), "{:11.7f}".format(mon.Tb_rap),
          "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:11.7f}".format(mon.Tb_f),
          "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:11.7f}".format(mon.Tb_f_rap),
          "{:14.7f}".format(mon_old.Tb_hdwe_filt_rate[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt_rate),
          "{:14.7f}".format(mon_old.Tb_f_rate[i_temp]), "{:11.7f}".format(mon.Tb_f_rate), "{:11.7f}".format(ST.Tb_f_rate),
          "{:14.7f}".format(mon_old.Tb_f_rate_rap[i]), "{:11.7f}".format(mon.Tb_f_rate_rap),
          "{:14.7f}".format(mon_old.tb_f_for_hx[i_ekf]), "{:10.7f}".format(mon.tb_f_for_hx),
          )
    return hdr

def print_volt_hist(i, i_temp, i_ekf, t, mon_old, mon, calc_temp, calc_ekf):
    hdr = "  i  time   r       rt   it   ct      re   ie  ce    sa       ib_charge             ib                    ib_hm               ib_dyn_m               ib_dyn_a_m            ib_dyn_b_m            ib_dyn_c_m            ib_dyn_T_m     ib_dyn_tau_m            ib_dyn_rstate_m       ib_dyn_lstate_m          dv_dyn_m             e_wrap_m             e_wrap_m_filt       e_wrap_m_trim         ib_hn                ib_dyn_n             ib_dyn               ib_dyn_a_n            ib_dyn_b_n            ib_dyn_c_n            ib_dyn_T_n     ib_dyn_tau_n            ib_dyn_rstate_n       ib_dyn_lstate_n         dv_dyn_n             e_wrap_n_a             e_wrap_n_b             e_wrap_n_T             e_wrap_n_tau           e_wrap_n_rate          e_wrap_n_state         e_wrap_n             e_wrap_n_filt        e_wrap               e_wrap_filt         ib_dyn                dv_dyn                   dv_hys                   soc                      dt              Tb_f                      Tb_f_rap                 voc_soc               voc                   voc_stat              voc_stat_f             soc_ekf               y_ekf"
    if calc_temp or calc_ekf:
        print(hdr)
    print("{:3d}".format(i), "{:6.3f}".format(t[i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(mon_old.sat[i]), "{:2.0f}".format(mon.sat),
          "{:11.5f}".format(mon_old.ib_charge[i]), "{:9.5f}".format(mon.ib_charge),
          "{:11.5f}".format(mon_old.ib[i]), "{:9.5f}".format(mon.ib),
          "{:11.5f}".format(mon_old.ibmh[i]), "{:8.5f}".format(mon.LoopIbAmp.ib),
          "{:11.5f}".format(mon_old.ib_dyn_m[i]), "{:8.5f}".format(mon.LoopIbAmp.ib_dyn),
          "{:12.6f}".format(mon_old.ib_dyn_a_m[i]), "{:8.6f}".format(mon.LoopIbAmp.ChargeTransfer.a),
          "{:12.6f}".format(mon_old.ib_dyn_b_m[i]), "{:8.6f}".format(mon.LoopIbAmp.ChargeTransfer.b),
          "{:12.6f}".format(mon_old.ib_dyn_c_m[i]), "{:8.6f}".format(mon.LoopIbAmp.ChargeTransfer.c),
          "{:9.3f}".format(mon_old.ib_dyn_T_m[i]), "{:5.3f}".format(mon.LoopIbAmp.ChargeTransfer.dt),
          "{:12.6f}".format(mon_old.ib_dyn_tau_m[i]), "{:8.6f}".format(mon.LoopIbAmp.ChargeTransfer.tau),
          "{:12.6f}".format(mon_old.ib_dyn_rstate_m[i]), "{:8.6f}".format(mon.LoopIbAmp.ChargeTransfer.rstate),
          "{:12.6f}".format(mon_old.ib_dyn_lstate_m[i]), "{:8.6f}".format(mon.LoopIbAmp.ChargeTransfer.state),
          "{:11.5f}".format(mon_old.dv_dyn_m[i]), "{:8.5f}".format(mon.LoopIbAmp.dv_dyn),
          "{:11.5f}".format(mon_old.e_wrap_m[i]), "{:8.5f}".format(mon.e_wrap_m),
          "{:11.5f}".format(mon_old.e_wrap_m_filt[i]), "{:8.5f}".format(mon.e_wrap_m_filt),
          "{:11.5f}".format(mon_old.e_wrap_m_trim[i]), "{:8.5f}".format(mon.e_wrap_m_trim),
          "{:11.5f}".format(mon_old.ibnh[i]), "{:8.5f}".format(mon.LoopIbNoa.ib),
          "{:11.5f}".format(mon_old.ib_dyn_n[i]), "{:8.5f}".format(mon.LoopIbNoa.ib_dyn),
          "{:10.5f}".format(mon_old.ib_dyn[i]), "{:9.5f}".format(mon.ib_dyn),
          "{:12.6f}".format(mon_old.ib_dyn_a_n[i]), "{:8.6f}".format(mon.LoopIbNoa.ChargeTransfer.a),
          "{:12.6f}".format(mon_old.ib_dyn_b_n[i]), "{:8.6f}".format(mon.LoopIbNoa.ChargeTransfer.b),
          "{:12.6f}".format(mon_old.ib_dyn_c_n[i]), "{:8.6f}".format(mon.LoopIbNoa.ChargeTransfer.c),
          "{:9.3f}".format(mon_old.ib_dyn_T_n[i]), "{:5.3f}".format(mon.LoopIbNoa.ChargeTransfer.dt),
          "{:12.6f}".format(mon_old.ib_dyn_tau_n[i]), "{:8.6f}".format(mon.LoopIbNoa.ChargeTransfer.tau),
          "{:12.6f}".format(mon_old.ib_dyn_rstate_n[i]), "{:9.6f}".format(mon.LoopIbNoa.ChargeTransfer.rstate),
          "{:12.6f}".format(mon_old.ib_dyn_lstate_n[i]), "{:9.6f}".format(mon.LoopIbNoa.ChargeTransfer.state),
          "{:11.5f}".format(mon_old.dv_dyn_n[i]), "{:8.5f}".format(mon.LoopIbNoa.dv_dyn),

          "{:12.6f}".format(mon_old.ib_wrp_a_n[i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.a),
          "{:12.6f}".format(mon_old.ib_wrp_b_n[i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.b),
          "{:12.6f}".format(mon_old.ib_wrp_T_n[i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.dt),
          "{:12.6f}".format(mon_old.ib_wrp_tau_n[i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.tau),
          "{:12.6f}".format(mon_old.ib_wrp_rate_n[i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.rate),
          "{:12.6f}".format(mon_old.ib_wrp_state_n[i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.state),

          "{:11.5f}".format(mon_old.e_wrap_n[i]), "{:8.5f}".format(mon.e_wrap_n),
          "{:11.5f}".format(mon_old.e_wrap_n_filt[i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:11.5f}".format(mon_old.e_wrap[i]), "{:8.5f}".format(mon.e_wrap),
          "{:11.5f}".format(mon_old.e_wrap_filt[i]), "{:8.5f}".format(mon.e_wrap_filt),
          "{:10.5f}".format(mon_old.ib_dyn[i]), "{:9.5f}".format(mon.ib_dyn),
          "{:13.7f}".format(mon_old.dv_dyn[i]), "{:10.7f}".format(mon.dv_dyn),
          "{:13.7f}".format(mon_old.dv_hys[i]), "{:10.7f}".format(mon.dv_hys),
          "{:13.7f}".format(mon_old.soc[i]), "{:10.7f}".format(mon.soc),
          "{:9.3f}".format(mon_old.dt[i]), "{:5.3f}".format(mon.dt),
          "{:14.7f}".format(mon_old.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(mon_old.Tb_f_rap[i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:11.5f}".format(mon_old.voc_soc[i]), "{:9.5f}".format(mon.voc_soc),
          "{:11.5f}".format(mon_old.voc[i]), "{:9.5f}".format(mon.voc),
          "{:11.5f}".format(mon_old.voc_stat[i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(mon_old.z[i_ekf]), "{:9.5f}".format(mon.voc_stat_f),
          "{:11.5f}".format(mon_old.soc_ekf[i]), "{:9.5f}".format(mon.soc_ekf),
          "{:11.5f}".format(mon_old.y_ekf[i]), "{:9.5f}".format(mon.y_ekf),
          )
    return hdr

def save_clean_file(mon_ver, csv_file, unit_key):
    default_header_str = "unit,               hm,                  cTime,        dt,       sat,sel,mod,\
      Tb,Tb_rap,Tb_f,Tb_f_rap,Tb_f_rate,Tb_f_rate_rap, vb,  ib,  ib_dyn, ioc,  voc_soc,    vsat,dv_dyn,voc_stat,voc_stat_f,voc_ekf,     y_ekf,    soc_s,soc_ekf,soc,ib_lag,voc_soc_new,"
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
            s += "{:7.3f},".format(mon_ver.ib_dyn[i])
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


