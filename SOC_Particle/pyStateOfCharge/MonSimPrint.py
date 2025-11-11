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
import Globals as G

def prn_soc_debug(OPT, leader="", time=None, i_temp=None, mon=None, sim=None):
    execute = False
    if not execute:
        return
    else:
        if OPT.request_history == 2:  # soc
            if G.i > 0:
                d_dq = OPT.mon_run.delta_q[G.i] - OPT.mon_run.delta_q[G.i - 1]
            else:
                d_dq = OPT.mon_run.delta_q[G.i + 1] - OPT.mon_run.delta_q[G.i]
            if time is not None:
                print("time {:7.3f}".format(time), end='')
            print(" " * 103 + leader, end='')
            print(
                  "{:14.7f}".format(OPT.mon_run.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
                  "{:14.7f}".format(OPT.mon_run.Tb_f_rap[G.i]), "{:10.7f}".format(mon.Tb_f_rap),
                  "{:12.4f}".format(d_dq), "{:11.4f}".format(mon.d_delta_q),
                  "{:12.4f}".format(OPT.mon_run.delta_q[G.i]), "{:11.4f}".format(mon.delta_q),
                  "{:12.1f}".format(OPT.mon_run.qcrs[G.i]), "{:9.1f}".format(mon.q_cap_rated_scaled),
                  "{:12.1f}".format(OPT.mon_run.q_capacity[G.i]), "{:9.1f}".format(mon.q_capacity),
            )
        elif OPT.request_history == 3:  # soc_s
            if time is not None:
                print("time {:7.3f}".format(time), end='')
            print(" " * 375 + leader, end='')
            print(
                "{:11.8f}".format(OPT.mon_run.soc_s[G.i]), "{:8.7f}".format(sim.soc),
                "{:14.7f}".format(OPT.sim_run.Tb_f_s[G.i]), "{:11.7f}".format(sim.Tb_f),
                "{:14.4f}".format(OPT.sim_run.d_delta_q_s[G.i]), "{:11.4f}".format(sim.d_delta_q),
                "{:14.4f}".format(OPT.sim_run.dq_s[G.i]), "{:11.4f}".format(sim.delta_q), "{:2.0f}".format(sim.reset_temp_past),
            )
        elif OPT.request_history == 4:  # temp
            if time is not None:
                print("time {:7.3f}".format(time), end='')
            print(" " * 75 + leader, end='')
            print(
        "{:14.7f}".format(OPT.mon_run.Tb_hdwe_filt[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt),
        "{:14.7f}".format(OPT.mon_run.Tb_rap[G.i]), "{:11.7f}".format(mon.Tb_rap),
        "{:14.7f}".format(OPT.mon_run.Tb_f[i_temp]), "{:11.7f}".format(mon.Tb_f),
        "{:14.7f}".format(OPT.mon_run.Tb_f_rap[G.i]), "{:11.7f}".format(mon.Tb_f_rap),
        "{:14.7f}".format(OPT.mon_run.Tb_hdwe_filt_rate[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt_rate),
        "{:14.7f}".format(OPT.mon_run.Tb_f_rate[i_temp]), "{:11.7f}".format(mon.Tb_f_rate),
        "{:14.7f}".format(OPT.mon_run.Tb_f_rate_rap[G.i]), "{:11.7f}".format(mon.Tb_f_rate_rap),
            )

def print_hist(OPT, SN, i_temp, i_ekf, t, mon, calc_temp, calc_ekf, sim):
    hdr = None
    match OPT.run_type:
        case 'RunSim':
            match OPT.request_history:
                case 0:
                    hdr = ''
                case 1:
                    hdr = print_ekf_RunSim(SN, i_temp, i_ekf, t, mon, calc_ekf, calc_temp)
                case 2:
                    hdr = print_soc_RunSim(SN, i_temp, t, mon, calc_temp, i_ekf, calc_ekf)
                case 3:
                    hdr = print_soc_s_RunSim(SN, i_temp, t, mon, calc_temp, sim, i_ekf, calc_ekf)
                case 4:
                    hdr = print_temp_RunSim(SN, i_temp, t, mon, calc_temp, i_ekf, calc_ekf)
                case 5:
                    hdr = print_volt_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf)
        case 'HistSim':
            match OPT.request_history:
                case 0:
                    hdr = ''
                # case 1:
                #     hdr = print_ekf_HistSim(SN, _temp, i_ekf, t, mon, calc_ekf, calc_temp)
                # case 2:
                #     hdr = print_soc_HistSim(SN, i_temp, t, mon, calc_temp)
                # case 3:
                #     hdr = print_soc_s_HistSim(SN, i_temp, t, SN.mon_run, mon, calc_temp, sim, i_ekf, calc_ekf, SN)
                # case 4:
                #     hdr = print_temp_HistSim(SN, i_temp, t, mon, calc_temp, Tb, Tb_past, SN, i_ekf, calc_ekf)
                case 5:
                    hdr = print_volt_HistSim(SN, i_temp, i_ekf, t, mon, calc_temp, calc_ekf)
    return hdr

def print_ekf_RunSim(SN, i_temp, i_ekf, t, mon, calc_ekf, calc_temp):
    hdr = "  i  time     r r_t  i_e  r_e  c_e   dt_ekf         sa      ib_charge             soc                    soc_ekf                 y_ekf                voc_ekf                Tb_f                    x_prior             fr     Tb_f_rap                x                       tb_f_for_hx             x_for_hx                  hx                       voc_stat_f            z                   z_ekf       P                              P_post                       P_prior                       H                      R                     S                    K                          x_post                 f_rstate             f_lstate              f_a                    f_b                    f_c                  f_tau                     f_T"
    i_ekf = max(i_ekf, 0)
    if calc_temp or calc_ekf:
        print(hdr)
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset), "{:2.0f}".format(mon.reset_temp),
          "{:4d}".format(i_ekf), "{:4d}".format(mon.reset_ekf), "{:4d}".format(calc_ekf),
          "{:9.3f}".format(SN.mon_run.dt_ekf[i_ekf]), "{:5.3f}".format(mon.dt_eframe),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:10.5f}".format(SN.mon_run.ib_charge[G.i]), "{:9.5f}".format(mon.ib_charge),
          "{:13.7f}".format(SN.mon_run.soc[G.i]), "{:10.7f}".format(mon.soc),
          "{:11.7f}".format(SN.mon_run.soc_ekf[G.i]), "{:9.7f}".format(mon.soc_ekf),
          "{:11.5f}".format(SN.mon_run.y_ekf[G.i]), "{:9.5f}".format(mon.y_ekf),
          "{:11.5f}".format(SN.mon_run.voc_ekf[G.i]), "{:9.5f}".format(mon.voc_ekf),
          "{:14.7f}".format(SN.mon_run.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:11.7f}".format(SN.mon_run.x_prior[i_ekf]), "{:9.7f}".format(mon.x_prior), "{:2.0f}".format(SN.mon_run.frz[i_ekf]),
          "{:14.7f}".format(SN.mon_run.Tb_f_rap[G.i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:11.7f}".format(SN.mon_run.x[i_ekf]), "{:9.7f}".format(mon.x),
          "{:14.7f}".format(SN.mon_run.tb_f_for_hx[i_ekf]), "{:10.7f}".format(mon.tb_f_for_hx),
          "{:11.7f}".format(SN.mon_run.x_for_hx[i_ekf]), "{:9.7f}".format(mon.x_for_hx),
          "{:14.5f}".format(SN.mon_run.hx[i_ekf]), "{:9.5f}".format(mon.hx),
          "{:14.5f}".format(SN.mon_run.z[i_ekf]), "{:9.5f}".format(mon.voc_stat_f),
          "{:11.5f}".format(SN.mon_run.z[i_ekf]), "{:9.5f}".format(mon.z), "{:9.5f}".format(mon.z_ekf),
          "{:16.11f}".format(SN.mon_run.P[i_ekf]), "{:12.11f}".format(mon.P),
          "{:16.11f}".format(SN.mon_run.P_post[i_ekf]), "{:12.11f}".format(mon.P_post),
          "{:14.11f}".format(SN.mon_run.P_prior[i_ekf]), "{:12.11f}".format(mon.P_prior),
          "{:11.7f}".format(SN.mon_run.H[i_ekf]), "{:9.7f}".format(mon.H),
          "{:11.6f}".format(SN.mon_run.R[i_ekf]), "{:9.6f}".format(mon.R),
          "{:11.6f}".format(SN.mon_run.S[i_ekf]), "{:9.6f}".format(mon.S),
          "{:13.9f}".format(SN.mon_run.K[i_ekf]), "{:10.9f}".format(mon.K),
          "{:12.7f}".format(SN.mon_run.x_post[i_ekf]), "{:9.7f}".format(mon.x_post),
          "{:11.5f}".format(SN.mon_run.voc_stat_f_rstate[i_ekf]), "{:8.5f}".format(mon.voc_stat_f_rstate),
          "{:11.5f}".format(SN.mon_run.voc_stat_f_lstate[i_ekf]), "{:8.5f}".format(mon.voc_stat_f_lstate),
          "{:12.6f}".format(SN.mon_run.voc_stat_f_a[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_a),
          "{:12.6f}".format(SN.mon_run.voc_stat_f_b[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_b),
          "{:12.6f}".format(SN.mon_run.voc_stat_f_c[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_c),
          "{:12.6f}".format(SN.mon_run.voc_stat_f_tau[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_tau),
          "{:12.6f}".format(SN.mon_run.voc_stat_f_T[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_T),
          )
    return hdr

def print_soc_RunSim(SN, i_temp, t, mon, calc_temp, i_ekf, calc_ekf):
    hdr = "  i  time     r       rt   it   ct      re   ie  ce    sa     ib_charge            soc                     dt                G.i * dt * coul_eff    Tb_f                      Tb_f_rap                    ddq                  delq                       qcrs                   q_cap                  Tb                       Tb_f_rate"
    if calc_temp:
        print(hdr)
    if G.i > 0:
        d_dq = SN.mon_run.delta_q[G.i]-SN.mon_run.delta_q[G.i-1]
    else:
        d_dq = SN.mon_run.delta_q[G.i+1]-SN.mon_run.delta_q[G.i]
    i_dt_old = SN.mon_run.dt[G.i] * SN.mon_run.ib_charge[G.i]
    i_dt_new = mon.dt * mon.ib_charge
    coul_eff = 0.9985
    if mon.ib_charge > 0:
        i_dt_old *= coul_eff
        i_dt_new *= coul_eff
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:10.5f}".format(SN.mon_run.ib_charge[G.i]), "{:9.5f}".format(mon.ib_charge),
          "{:11.7f}".format(SN.mon_run.soc[G.i]), "{:8.7f}".format(mon.soc),
          "{:9.3f}".format(SN.mon_run.dt[G.i]), "{:5.3f}".format(mon.dt),
          "{:12.4f}".format(i_dt_old), "{:9.4f}".format(i_dt_new),
          "{:14.7f}".format(SN.mon_run.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f_rap[G.i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:12.4f}".format(d_dq), "{:11.4f}".format(mon.d_delta_q),
          "{:12.4f}".format(SN.mon_run.delta_q[G.i]), "{:11.4f}".format(mon.delta_q),
          "{:12.1f}".format(SN.mon_run.qcrs[G.i]), "{:9.1f}".format(mon.q_cap_rated_scaled),
          "{:12.1f}".format(SN.mon_run.q_capacity[G.i]), "{:9.1f}".format(mon.q_capacity),
          "{:14.7f}".format(SN.mon_run.Tb[i_temp]), "{:10.7f}".format(mon.Tb),
          "{:12.7f}".format(SN.mon_run.Tb_f_rate[i_temp]), "{:10.7f}".format(mon.Tb_f_rate),
         )
    return hdr

def print_soc_s_RunSim(SN, i_temp, t, mon, calc_temp, sim, i_ekf, calc_ekf):
    hdr = "  i  time     r       rt   it   ct      re   ie  ce    sa       sa_s     dt               dt_s            ib_in_s               ib_s                  ib_fut       ib_dyn_s_rstate         ib_dyn_s_lstate          ib_dyn_s       ib_dyn_s_init     ib_dyn           ib_dyn_init      dv_hys_s              ib_charge_s            ioc_s                soc                      delq                    i * dt_s * coul_eff    soc_s                      Tb_f_s                       d_delq_s                delq_s                     qcrs                   q_cap                  q_cap_s                Tb_f_s                    Tb_f                      Tb_f_rap                 Tb_f_rate               vb                    vb_s                  voc_stat              voc_stat_s            voc_s                  dv_dyn_s             vsat                 "
    if calc_temp:
        print(hdr)
    # if G.i > 0:
    #     d_dq_s = SN.sim_run.dq_s[G.i]-SN.sim_run.dq_s[G.i-1]
    # else:
    #     d_dq_s = SN.sim_run.dq_s[G.i+1]-SN.sim_run.dq_s[G.i]
    i_dt_old = SN.sim_run.dt_s[G.i] * SN.sim_run.ib_charge_s[G.i]
    i_dt_new = sim.dt * sim.ib_charge
    coul_eff = 0.9985
    if sim.ib_charge > 0:
        i_dt_old *= coul_eff
        i_dt_new *= coul_eff
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:5.0f}".format(SN.sim_run.sat_s[G.i]), "{:2.0f}".format(sim.sat),
          "{:9.3f}".format(SN.mon_run.dt[G.i]), "{:5.3f}".format(mon.dt),
          "{:9.3f}".format(SN.sim_run.dt_s[G.i]), "{:5.3f}".format(sim.dt),
          "{:12.5f}".format(SN.sim_run.ib_in_s[G.i]), "{:9.5f}".format(sim.ib_in),
          "{:12.6f}".format(SN.sim_run.ib_s[G.i]), "{:10.6f}".format(sim.ib), "{:10.6f}".format(sim.ib_fut),
          "{:12.6f}".format(SN.sim_run.ib_dyn_s_rstate[G.i]), "{:10.6f}".format(sim.ChargeTransfer.rstate),
          "{:12.6f}".format(SN.sim_run.ib_dyn_s_lstate[G.i]), "{:10.6f}".format(sim.ChargeTransfer.state),
          "{:12.5f}".format(SN.sim_run.ib_dyn_s[G.i]), "{:9.5f}".format(sim.ib_dyn), "{:9.5f}".format(SN.ib_dyn_s_init),
          "{:12.5f}".format(SN.mon_run.ib_dyn[G.i]), "{:9.5f}".format(mon.ib_dyn), "{:9.5f}".format(SN.ib_dyn[0]),
          "{:12.5f}".format(SN.sim_run.dv_hys_s[G.i]), "{:9.5f}".format(sim.dv_hys),
          "{:12.5f}".format(SN.sim_run.ib_charge_s[G.i]), "{:9.5f}".format(sim.ib_charge),
          "{:12.5f}".format(SN.sim_run.ioc_s[G.i]), "{:9.5f}".format(sim.ioc),
          "{:11.7f}".format(SN.mon_run.soc[G.i]), "{:8.7f}".format(mon.soc),
          "{:14.4f}".format(SN.mon_run.delta_q[G.i]), "{:11.4f}".format(mon.delta_q),
          "{:12.4f}".format(i_dt_old), "{:9.4f}".format(i_dt_new),
          "{:11.8f}".format(SN.mon_run.soc_s[G.i]), "{:9.8f}".format(sim.soc),
          "{:14.8f}".format(SN.sim_run.Tb_f_s[G.i]), "{:11.8f}".format(sim.Tb_f),
          "{:14.5f}".format(SN.sim_run.d_delta_q_s[G.i]), "{:11.5f}".format(sim.d_delta_q),
          "{:14.5f}".format(SN.sim_run.dq_s[G.i]), "{:11.5f}".format(sim.delta_q),
          "{:12.2f}".format(SN.mon_run.qcrs[G.i]), "{:9.2f}".format(mon.q_cap_rated_scaled),
          "{:12.2f}".format(SN.mon_run.q_capacity[G.i]), "{:9.2f}".format(mon.q_capacity),
          "{:12.2f}".format(SN.sim_run.qcap_s[G.i]), "{:9.2f}".format(sim.q_capacity),
          "{:14.7f}".format(SN.sim_run.Tb_f_s[G.i]), "{:10.7f}".format(sim.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f_rap[G.i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:12.7f}".format(SN.mon_run.Tb_f_rate[i_temp]), "{:10.7f}".format(mon.Tb_f_rate),
          "{:11.5f}".format(SN.mon_run.vb[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.sim_run.vb_s[G.i]), "{:9.5f}".format(sim.vb),
          "{:11.5f}".format(SN.mon_run.voc_stat[G.i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(SN.sim_run.voc_stat_s[G.i]), "{:9.5f}".format(sim.voc_stat),
          "{:11.5f}".format(SN.sim_run.voc_s[G.i]), "{:9.5f}".format(sim.voc),
          "{:11.5f}".format(SN.sim_run.dv_dyn_s[G.i]), "{:9.5f}".format(sim.dv_dyn),
          "{:11.5f}".format(SN.mon_run.vsat[G.i]), "{:9.5f}".format(mon.vsat),
          )
    if G.i == 2:
        pass
    return hdr

def print_temp_RunSim(SN, i_temp, t, mon, calc_temp, i_ekf, calc_ekf):
    hdr = "  i  time     r       rt   it   ct      re   ie  ce     Tt       Tb_hdwe                    Tb                         Tb_past_  Tb_hdwe_filt     Tb_rap                     Tb_f                       Tb_f_rap                    Tb_h_f_r                   Tb_f_rate                              Tb_f_rate_rap             tb_f_for_hx"
    if calc_temp:
        print(hdr)
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:9.3f}".format(SN.mon_run.Tt[i_temp]),
          "{:13.7f}".format(SN.mon_run.Tb_hdwe[i_temp]), "{:11.7f}".format(mon.Tb_hdwe),
          "{:14.7f}".format(SN.mon_run.Tb[i_temp]), "{:11.7f}".format(mon.Tb),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_filt[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt),
          "{:14.7f}".format(SN.mon_run.Tb_rap[G.i]), "{:11.7f}".format(mon.Tb_rap),
          "{:14.7f}".format(SN.mon_run.Tb_f[i_temp]), "{:11.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f_rap[G.i]), "{:11.7f}".format(mon.Tb_f_rap),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_filt_rate[i_temp]), "{:11.7f}".format(mon.Tb_hdwe_filt_rate),
          "{:14.7f}".format(SN.mon_run.Tb_f_rate[i_temp]), "{:11.7f}".format(mon.Tb_f_rate), "{:11.7f}".format(SN.Tb_f_rate),
          "{:14.7f}".format(SN.mon_run.Tb_f_rate_rap[G.i]), "{:11.7f}".format(mon.Tb_f_rate_rap),
          "{:14.7f}".format(SN.mon_run.tb_f_for_hx[i_ekf]), "{:10.7f}".format(mon.tb_f_for_hx),
          )
    return hdr

def print_volt_HistSim(SN, i_temp, i_ekf, t, mon, calc_temp, calc_ekf):
    hdr = "  i   time r    rt it     ct   re ie     ce   sa        Tb_f                     vb_f                   ib_f                  ib_nh_f               ib_mh_f               ib_dyn_m              e_wrap_m_filt        e_wrap_m_trim       ib_hn                 ib_dyn_n               e_wrap_n_filt        e_wrap_f             soc                        dt                 Tb_f                     vb_f                  ib_dyn                voc_f     voc         voc_stat_f             soc_ekf"
    if G.i % 10 == 0:
        print(hdr)
    print("{:4d}".format(G.i), "{:4.0f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:4d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:4d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:11.7f}".format(mon.Tb_f),
          "{:11.5f}".format(SN.mon_run.vb_f[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.ib_f[G.i]), "{:9.5f}".format(mon.ib),
          "{:11.5f}".format(SN.mon_run.ibnh_f[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib),
          "{:11.5f}".format(SN.mon_run.ibmh_f[G.i]), "{:9.5f}".format(mon.LoopIbAmp.ib),
          "{:11.5f}".format(SN.mon_run.ib_dyn_m[G.i]), "{:9.5f}".format(mon.LoopIbAmp.ib_dyn),
          "{:11.5f}".format(SN.mon_run.e_wm_f[G.i]), "{:8.5f}".format(mon.e_wrap_m_filt),
          "{:11.5f}".format(SN.mon_run.e_wm_t[G.i]), "{:8.5f}".format(mon.e_wrap_m_trim),
          "{:11.5f}".format(SN.mon_run.ibnh_f[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib),
          "{:11.5f}".format(SN.mon_run.ib_dyn_n[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib_dyn),
          "{:11.5f}".format(SN.mon_run.e_wn_f[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:11.5f}".format(SN.mon_run.e_wrap_f[G.i]), "{:8.5f}".format(mon.e_wrap_filt),
          "{:13.7f}".format(SN.mon_run.soc[G.i]), "{:10.7f}".format(mon.soc),
          "{:11.3f}".format(SN.mon_run.dt[G.i]), "{:8.3f}".format(mon.dt),
          "{:14.7f}".format(SN.mon_run.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:11.5f}".format(SN.mon_run.vb_f[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.ib_dyn[G.i]), "{:9.5f}".format(mon.ib_dyn),
          "{:11.7f}".format(SN.mon_run.voc_f[G.i]), "{:10.7f}".format(mon.voc),
          "{:11.7f}".format(SN.mon_run.z[i_ekf]), "{:10.7f}".format(mon.voc_stat_f),
          "{:11.5f}".format(SN.mon_run.soc_ekf[G.i]), "{:9.5f}".format(mon.soc_ekf),
          )
    return hdr

def print_volt_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf):
    hdr = "  i   time     r       rt   it   ct      re   ie  ce    sa      vb                        ib_charge             ib                     ibmh      ibmm     ib_amp       ib_dyn_m      ib_dyn_m_init     ib_dyn_T_m     ib_dyn_tau_m            ib_dyn_rstate_m         ib_dyn_lstate_m        vb                     dv_dyn_m            voc                   voc_soc                e_wrap_m             e_wrap_m_filt        e_wrap_m_trim     init        ibnh      ibnm     ib_noa       ib_dyn_n      ib_dyn_n_init     ib_dyn_T_n     ib_dyn_tau_n            dv_dyn_n             e_wrap_n             e_wrap_n_filt        ib_dyn_n             ib_dyn                 ib_dyn_T_n     ib_dyn_tau_n           ib_dyn_rstate_n         ib_dyn_lstate_n          dv_dyn_n              e_wrap_n_T             e_wrap_n_tau           e_wrap_n_rate          e_wrap_n_state         e_wrap_n             e_wrap_n_filt      ib                     e_wrap               e_wrap_filt          ib_dyn_rstate          ib_dyn_lstate         ib_dyn                dv_dyn                dv_hys                soc                      dt              Tb_f                      Tb_f_rap                 voc_soc               voc                    voc_stat                voc_stat_s              voc_stat_f                soc_ekf               y_ekf"
    if calc_temp or calc_ekf:
        print(hdr)
    print("{:4d}".format(G.i), "{:8.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:13.7f}".format(SN.mon_run.vb[G.i]), "{:11.7f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.ib_charge[G.i]), "{:9.5f}".format(mon.ib_charge),
          "{:11.5f}".format(SN.mon_run.ib[G.i]), "{:9.5f}".format(mon.ib),
          "{:11.5f}".format(SN.mon_run.ibmh[G.i]), "{:9.5f}".format(SN.mon_run.ibmm[G.i]), "{:9.5f}".format(mon.LoopIbAmp.ib),
          "{:11.5f}".format(SN.mon_run.ib_dyn_m[G.i]), "{:9.5f}".format(mon.LoopIbAmp.ib_dyn), "{:9.5f}".format(SN.LoopAmp.ib_init),
          "{:9.3f}".format(SN.mon_run.ib_dyn_T_m[G.i]), "{:5.3f}".format(mon.LoopIbAmp.ChargeTransfer.dt),
          "{:12.6f}".format(SN.mon_run.ib_dyn_tau_m[G.i]), "{:8.6f}".format(mon.LoopIbAmp.ChargeTransfer.tau),
          "{:12.6f}".format(SN.mon_run.ib_dyn_rstate_m[G.i]), "{:10.6f}".format(mon.LoopIbAmp.ChargeTransfer.rstate),
          "{:12.6f}".format(SN.mon_run.ib_dyn_lstate_m[G.i]), "{:10.6f}".format(mon.LoopIbAmp.ChargeTransfer.state),
          "{:11.5f}".format(SN.mon_run.vb[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.dv_dyn_m[G.i]), "{:8.5f}".format(mon.LoopIbAmp.dv_dyn),
          "{:11.5f}".format(SN.mon_run.voc[G.i]), "{:9.5f}".format(mon.voc),
          "{:11.5f}".format(SN.mon_run.voc_soc[G.i]), "{:9.5f}".format(mon.voc_soc),
          "{:11.5f}".format(SN.mon_run.e_wrap_m[G.i]), "{:8.5f}".format(mon.e_wrap_m),
          "{:11.5f}".format(SN.mon_run.e_wrap_m_filt[G.i]), "{:8.5f}".format(mon.e_wrap_m_filt),
          "{:11.5f}".format(SN.mon_run.e_wrap_m_trim[G.i]), "{:8.5f}".format(mon.e_wrap_m_trim), "{:8.5f}".format(SN.e_wrap_m_trim_init),
          "{:11.5f}".format(SN.mon_run.ibnh[G.i]),  "{:9.5f}".format(SN.mon_run.ibnm[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib),
          "{:11.5f}".format(SN.mon_run.ib_dyn_n[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib_dyn), "{:9.5f}".format(SN.LoopAmp.ib_init),
          "{:9.3f}".format(SN.mon_run.ib_dyn_T_n[G.i]), "{:5.3f}".format(mon.LoopIbNoa.ChargeTransfer.dt),
          "{:12.6f}".format(SN.mon_run.ib_dyn_tau_n[G.i]), "{:8.6f}".format(mon.LoopIbNoa.ChargeTransfer.tau),
          "{:11.5f}".format(SN.mon_run.dv_dyn_n[G.i]), "{:8.5f}".format(mon.LoopIbNoa.dv_dyn),
          "{:11.5f}".format(SN.mon_run.e_wrap_n[G.i]), "{:8.5f}".format(mon.e_wrap_n),
          "{:11.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:11.5f}".format(SN.mon_run.ib_dyn_n[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib_dyn),
          "{:10.5f}".format(SN.mon_run.ib_dyn[G.i]), "{:9.5f}".format(mon.ib_dyn),
          "{:9.3f}".format(SN.mon_run.ib_dyn_T_n[G.i]), "{:5.3f}".format(mon.LoopIbNoa.ChargeTransfer.dt),
          "{:12.6f}".format(SN.mon_run.ib_dyn_tau_n[G.i]), "{:8.6f}".format(mon.LoopIbNoa.ChargeTransfer.tau),
          "{:12.6f}".format(SN.mon_run.ib_dyn_rstate_n[G.i]), "{:10.6f}".format(mon.LoopIbNoa.ChargeTransfer.rstate),
          "{:12.6f}".format(SN.mon_run.ib_dyn_lstate_n[G.i]), "{:10.6f}".format(mon.LoopIbNoa.ChargeTransfer.state),
          "{:11.5f}".format(SN.mon_run.dv_dyn_n[G.i]), "{:9.5f}".format(mon.LoopIbNoa.dv_dyn),
          "{:12.6f}".format(SN.mon_run.ib_wrp_T_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.dt),
          "{:12.6f}".format(SN.mon_run.ib_wrp_tau_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.tau),
          "{:12.6f}".format(SN.mon_run.ib_wrp_rate_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.rate),
          "{:12.6f}".format(SN.mon_run.ib_wrp_state_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.state),
          "{:11.5f}".format(SN.mon_run.e_wrap_n[G.i]), "{:8.5f}".format(mon.e_wrap_n),
          "{:11.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:11.5f}".format(SN.mon_run.ib[G.i]), "{:9.5f}".format(mon.ib),
          "{:11.5f}".format(SN.mon_run.e_wrap[G.i]), "{:8.5f}".format(mon.e_wrap),
          "{:11.5f}".format(SN.mon_run.e_wrap_filt[G.i]), "{:8.5f}".format(mon.e_wrap_filt),
          "{:12.6f}".format(SN.mon_run.ib_dyn_lstate[G.i]), "{:9.6f}".format(mon.ib_dyn_lstate),
          "{:12.6f}".format(SN.mon_run.ib_dyn_rstate[G.i]), "{:9.6f}".format(mon.ib_dyn_rstate),
          "{:10.5f}".format(SN.mon_run.ib_dyn[G.i]), "{:9.5f}".format(mon.ib_dyn),
          "{:11.5f}".format(SN.mon_run.dv_dyn[G.i]), "{:9.5f}".format(mon.dv_dyn),
          "{:11.5f}".format(SN.mon_run.dv_hys[G.i]), "{:9.5f}".format(mon.dv_hys),
          "{:13.7f}".format(SN.mon_run.soc[G.i]), "{:10.7f}".format(mon.soc),
          "{:9.3f}".format(SN.mon_run.dt[G.i]), "{:5.3f}".format(mon.dt),
          "{:14.7f}".format(SN.mon_run.Tb_f[i_temp]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f_rap[G.i]), "{:10.7f}".format(mon.Tb_f_rap),
          "{:11.5f}".format(SN.mon_run.voc_soc[G.i]), "{:9.5f}".format(mon.voc_soc),
          "{:11.5f}".format(SN.mon_run.voc[G.i]), "{:9.5f}".format(mon.voc),
          "{:11.5f}".format(SN.mon_run.voc_stat[G.i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(SN.sim_run.voc_stat_s[G.i]), "{:9.5f}".format(sim.voc_stat),
          "{:11.5f}".format(SN.mon_run.z[i_ekf]), "{:9.6f}".format(mon.voc_stat_f),
          "{:11.5f}".format(SN.mon_run.soc_ekf[G.i]), "{:9.5f}".format(mon.soc_ekf),
          "{:11.5f}".format(SN.mon_run.y_ekf[G.i]), "{:9.5f}".format(mon.y_ekf),
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
            s += "{:7.3f},".format(mon_ver.time[i] + mon_ver.time_run)
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


