# MonSimPrint:  Debug prints for MonSim
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

""" Python model of what's installed on the Particle Photon.  Includes
a monitor object (MON) and a simulation object (SIM).   The monitor is
the EKF and Coulomb Counter.   The SIM is a battery model, that also has a
Coulomb Counter built in."""

from datetime import datetime, timedelta
# noinspection PyPep8Naming
import Globals as G
from Colors import Colors
count_since_last_header = 0
vv_warning_printed = False
HDR_SPREAD = 10


# noinspection PyPep8Naming
def prn_soc_debug(OPT, leader="", time=None, i_temp=None, mon=None, sim=None):
    execute = False
    # execute = True
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
                  "{:14.7f}".format(OPT.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
                  "{:14.7f}".format(OPT.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
                  "{:12.4f}".format(d_dq), "{:11.4f}".format(mon.d_delta_q),
                  "{:12.4f}".format(OPT.mon_run.delta_q[G.i]), "{:11.4f}".format(mon.delta_q),
                  "{:12.1f}".format(OPT.mon_run.qcrs[G.i]), "{:9.1f}".format(mon.q_cap_rated_scaled),
                  "{:12.1f}".format(OPT.mon_run.q_capacity[G.i]), "{:9.1f}".format(mon.q_capacity),
            )
        elif OPT.request_history == 3:  # soc_s
            if time is not None:
                print("time {:7.3f}".format(time), end='')
            print(" " * 522 + leader, end='')
            print(
                "{:11.8f}".format(OPT.mon_run.soc_s[G.i]), "{:9.8f}".format(sim.soc),
                "{:14.8f}".format(OPT.sim_run.Tb_f_s[G.i]), "{:11.8f}".format(sim.Tb_f),
                "{:15.6f}".format(OPT.sim_run.d_delta_q_s[G.i]), "{:13.6f}".format(sim.d_delta_q),
                "{:15.6f}".format(OPT.sim_run.delta_q_s[G.i]), "{:13.6f}".format(sim.delta_q), "{:2.0f}".format(sim.reset_temp_past),
            )
        elif OPT.request_history == 4:  # temp
            if time is not None:
                print("time {:7.3f}".format(time), end='')
            print(" " * 75 + leader, end='')
            print(
        "{:14.7f}".format(OPT.mon_run.Tb_hdwe_f[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f),
        "{:14.7f}".format(OPT.mon_run.Tb_rap[G.i]), "{:11.7f}".format(mon.Tb_rap),
        "{:14.7f}".format(OPT.mon_run.Tb_f[G.i]), "{:11.7f}".format(mon.Tb_f),
        "{:14.7f}".format(OPT.mon_run.Tb_f[G.i]), "{:11.7f}".format(mon.Tb_f),
        "{:14.7f}".format(OPT.mon_run.Tb_hdwe_f_rate[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f_rate),
        "{:14.7f}".format(OPT.mon_run.Tb_f_rate[G.i]), "{:11.7f}".format(mon.Tb_f_rate),
        "{:14.7f}".format(OPT.mon_run.Tb_f_rate_rap[G.i]), "{:11.7f}".format(mon.Tb_f_rate_rap),
            )


# noinspection PyPep8Naming
def print_hist(OPT, SN, i_temp, i_ekf, t, mon, calc_temp, calc_ekf, sim):
    hdr = None
    match OPT.run_type:
        case 'RunSim':
            match OPT.request_history:
                case 0:
                    hdr = ''
                case 1:  # request_history for ekf
                    hdr = print_ekf_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_ekf, calc_temp)
                case 2:  # request_history for soc
                    hdr = print_soc_RunSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf)
                case 3:  # request_history for soc_s
                    hdr = print_soc_s_RunSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf)
                case 4:  # request_history for temp
                    hdr = print_temp_RunSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf)
                case 5:  # request_history for volt all
                    hdr = print_volt_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf)
                case 6:  # request_history for kf
                    hdr = print_kf_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf)
                case 7:  # request_history for dyn_m
                    hdr = print_dyn_m_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf)
                case 8:  # request_history for vb_wrap
                    hdr = print_vb_wrap_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf)
                case 9:  # request_history for dyn_n
                    hdr = print_dyn_n_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf)
        case 'HistSim':
            match OPT.request_history:
                case 0:
                    hdr = ''
                # case 1:
                #     hdr = print_ekf_HistSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf)
                # case 2:
                #     hdr = print_soc_HistSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf)
                case 3:
                    hdr = print_soc_s_HistSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf)
                # case 4:
                #     hdr = print_temp_HistSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf)
                case 5:
                    hdr = print_volt_HistSim(SN, i_temp, i_ekf, t, mon, calc_temp, calc_ekf)
    return hdr

# 7
# noinspection PyPep8Naming,PyUnusedLocal
def print_dyn_m_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf):
    global count_since_last_header, vv_warning_printed
    if not hasattr(SN.mon_run, 'ib_amp_lo') or SN.mon_run.ib_amp_lo is None:
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1-vv3 run.  Not printing print_dyn_m_RunSim  (request_hist_in=7)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i   time     r       rt   rk   it   ct      re   ie  ce    reset  reset_temp     reset_all_faults   soft_reset  soft_reset_sim  init_mon     init_sim     sa      dt                vb                           ibmh                        ibmm                    ib_amp_lo     ib_amp_hi      ib_lo_active     dis_amp_flt   dt                    ib_amp                    ib_dyn_T_m          ib_dyn_rstate_m                ib_dyn_lstate_m                      ib_dyn_m                vb                     dv_dyn_m             vb_m                      voc_m                     voc_soc               voc_soc_m                   e_wrap_m                  e_wrap_m_trim        e_wrap_trimmed_m         e_wrap_m_T           e_wrap_m_rate            e_wrap_m_reset       e_wrap_m_state         e_wrap_m_filt        voc_soc                   voc_stat              vsat                      ib                      soc                       ewmhi_thr            ewmlo_thr           e_wrap_m_flt   e_wrap_m_fa    disable_amp_fault      ib_amp_lo     e_wrap_m_reset  fltw   falw"
    if (calc_temp or calc_ekf) and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.red, end='')
    elif mon.reset_temp:
        print(Colors.fg.orange, end='')
    print("{:4d}".format(G.i), "{:8.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(mon.reset_kf), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:7d}".format(bool(SN.mon_run.reset[G.i])),
          "{:7d}".format(bool(SN.mon_run.reset_temp[G.i])),
          "{:14d}".format(bool(SN.mon_run.reset_all_faults[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset_sim[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_mon[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_sim[G.i])),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:9.4f}".format(SN.mon_run.dt[G.i]), "{:7.4f}".format(mon.dt),
          "{:13.7f}".format(SN.mon_run.vb[G.i]), "{:11.7f}".format(mon.vb),
          "{:14.5f}".format(SN.mon_run.ib_amp_hdwe[G.i]), "{:12.5f}".format(mon.ib_amp_hdwe),
          "{:14.5f}".format(SN.mon_run.ib_amp_model[G.i]), "{:12.5f}".format(mon.ib_amp_model),
          "{:7d}".format(bool(SN.mon_run.ib_amp_lo[G.i])), "{:2d}".format(bool(mon.ib_amp_lo)),
          "{:7d}".format(bool(SN.mon_run.ib_amp_hi[G.i])), "{:2d}".format(bool(mon.ib_amp_hi)),
          "{:18d}".format(bool(SN.mon_run.ib_lo_active[G.i])), "{:4d}".format(mon.ib_lo_active),
          "{:7d}".format(bool(SN.mon_run.disable_amp_fault[G.i])), "{:2d}".format(bool(mon.disable_amp_fault)),
          "{:11.4f}".format(SN.mon_run.dt[G.i]), "{:7.4f}".format(mon.dt),
          "{:15.6f}".format(SN.mon_run.ib_amp[G.i]), "{:13.6f}".format(mon.ib_amp),
          "{:9.4f}".format(SN.mon_run.ib_dyn_T_m[G.i]), "{:5.4f}".format(mon.LoopIbAmp.ChargeTransfer.dt),
          "{:15.6f}".format(SN.mon_run.ib_dyn_rstate_m[G.i]), "{:13.6f}".format(mon.LoopIbAmp.ChargeTransfer.rstate),
          "{:15.6f}".format(SN.mon_run.ib_dyn_lstate_m[G.i]), "{:13.6f}".format(mon.LoopIbAmp.ChargeTransfer.state),
          "{:21.5f}".format(SN.mon_run.ib_dyn_m[G.i]), "{:12.5f}".format(mon.LoopIbAmp.ib_dyn),
          "{:11.5f}".format(SN.mon_run.vb[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.dv_dyn_m[G.i]), "{:8.5f}".format(mon.LoopIbAmp.dv_dyn),
          "{:13.6f}".format(SN.mon_run.vb_model[G.i]), "{:11.6f}".format(mon.LoopIbAmp.vb),
          "{:13.6f}".format(SN.mon_run.voc_m[G.i]), "{:11.6f}".format(mon.LoopIbAmp.voc),
          "{:13.6f}".format(SN.mon_run.voc_soc[G.i]),  "{:9.6f}".format(mon.voc_soc),
          "{:11.6f}".format(SN.mon_run.voc_soc_m[G.i]), "{:11.6f}".format(mon.LoopIbAmp.voc_soc),
          "{:13.5f}".format(SN.mon_run.e_wrap_m[G.i]), "{:8.5f}".format(mon.e_wrap_m),
          "{:16.5f}".format(SN.mon_run.e_wrap_m_trim[G.i]), "{:8.5f}".format(mon.e_wrap_m_trim),
          "{:12.6f}".format(SN.mon_run.e_wrap_m_trimmed[G.i]), "{:9.6f}".format(mon.LoopIbAmp.e_wrap_trimmed),
          "{:12.4f}".format(SN.mon_run.ib_wrp_T_m[G.i]), "{:9.4f}".format(mon.LoopIbAmp.WrapErrFilt.dt),
          "{:12.6f}".format(SN.mon_run.ib_wrp_rate_m[G.i]), "{:9.6f}".format(mon.LoopIbAmp.WrapErrFilt.rate),
          "{:12d}".format(bool(SN.mon_run.ib_wrp_reset_m[G.i])), "{:9d}".format(bool(mon.LoopIbAmp.WrapErrFilt.reset)),
          "{:12.6f}".format(SN.mon_run.ib_wrp_state_m[G.i]), "{:9.6f}".format(mon.LoopIbAmp.WrapErrFilt.state),
          "{:11.5f}".format(SN.mon_run.e_wrap_m_filt[G.i]), "{:8.5f}".format(mon.e_wrap_m_filt),
          "{:13.6f}".format(SN.mon_run.voc_soc[G.i]), "{:12.6f}".format(mon.voc_soc),
          "{:11.5f}".format(SN.mon_run.voc_stat[G.i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(SN.mon_run.vsat[G.i]), "{:9.5f}".format(mon.vsat),
          "{:14.5f}".format(SN.mon_run.ib[G.i]), "{:12.5f}".format(mon.ib),
          "{:13.8f}".format(SN.mon_run.soc[G.i]), "{:10.8f}".format(mon.soc),
          "{:11.5f}".format(SN.mon_run.ewmhi_thr[G.i]), "{:8.5f}".format(mon.ewmhi_thr),
          "{:11.5f}".format(SN.mon_run.ewmlo_thr[G.i]), "{:8.5f}".format(mon.ewmlo_thr),
          "{:8d}".format(SN.mon_run.wrap_hi_m_flt[G.i]), "{:4d}".format(mon.wrap_hi_m_flt),
          "{:8d}".format(SN.mon_run.wrap_hi_m_fa[G.i]), "{:4d}".format(mon.wrap_hi_m_fa),
          "{:5.0f}".format(SN.mon_run.disable_amp_fault[G.i]),  "{:2.0f}".format(mon.disable_amp_fault),  "{:2.0f}".format(mon.ib_amp_lo),
          "{:26.0f}".format(SN.mon_run.e_wrap_m_reset[G.i]), "{:2d}".format(mon.e_wrap_m_reset),
          "{:18d}".format(SN.mon_run.fltw[G.i]), "{:4d}".format(SN.mon_run.falw[G.i]),
          )
    print(Colors.reset, end='')
    return hdr

# 9
# noinspection PyPep8Naming,PyUnusedLocal
def print_dyn_n_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf):
    global count_since_last_header, vv_warning_printed
    if not hasattr(SN.mon_run, 'ib_noa_lo') or SN.mon_run.ib_noa_lo is None:
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1-vv3 run.  Not printing print_dyn_n_RunSim  (request_hist_in=7)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i   time     r       rt   rk   it   ct      re   ie  ce    reset  reset_temp     reset_all_faults   soft_reset  soft_reset_sim  init_mon     init_sim     sa      dt                 vb                            ibmh                         ibmm                    ib_noa_lo    ib_noa_hi      dt                    ib_noa                    ib_dyn_T_n          ib_dyn_rstate_n                ib_dyn_lstate_n                      ib_dyn_n                 vb                      dv_dyn_n                vb_n                       voc_n                      voc_soc                   voc_soc_n                    e_wrap_n                  e_wrap_n_trim        e_wrap_trimmed_n         e_wrap_n_T           e_wrap_n_rate            e_wrap_n_reset       e_wrap_n_state          e_wrap_n_filt          ewmhi_thr              ewmlo_thr          e_wrap_n_flt   e_wrap_n_fa    ib_noa_lo   fltw   falw  e_wrap_n_filt"
    if (calc_temp or calc_ekf) and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.red, end='')
    elif mon.reset_temp:
        print(Colors.fg.orange, end='')
    print("{:4d}".format(G.i), "{:8.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(mon.reset_kf), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:7d}".format(bool(SN.mon_run.reset[G.i])),
          "{:7d}".format(bool(SN.mon_run.reset_temp[G.i])),
          "{:14d}".format(bool(SN.mon_run.reset_all_faults[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset_sim[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_mon[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_sim[G.i])),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:9.4f}".format(SN.mon_run.dt[G.i]), "{:7.4f}".format(mon.dt),
          "{:14.7f}".format(SN.mon_run.vb[G.i]), "{:12.7f}".format(mon.vb),
          "{:15.5f}".format(SN.mon_run.ib_noa_hdwe[G.i]), "{:13.5f}".format(mon.ib_noa_hdwe),
          "{:15.5f}".format(SN.mon_run.ib_noa_model[G.i]), "{:13.5f}".format(mon.ib_noa_model),
          "{:7d}".format(bool(SN.mon_run.ib_noa_lo[G.i])), "{:2d}".format(bool(mon.ib_noa_lo)),
          "{:7d}".format(bool(SN.mon_run.ib_noa_hi[G.i])), "{:2d}".format(bool(mon.ib_noa_hi)),
          "{:11.4f}".format(SN.mon_run.dt[G.i]), "{:7.4f}".format(mon.dt),
          "{:15.6f}".format(SN.mon_run.ib_noa[G.i]), "{:13.6f}".format(mon.ib_noa),
          "{:9.4f}".format(SN.mon_run.ib_dyn_T_n[G.i]), "{:5.4f}".format(mon.LoopIbNoa.ChargeTransfer.dt),
          "{:15.6f}".format(SN.mon_run.ib_dyn_rstate_n[G.i]), "{:13.6f}".format(mon.LoopIbNoa.ChargeTransfer.rstate),
          "{:15.6f}".format(SN.mon_run.ib_dyn_lstate_n[G.i]), "{:13.6f}".format(mon.LoopIbNoa.ChargeTransfer.state),
          "{:21.5f}".format(SN.mon_run.ib_dyn_n[G.i]), "{:12.5f}".format(mon.LoopIbNoa.ib_dyn),
          "{:12.5f}".format(SN.mon_run.vb[G.i]), "{:10.5f}".format(mon.vb),
          "{:12.5f}".format(SN.mon_run.dv_dyn_n[G.i]), "{:10.5f}".format(mon.LoopIbNoa.dv_dyn),
          "{:13.6f}".format(SN.mon_run.vb_model[G.i]), "{:12.6f}".format(mon.LoopIbNoa.vb),
          "{:13.6f}".format(SN.mon_run.voc_n[G.i]), "{:12.6f}".format(mon.LoopIbNoa.voc),
          "{:13.6f}".format(SN.mon_run.voc_soc[G.i]),  "{:12.6f}".format(mon.voc_soc),
          "{:12.6f}".format(SN.mon_run.voc_soc_n[G.i]), "{:12.6f}".format(mon.LoopIbNoa.voc_soc),
          "{:13.5f}".format(SN.mon_run.e_wrap_n[G.i]), "{:8.5f}".format(mon.e_wrap_n),
          "{:16.5f}".format(SN.mon_run.e_wrap_n_trim[G.i]), "{:8.5f}".format(mon.e_wrap_n_trim),
          "{:12.6f}".format(SN.mon_run.e_wrap_n_trimmed[G.i]), "{:9.6f}".format(mon.LoopIbNoa.e_wrap_trimmed),
          "{:12.4f}".format(SN.mon_run.ib_wrp_T_n[G.i]), "{:9.4f}".format(mon.LoopIbNoa.WrapErrFilt.dt),
          "{:12.6f}".format(SN.mon_run.ib_wrp_rate_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.rate),
          "{:12d}".format(bool(SN.mon_run.ib_wrp_reset_n[G.i])), "{:9d}".format(bool(mon.LoopIbNoa.WrapErrFilt.reset)),
          "{:12.6f}".format(SN.mon_run.ib_wrp_state_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.state),
          "{:12.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:9.5f}".format(mon.e_wrap_n_filt),
          "{:12.5f}".format(SN.mon_run.ewmhi_thr[G.i]), "{:9.5f}".format(mon.ewmhi_thr),
          "{:12.5f}".format(SN.mon_run.ewmlo_thr[G.i]), "{:9.5f}".format(mon.ewmlo_thr),
          "{:8d}".format(SN.mon_run.wrap_hi_n_flt[G.i]), "{:4d}".format(mon.wrap_hi_n_flt),
          "{:8d}".format(SN.mon_run.wrap_hi_n_fa[G.i]), "{:4d}".format(mon.wrap_hi_n_fa),
          "{:18d}".format(SN.mon_run.fltw[G.i]), "{:4d}".format(SN.mon_run.falw[G.i]),
          "{:11.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          )
    print(Colors.reset, end='')
    return hdr

# 1
# noinspection PyPep8Naming,PyUnusedLocal
def print_ekf_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_ekf, calc_temp):
    global count_since_last_header, vv_warning_printed
    if (not hasattr(SN.mon_run, 'voltage_low') or SN.mon_run.voltage_low is None) \
            or (not hasattr(SN.mon_run, 'frz') or SN.mon_run.frz is None):
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1-vv3 run.  Not printing print_ekf_RunSim  (request_hist_in=1)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i  time     r r_t    dt             i_e  r_e  c_e   dt_ekf         sa        voc_stat              voc_stat_past       bms_off_past  volt_low       bms_off     frz     ib_charge                   soc                    soc_ekf                 x_ekf                    y                          y_ekf_f                    y_ekf_f_T                  y_ekf_f_tau                y_ekf_f_state              z                            hx                     voc_ekf                Tb_f                     x_prior                  x                        x_for_hx                     x_post                    Tb_f                      tb_f_for_hx                hx                        u_ekf                      voc_stat_f              voc_soc                z                          P                              P_post                       P_prior                       Fx                        Bu                         H                          R                     S                    K                         Q                         R                         voc_stat_f_rstate     voc_stat_f_lstate       voc_stat_f_T"
    i_ekf = max(i_ekf, 0)
    if (calc_temp or calc_ekf) and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset_ekf:
        print(Colors.fg.red, end='')
    elif mon.u_ekf == 0.:
        print(Colors.fg.yellow, end='')
    elif calc_ekf:
        print(Colors.fg.green, end='')
    elif mon.reset_ekf:
        print(Colors.fg.lightblue, end='')
    elif mon.reset:
        print(Colors.fg.red, end='')

    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset), "{:2.0f}".format(mon.reset_temp),
          "{:9.3f}".format(SN.mon_run.dt[G.i]), "{:6.3f}".format(mon.dt),
          "{:4d}".format(i_ekf), "{:4d}".format(mon.reset_ekf), "{:4d}".format(calc_ekf),
          "{:9.3f}".format(SN.mon_run.dt_ekf[i_ekf]), "{:6.3f}".format(mon.dt_eframe),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:11.5f}".format(SN.mon_run.voc_stat[G.i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(SN.mon_run.voc_stat[G.i-1]), "{:9.5f}".format(mon.voc_stat_past),
          "{:7d}".format(bool(SN.mon_run.bms_off[G.i-1])), "{:3d}".format(bool(mon.bms_off_past)),
          "{:7d}".format(bool(SN.mon_run.voltage_low[G.i])), "{:3d}".format(bool(mon.voltage_low)),
          "{:7d}".format(bool(SN.mon_run.bms_off[G.i])), "{:3d}".format(bool(mon.bms_off)),
          "{:7.0f}".format(SN.mon_run.frz[i_ekf]), "{:3.0f}".format(mon.frz),
          "{:13.6f}".format(SN.mon_run.ib_charge[G.i]), "{:12.6f}".format(mon.ib_charge),
          "{:13.8f}".format(SN.mon_run.soc[G.i]), "{:10.8f}".format(mon.soc),
          "{:11.8f}".format(SN.mon_run.soc_ekf[G.i]), "{:9.8f}".format(mon.soc_ekf),
          "{:12.8f}".format(SN.mon_run.soc_ekf[G.i]), "{:10.8f}".format(mon.x),
          "{:13.8f}".format(SN.mon_run.y_ekf[G.i]), "{:12.8f}".format(mon.y_ekf),
          "{:13.8f}".format(SN.mon_run.y_ekf_f[G.i]), "{:12.8f}".format(mon.y_ekf_f),
          "{:13.8f}".format(SN.mon_run.y_ekf_f_T[G.i]), "{:12.8f}".format(mon.y_ekf_f_T),
          "{:13.8f}".format(SN.mon_run.y_ekf_f_tau[G.i]), "{:12.8f}".format(mon.y_ekf_f_tau),
          "{:13.8f}".format(SN.mon_run.y_ekf_f_lstate[G.i]), "{:12.8f}".format(mon.y_ekf_f_state),
          "{:12.6f}".format(SN.mon_run.z[i_ekf]), "{:13.6f}".format(mon.z),
          "{:14.6f}".format(SN.mon_run.hx[i_ekf]), "{:9.6f}".format(mon.hx),
          "{:11.5f}".format(SN.mon_run.voc_ekf[G.i]), "{:9.5f}".format(mon.voc_ekf),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:13.8f}".format(SN.mon_run.x_prior[i_ekf]), "{:10.8f}".format(mon.x_prior),
          "{:13.8f}".format(SN.mon_run.x[i_ekf]), "{:10.8f}".format(mon.x),
          "{:15.10f}".format(SN.mon_run.x_for_hx[i_ekf]), "{:12.10f}".format(mon.x_for_hx),
          "{:13.8f}".format(SN.mon_run.x_post[i_ekf]), "{:10.8f}".format(mon.x_post),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.mon_run.tb_f_for_hx[i_ekf]), "{:10.7f}".format(mon.tb_f_for_hx),
          "{:14.6f}".format(SN.mon_run.hx[i_ekf]), "{:9.6f}".format(mon.hx),
          "{:14.6f}".format(SN.mon_run.u[i_ekf]), "{:12.6f}".format(mon.u_ekf),
          "{:14.6f}".format(SN.mon_run.z[i_ekf]), "{:9.6f}".format(mon.voc_stat_f),
          "{:13.6f}".format(SN.mon_run.voc_soc[G.i]),  "{:9.6f}".format(mon.voc_soc),
          "{:12.6f}".format(SN.mon_run.z[i_ekf]), "{:13.6f}".format(mon.z),
          "{:16.11f}".format(SN.mon_run.P[i_ekf]), "{:12.11f}".format(mon.P),
          "{:16.11f}".format(SN.mon_run.P_post[i_ekf]), "{:12.11f}".format(mon.P_post),
          "{:14.11f}".format(SN.mon_run.P_prior[i_ekf]), "{:12.11f}".format(mon.P_prior),
          "{:13.9f}".format(SN.mon_run.Fx[i_ekf]), "{:11.9f}".format(mon.Fx),
          "{:13.9f}".format(SN.mon_run.Bu[i_ekf]), "{:11.9f}".format(mon.Bu),
          "{:14.7f}".format(SN.mon_run.H[i_ekf]), "{:11.7f}".format(mon.H),
          "{:11.6f}".format(SN.mon_run.R[i_ekf]), "{:9.6f}".format(mon.R),
          "{:11.6f}".format(SN.mon_run.S[i_ekf]), "{:9.6f}".format(mon.S),
          "{:13.9f}".format(SN.mon_run.K[i_ekf]), "{:10.9f}".format(mon.K),
          "{:13.9f}".format(SN.mon_run.Q[i_ekf]), "{:10.9f}".format(mon.Q),
          "{:13.9f}".format(SN.mon_run.R[i_ekf]), "{:10.9f}".format(mon.R),
          "{:11.6f}".format(SN.mon_run.voc_stat_f_rstate[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_rstate),
          "{:11.6f}".format(SN.mon_run.voc_stat_f_lstate[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_lstate),
          "{:12.6f}".format(SN.mon_run.voc_stat_f_T[i_ekf]), "{:9.6f}".format(mon.voc_stat_f_T),
          )
    print(Colors.reset, end='')
    return hdr

# 6
# noinspection PyPep8Naming,PyUnusedLocal
def print_kf_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf):
    global count_since_last_header, vv_warning_printed
    if not hasattr(SN.mon_run, 'dtm'):
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1 or vv3-vv4 run.  Not printing print_kf_RunSim  (request_hist_in=6)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i   time     r       rt   rk   dt               dtm              dtn              VoVcn                    VoVcnf                 x0                     ib_shunt_noa            [Fxn                                      ]    [Qn                                                                                                                ]  [Xpn                                      ]     [Ppn                                                                                                               ]   S                         [K                                                           ]   y                     [x                                         ]    [Pn                                                                                                                ]"
    if (calc_temp or calc_ekf) and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.red, end='')
    print("{:4d}".format(G.i), "{:8.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(mon.reset_kf),
          "{:9.4f}".format(SN.mon_run.dt[G.i]), "{:5.4f}".format(mon.dt),
          "{:9.4f}".format(SN.mon_run.dtm[G.i]), "{:5.4f}".format(SN.KfShuntAmp.dt),
          "{:9.4f}".format(SN.mon_run.dtn[G.i]), "{:5.4f}".format(SN.KfShuntNoa.dt),
          "{:12.7f}".format(SN.mon_run.vovcn[G.i]), "{:11.7f}".format(SN.VoVcn),
          "{:11.6f}".format(SN.mon_run.vovcnkf[G.i]), "{:10.6f}".format(SN.VoVcn_f),
          "{:11.6f}".format(SN.mon_run.x0n[G.i]), "{:10.6f}".format(SN.KfShuntNoa.x[0][0]),
          "{:11.6f}".format(SN.mon_run.iscn[G.i]), "{:10.6f}".format(SN.iscn),
          "{:8.1f}".format(SN.mon_run.Fx00n[G.i]), "{:3.1f}".format(SN.KfShuntNoa.Fx[0][0]),
          "{:7.3f}".format(SN.mon_run.Fx01n[G.i]), "{:5.3f}".format(SN.KfShuntNoa.Fx[0][1]),
          "{:5.1f}".format(SN.mon_run.Fx10n[G.i]), "{:3.1f}".format(SN.KfShuntNoa.Fx[1][0]),
          "{:5.1f}".format(SN.mon_run.Fx11n[G.i]), "{:3.1f}".format(SN.KfShuntNoa.Fx[1][1]),
          "{:18.7e}".format(SN.mon_run.Q00n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.Q[0][0]),
          "{:14.7e}".format(SN.mon_run.Q01n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.Q[0][1]),
          "{:14.7e}".format(SN.mon_run.Q10n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.Q[1][0]),
          "{:14.7e}".format(SN.mon_run.Q11n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.Q[1][1]),
          "{:11.6f}".format(SN.mon_run.xp0n[G.i]), "{:10.6f}".format(float(SN.KfShuntNoa.x_prior[0])),
          "{:11.6f}".format(SN.mon_run.xp1n[G.i]), "{:10.6f}".format(float(SN.KfShuntNoa.x_prior[1])),
          "{:18.7e}".format(SN.mon_run.Pp00n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P_prior[0][0]),
          "{:14.7e}".format(SN.mon_run.Pp01n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P_prior[0][1]),
          "{:14.7e}".format(SN.mon_run.Pp10n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P_prior[1][0]),
          "{:14.7e}".format(SN.mon_run.Pp11n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P_prior[1][1]),
          "{:13.8f}".format(SN.mon_run.Sn[G.i]), "{:10.8f}".format(SN.KfShuntNoa.S),
          "{:18.7e}".format(SN.mon_run.K0n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.K[0,0]),
          "{:18.7e}".format(SN.mon_run.K1n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.K[1,0]),
          "{:11.6f}".format(SN.mon_run.yn[G.i]), "{:10.6f}".format(SN.KfShuntNoa.y_kf),
          "{:11.6f}".format(SN.mon_run.x0n[G.i]), "{:10.6f}".format(SN.KfShuntNoa.x[0][0]),
          "{:11.6f}".format(SN.mon_run.kf_v_n[G.i]), "{:10.6f}".format(SN.KfShuntNoa.x[1][0]),
          "{:18.7e}".format(SN.mon_run.P00n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P[0][0]),
          "{:14.7e}".format(SN.mon_run.P01n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P[0][1]),
          "{:14.7e}".format(SN.mon_run.P10n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P[1][0]),
          "{:14.7e}".format(SN.mon_run.P11n[G.i]), "{:12.7e}".format(SN.KfShuntNoa.P[1][1]),
          )
    print(Colors.reset, end='')
    return hdr


# noinspection PyPep8Naming,PyUnusedLocal
def print_soc_RunSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf):
    global count_since_last_header
    hdr = "  i  time     r       rt   rk   it   ct      re   ie  ce    sa     ib_charge            soc                    dt                  i * dt * coul_eff     d_delq                      delq                         Tb_f                        ddq                       delq                        qcrs                          q_capacity                  Tb                       Tb_f_rate"
    if calc_temp and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        d_dq = SN.mon_run.delta_q[G.i]-SN.mon_run.delta_q[G.i-1]
        count_since_last_header += 1
    else:
        d_dq = SN.mon_run.delta_q[G.i+1]-SN.mon_run.delta_q[G.i]
    i_dt_old = SN.mon_run.dt[G.i] * SN.mon_run.ib_charge[G.i]
    i_dt_new = mon.dt * mon.ib_charge
    if mon.ib_charge > 0:
        i_dt_old *= mon.chemistry.coul_eff
        i_dt_new *= mon.chemistry.coul_eff
    if mon.reset:
        print(Colors.fg.red, end='')
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(mon.reset_kf), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:10.5f}".format(SN.mon_run.ib_charge[G.i]), "{:9.5f}".format(mon.ib_charge),
          "{:11.7f}".format(SN.mon_run.soc[G.i]), "{:8.7f}".format(mon.soc),
          "{:9.4f}".format(SN.mon_run.dt[G.i]), "{:5.4f}".format(mon.dt),
          "{:12.4f}".format(i_dt_old), "{:9.4f}".format(i_dt_new),
          "{:14.7f}".format(SN.mon_run.d_delta_q[G.i]), "{:11.7f}".format(mon.d_delta_q),
          "{:16.3f}".format(SN.mon_run.delta_q[G.i]), "{:13.3f}".format(mon.delta_q),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:12.3f}".format(d_dq), "{:11.3f}".format(mon.d_delta_q),
          "{:16.2f}".format(SN.mon_run.delta_q[G.i]), "{:11.2f}".format(mon.delta_q),
          "{:15.2f}".format(SN.mon_run.qcrs[G.i]), "{:13.2f}".format(mon.q_cap_rated_scaled),
          "{:15.2f}".format(SN.mon_run.q_capacity[G.i]), "{:13.2f}".format(mon.q_capacity),
          "{:14.7f}".format(SN.mon_run.Tb[G.i]), "{:10.7f}".format(mon.Tb),
          "{:12.7f}".format(SN.mon_run.Tb_f_rate[G.i]), "{:10.7f}".format(mon.Tb_f_rate),
         )
    print(Colors.reset, end='')
    return hdr

# 3
# noinspection PyPep8Naming
def print_soc_s_HistSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf):
    global count_since_last_header
    hdr = "  i  time     r       rt   rk   it   ct      re   ie  ce    sa       sa_s       dt                    dt_s                  ib_in_s                     ib_dyn                     dv_hys_s            soc                    delq                            soc_s                delta_q_s                       qcrs                    Tb_f                     vb                    voc_stat               voc_stat_s            dv_hys_s              dv_dyn_s             vsat                   bms_off_s"
    if calc_temp and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.red, end='')
    elif mon.reset_temp:
        print(Colors.fg.orange, end='')
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(mon.reset_kf), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:5.0f}".format(SN.sim_run.sat_s[G.i]), "{:2.0f}".format(sim.sat),
          "{:12.4f}".format(SN.mon_run.dt[G.i]), "{:8.4f}".format(mon.dt),
          "{:12.4f}".format(SN.sim_run.dt_s[G.i]), "{:8.4f}".format(sim.dt),
          "{:14.5f}".format(SN.sim_run.ib_in_s[G.i]), "{:12.5f}".format(sim.ib_in),
          "{:14.5f}".format(SN.mon_run.ib_dyn[G.i]), "{:12.5f}".format(mon.ib_dyn),
          "{:12.5f}".format(SN.sim_run.dv_hys_s[G.i]), "{:9.5f}".format(sim.dv_hys),
          "{:11.7f}".format(SN.mon_run.soc[G.i]), "{:8.7f}".format(mon.soc),
          "{:16.6f}".format(SN.mon_run.delta_q[G.i]), "{:13.6f}".format(mon.delta_q),
          "{:11.6f}".format(SN.mon_run.soc_s[G.i]), "{:9.6f}".format(sim.soc),
          "{:15.6f}".format(SN.sim_run.delta_q_s[G.i]), "{:13.6f}".format(sim.delta_q),
          "{:12.2f}".format(SN.mon_run.qcrs[G.i]), "{:9.2f}".format(mon.q_cap_rated_scaled),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:11.5f}".format(SN.mon_run.vb_f[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.voc_stat_f[G.i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(SN.sim_run.voc_stat_s[G.i]), "{:9.5f}".format(sim.voc_stat),
          "{:11.5f}".format(SN.sim_run.dv_hys_s[G.i]), "{:9.5f}".format(sim.dv_hys),
          "{:11.5f}".format(SN.sim_run.dv_dyn_s[G.i]), "{:9.5f}".format(sim.dv_dyn),
          "{:11.5f}".format(SN.mon_run.vsat[G.i]), "{:9.5f}".format(mon.vsat),
          "{:7d}".format(bool(SN.sim_run.bms_off_s[G.i])), "{:4d}".format(sim.bms_off),
          )
    if G.i == 2:
        pass
    print(Colors.reset, end='')
    return hdr

# 3
# noinspection PyPep8Naming
def print_soc_s_RunSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf):
    global count_since_last_header, vv_warning_printed
    if not hasattr(SN.sim_run, 'ib_charge_s') or SN.sim_run.ib_charge_s is None:
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1-vv3 run.  Not printing print_soc_s_RunSim (request_hist_in=3)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i  time     r       rt   rtps rk   it   ct      re   ie  ce    sa       sa_s       dt                    dt_s                  ib                         ib_in_s                       ib_s                          ib_charge_s                ib_dyn_rstate_s                ib_dyn_lstate_s              ib_dyn_T_s             ib_dyn_s                    ib_dyn                      dv_hys_s                 ib_charge_s                 ioc_s                  soc                      d_delq                   delq                             i * dt_s * coul_eff       soc_s                 Tb_model_f     Tb_hdwe_f      Tb_f_s                         d_delta_q_s              delta_q_s                       qcrs                   q_cap                  q_cap_s                 Tb_f_s                    Tb_f                      Tb_f                     Tb_f_rate               vb                    vb_s                  voc_stat              voc_stat_s            voc_s                  dv_hys_s              dv_dyn_s             vsat                bms_off_s    voltage_low_s"
    if calc_temp and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    i_dt_old = SN.sim_run.dt_s[G.i] * SN.sim_run.ib_charge_s[G.i]
    i_dt_new = sim.dt * sim.ib_charge
    if sim.ib_charge > 0:
        i_dt_old *= sim.chemistry.coul_eff
        i_dt_new *= sim.chemistry.coul_eff
    if mon.reset:
        print(Colors.fg.red, end='')
    if sim.reset_temp_past:
        print(Colors.fg.orange, end='')
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(sim.reset_temp_past), "{:4d}".format(mon.reset_kf), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:5.0f}".format(SN.sim_run.sat_s[G.i]), "{:2.0f}".format(sim.sat),
          "{:12.4f}".format(SN.mon_run.dt[G.i]), "{:8.4f}".format(mon.dt),
          "{:12.4f}".format(SN.sim_run.dt_s[G.i]), "{:8.4f}".format(sim.dt),
          "{:14.5f}".format(SN.mon_run.ib[G.i]), "{:12.5f}".format(mon.ib),
          "{:14.5f}".format(SN.sim_run.ib_in_s[G.i]), "{:12.5f}".format(sim.ib_in),
          "{:15.6f}".format(SN.sim_run.ib_s[G.i]), "{:13.6f}".format(sim.ib),
          "{:15.6f}".format(SN.sim_run.ib_charge_s[G.i]), "{:13.6f}".format(sim.ib_charge),
          "{:15.6f}".format(SN.sim_run.ib_dyn_rstate_s[G.i]), "{:13.6f}".format(sim.ChargeTransfer.rstate),
          "{:15.6f}".format(SN.sim_run.ib_dyn_lstate_s[G.i]), "{:13.6f}".format(sim.ChargeTransfer.state),
          "{:12.4f}".format(SN.sim_run.ib_dyn_T_s[G.i]), "{:8.4f}".format(sim.ChargeTransfer.dt),
          "{:14.5f}".format(SN.sim_run.ib_dyn_s[G.i]), "{:12.5f}".format(sim.ib_dyn),
          "{:14.5f}".format(SN.mon_run.ib_dyn[G.i]), "{:12.5f}".format(mon.ib_dyn),
          "{:12.5f}".format(SN.sim_run.dv_hys_s[G.i]), "{:9.5f}".format(sim.dv_hys),
          "{:14.5f}".format(SN.sim_run.ib_charge_s[G.i]), "{:12.5f}".format(sim.ib_charge),
          "{:14.5f}".format(SN.sim_run.ioc_s[G.i]), "{:12.5f}".format(sim.ioc),
          "{:11.7f}".format(SN.mon_run.soc[G.i]), "{:8.7f}".format(mon.soc),
          "{:14.7f}".format(SN.mon_run.d_delta_q[G.i]), "{:11.7f}".format(mon.d_delta_q),
          "{:16.6f}".format(SN.mon_run.delta_q[G.i]), "{:13.6f}".format(mon.delta_q),
          "{:14.5f}".format(i_dt_old), "{:11.5f}".format(i_dt_new),
          "{:11.7f}".format(SN.mon_run.soc_s[G.i]), "{:8.7f}".format(sim.soc),
          "{:14.8f}".format(SN.mon_run.Tb_model_f[G.i]), "{:14.8f}".format(SN.mon_run.Tb_hdwe_f[G.i]),
          "{:14.8f}".format(SN.sim_run.Tb_f_s[G.i]), "{:11.8f}".format(sim.Tb_f),
          "{:15.6f}".format(SN.sim_run.d_delta_q_s[G.i]), "{:13.6f}".format(sim.d_delta_q),
          "{:15.6f}".format(SN.sim_run.delta_q_s[G.i]), "{:13.6f}".format(sim.delta_q),
          "{:12.2f}".format(SN.mon_run.qcrs[G.i]), "{:9.2f}".format(mon.q_cap_rated_scaled),
          "{:12.2f}".format(SN.mon_run.q_capacity[G.i]), "{:9.2f}".format(mon.q_capacity),
          "{:12.2f}".format(SN.sim_run.qcap_s[G.i]), "{:9.2f}".format(sim.q_capacity),
          "{:14.7f}".format(SN.sim_run.Tb_f_s[G.i]), "{:10.7f}".format(sim.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:12.7f}".format(SN.mon_run.Tb_f_rate[G.i]), "{:10.7f}".format(mon.Tb_f_rate),
          "{:11.5f}".format(SN.mon_run.vb[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.sim_run.vb_s[G.i]), "{:9.5f}".format(sim.vb),
          "{:11.5f}".format(SN.mon_run.voc_stat[G.i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(SN.sim_run.voc_stat_s[G.i]), "{:9.5f}".format(sim.voc_stat),
          "{:11.5f}".format(SN.sim_run.voc_s[G.i]), "{:9.5f}".format(sim.voc),
          "{:11.5f}".format(SN.sim_run.dv_hys_s[G.i]), "{:9.5f}".format(sim.dv_hys),
          "{:11.5f}".format(SN.sim_run.dv_dyn_s[G.i]), "{:9.5f}".format(sim.dv_dyn),
          "{:11.5f}".format(SN.mon_run.vsat[G.i]), "{:9.5f}".format(mon.vsat),
          "{:7d}".format(bool(SN.sim_run.bms_off_s[G.i])), "{:4d}".format(sim.bms_off),
          "{:7d}".format(bool(SN.sim_run.voltage_low_s[G.i])), "{:4d}".format(sim.voltage_low),
          )
    if G.i == 2:
        pass
    print(Colors.reset, end='')
    return hdr

#4
# noinspection PyPep8Naming
def print_temp_RunSim(SN, i_temp, t, mon, sim, calc_temp, i_ekf, calc_ekf):
    global count_since_last_header, vv_warning_printed
    if not hasattr(SN.sim_run, 'Tb_f_s') or SN.sim_run.Tb_f_s is None:
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1-vv3 run.  Not printing print_temp_RunSim  (request_hist_in=4)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i  time     r  rt rk  mtb     re  ie   ce    Tb_hdwe                       Tb_flt        Tb_fa      Tb                         Tb_hdwe_f                  Tb_model                   Tb_model_f                 Tb_f                       Tb_s                       Tb_f_s                      Tb_model_f_rate            Tb_hdwe_f_rate           Tb_hdwe                    Tb_hdwe_f                   Tb_hdwe_f_dt             Tb_hdwe_f_tau               Tb_hdwe_f_rstate           Tb_hdwe_f_lstate            Tb_f_rate                 Tb_hdwe_f                   Tb_model_f_dt             Tb_model_f_rstate          Tb_model_f_lstate           Tb_f_rate                 Tb_f_for_hx"
    if calc_temp and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.red, end='')
    elif mon.reset_temp:
        print(Colors.fg.orange, end='')
    print("{:4d}".format(G.i), "{:7.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:3d}".format(mon.reset_temp), "{:2d}".format(mon.reset_kf), "{:2d}".format(mon.mtb),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:13.7f}".format(SN.mon_run.Tb_hdwe[G.i]), "{:11.7f}".format(mon.Tb_hdwe),
          "{:8d}".format(bool(SN.mon_run.Tb_flt[G.i])), "{:4d}".format(mon.Tb_flt),
          "{:8d}".format(bool(SN.mon_run.Tb_fa[G.i])), "{:4d}".format(mon.Tb_fa),
          "{:14.7f}".format(SN.mon_run.Tb[G.i]), "{:11.7f}".format(mon.Tb),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_f[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f),
          "{:14.7f}".format(SN.mon_run.Tb_model[G.i]), "{:11.7f}".format(mon.Tb_model),
          "{:14.7f}".format(SN.mon_run.Tb_model_f[G.i]), "{:11.7f}".format(mon.Tb_model_f),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:11.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.sim_run.Tb_s[G.i]), "{:11.7f}".format(sim.Tb_s),
          "{:14.7f}".format(SN.sim_run.Tb_f_s[G.i]), "{:11.7f}".format(sim.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_model_f_rate[G.i]), "{:11.7f}".format(mon.Tb_model_f_rate),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_f_rate[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f_rate),
          "{:13.7f}".format(SN.mon_run.Tb_hdwe[G.i]), "{:11.7f}".format(mon.Tb_hdwe),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_f[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f),
          "{:14.7f}".format(SN.mon_run.dt_sel[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f_dt),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_f_tau[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f_tau),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_f_rstate[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f_rstate),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_f_lstate[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f_lstate),
          "{:14.7f}".format(SN.mon_run.Tb_f_rate[G.i]), "{:11.7f}".format(mon.Tb_f_rate),
          "{:14.7f}".format(SN.mon_run.Tb_hdwe_f[G.i]), "{:11.7f}".format(mon.Tb_hdwe_f),
          "{:14.7f}".format(SN.mon_run.dt_sel[G.i]), "{:11.7f}".format(mon.Tb_model_f_dt),
          "{:14.7f}".format(SN.mon_run.Tb_model_f_rstate[G.i]), "{:11.7f}".format(mon.Tb_model_f_rstate),
          "{:14.7f}".format(SN.mon_run.Tb_model_f_lstate[G.i]), "{:11.7f}".format(mon.Tb_model_f_lstate),
          "{:14.7f}".format(SN.mon_run.Tb_f_rate[G.i]), "{:11.7f}".format(mon.Tb_f_rate),
          "{:14.7f}".format(SN.mon_run.tb_f_for_hx[i_ekf]), "{:10.7f}".format(mon.tb_f_for_hx),
          )
    print(Colors.reset, end='')
    return hdr

# 5
# noinspection PyPep8Naming
def print_volt_HistSim(SN, i_temp, i_ekf, t, mon, calc_temp, calc_ekf):
    global count_since_last_header
    hdr = "  i   time r    rt it   ct   rk   re ie     ce   sa       Tb_f                      vb_f                   ib_f                  ib_nh_f               ib_mh_f               ib_dyn_m              e_wrap_n_filt        e_wrap_m_filt        e_wrap_m_trim       ib_hn                 ib_dyn_n               e_wrap_n_filt        e_wrap_filt          soc                        dt                 Tb_f                     vb_f                  ib_dyn                voc_f     voc         voc_stat_f             soc_ekf"
    if count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.yellow, end='')
    print("{:4d}".format(G.i), "{:4.0f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:4d}".format(mon.reset_temp), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:4d}".format(mon.reset_kf),  "{:4d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:11.7f}".format(mon.Tb_f),
          "{:11.5f}".format(SN.mon_run.vb_f[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.ib_f[G.i]), "{:9.5f}".format(mon.ib),
          "{:11.5f}".format(SN.mon_run.ib_noa_hdwe_f[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib),
          "{:11.5f}".format(SN.mon_run.ib_amp_hdwe_f[G.i]), "{:9.5f}".format(mon.LoopIbAmp.ib),
          "{:11.5f}".format(SN.mon_run.ib_dyn_m[G.i]), "{:9.5f}".format(mon.LoopIbAmp.ib_dyn),
          "{:11.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:11.5f}".format(SN.mon_run.e_wrap_m_filt[G.i]), "{:8.5f}".format(mon.e_wrap_m_filt),
          "{:11.5f}".format(SN.mon_run.e_wrap_m_trim[G.i]), "{:8.5f}".format(mon.e_wrap_m_trim),
          "{:11.5f}".format(SN.mon_run.ib_noa_hdwe_f[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib),
          "{:11.5f}".format(SN.mon_run.ib_dyn_n[G.i]), "{:9.5f}".format(mon.LoopIbNoa.ib_dyn),
          "{:11.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:11.5f}".format(SN.mon_run.e_wrap_filt[G.i]), "{:8.5f}".format(mon.e_wrap_filt),
          "{:13.7f}".format(SN.mon_run.soc[G.i]), "{:10.7f}".format(mon.soc),
          "{:11.4f}".format(SN.mon_run.dt[G.i]), "{:8.4f}".format(mon.dt),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:11.5f}".format(SN.mon_run.vb_f[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.ib_dyn[G.i]), "{:9.5f}".format(mon.ib_dyn),
          "{:11.7f}".format(SN.mon_run.voc_f[G.i]), "{:10.7f}".format(mon.voc),
          "{:11.7f}".format(SN.mon_run.z[i_ekf]), "{:10.7f}".format(mon.voc_stat_f),
          "{:11.5f}".format(SN.mon_run.soc_ekf[G.i]), "{:9.5f}".format(mon.soc_ekf),
          )
    print(Colors.reset, end='')
    return hdr

#5
# noinspection PyPep8Naming
def print_volt_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf):
    global count_since_last_header, vv_warning_printed
    if not hasattr(SN.mon_run, 'ib_amp_lo') or SN.mon_run.ib_amp_lo is None:
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1 run.  Not printing print_volt_RunSim  (request_hist_in=5)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i   time     r       rt   rk   it   ct      re   ie  ce    reset  reset_temp     reset_all_faults   soft_reset  soft_reset_sim  init_mon     init_sim     sa      dt                vb                          ib_charge                   ib_sel         ib                          ib_amp_hdwe                 ib_amp_model                ib_amp                      ib_noa_hdwe                 ib_noa_model               ib_noa                disable_amp_fault  ib_diff                     ibh                         ib_s                 ib_amp_lo    ib_amp_hi   ib_noa_lo   ib_noa_hi dis_amp_flt    dt                  ib_amp                   ib_dyn_T_m           ib_dyn_rstate_m               ib_dyn_lstate_m                     ib_dyn_m                 vb                    vb_model               vb_hdwe               vb_hdwe_f             dv_dyn_m               e_wrap_m_T           e_wrap_m_tau          e_wrap_m_rate          e_wrap_m_reset         e_wrap_m_state        voc                   voc_soc                e_wrap_m             e_wrap_m_filt   disable_amp_fault ib_amp_lo  ib_noa_lo  e_wrap_m_reset  e_wrap_m_trim         ib_dyn_n                 ib_dyn_T_n        dv_dyn_n               e_wrap_n             e_wrap_n_filt         ib_dyn_n                    ib_dyn                   ib_dyn_T_n           ib_dyn_rstate_n               ib_dyn_lstate_n             dv_dyn_n                e_wrap_n_T             e_wrap_n_tau         e_wrap_n_rate          e_wrap_n_state              e_wrap_n_trim        e_wrap_n_trimmed       e_wrap_n             e_wrap_n_filt         ib                         e_wrap               e_wrap_filt          ib_dyn_r     ib_dyn_in                     ib_dyn_T                     ib_dyn_rstate                 ib_dyn_lstate                 ib_dyn                       dv_dyn                  dv_hys                  soc                     dt                Tb_f                      Tb_f                 voc_soc               voc                   voc_stat              voc_stat_s            voc_stat_f             soc_ekf               y_ekf                 y_ekf_f                fltw falw    soc_min"
    if (calc_temp or calc_ekf) and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.red, end='')
    if  t[max(G.i,0)] > 14.7:
        pass
    print("{:4d}".format(G.i), "{:8.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(mon.reset_kf), "{:4d}".format(i_temp), "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:7d}".format(bool(SN.mon_run.reset[G.i])),
          "{:7d}".format(bool(SN.mon_run.reset_temp[G.i])),
          "{:14d}".format(bool(SN.mon_run.reset_all_faults[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset_sim[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_mon[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_sim[G.i])),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:9.4f}".format(SN.mon_run.dt[G.i]), "{:7.4f}".format(mon.dt),
          "{:13.7f}".format(SN.mon_run.vb[G.i]), "{:11.7f}".format(mon.vb),
          "{:14.6f}".format(SN.mon_run.ib_charge[G.i]), "{:12.6f}".format(mon.ib_charge),
          "{:14.6f}".format(SN.mon_run.ib_sel[G.i]),
          "{:14.6f}".format(SN.mon_run.ib[G.i]), "{:12.6f}".format(mon.ib),
          "{:14.6f}".format(SN.mon_run.ib_amp_hdwe[G.i]), "{:12.6f}".format(mon.ib_amp_hdwe),
          "{:14.6f}".format(SN.mon_run.ib_amp_model[G.i]), "{:12.6f}".format(mon.ib_amp_model),
          "{:14.6f}".format(SN.mon_run.ib_amp[G.i]), "{:12.6f}".format(mon.ib_amp),
          "{:14.6f}".format(SN.mon_run.ib_noa_hdwe[G.i]), "{:12.6f}".format(mon.ib_noa_hdwe),
          "{:14.6f}".format(SN.mon_run.ib_noa_model[G.i]), "{:12.6f}".format(mon.ib_noa_model),
          "{:14.6f}".format(SN.mon_run.ib_noa[G.i]), "{:12.6f}".format(mon.ib_noa),
          "{:7d}".format(bool(SN.mon_run.disable_amp_fault[G.i])), "{:2d}".format(bool(mon.disable_amp_fault)),
          "{:14.6f}".format(SN.mon_run.ib_diff[G.i]), "{:12.6f}".format(mon.ib_diff),
          "{:14.6f}".format(SN.mon_run.ib_h[G.i]), "{:12.6f}".format(mon.ib_hdwe),
          "{:14.6f}".format(SN.mon_run.ib_s[G.i]), "{:12.6f}".format(sim.ib),
          "{:7d}".format(bool(SN.mon_run.ib_amp_lo[G.i])), "{:2d}".format(bool(mon.ib_amp_lo)),
          "{:7d}".format(bool(SN.mon_run.ib_amp_hi[G.i])), "{:2d}".format(bool(mon.ib_amp_hi)),
          "{:7d}".format(bool(SN.mon_run.ib_noa_lo[G.i])), "{:2d}".format(bool(mon.ib_noa_lo)),
          "{:7d}".format(bool(SN.mon_run.ib_noa_hi[G.i])), "{:2d}".format(bool(mon.ib_noa_hi)),
          "{:7d}".format(bool(SN.mon_run.disable_amp_fault[G.i])), "{:2d}".format(bool(mon.disable_amp_fault)),
          "{:11.4f}".format(SN.mon_run.dt[G.i]), "{:7.4f}".format(mon.dt),
          "{:14.6f}".format(SN.mon_run.ib_amp[G.i]), "{:12.6f}".format(mon.ib_amp),
          "{:9.4f}".format(SN.mon_run.ib_dyn_T_m[G.i]), "{:5.4f}".format(mon.LoopIbAmp.ChargeTransfer.dt),
          "{:15.6f}".format(SN.mon_run.ib_dyn_rstate_m[G.i]), "{:13.6f}".format(mon.LoopIbAmp.ChargeTransfer.rstate),
          "{:15.6f}".format(SN.mon_run.ib_dyn_lstate_m[G.i]), "{:13.6f}".format(mon.LoopIbAmp.ChargeTransfer.state),
          "{:21.6f}".format(SN.mon_run.ib_dyn_m[G.i]), "{:12.6f}".format(mon.LoopIbAmp.ib_dyn),
          "{:11.5f}".format(SN.mon_run.vb[G.i]), "{:9.5f}".format(mon.vb),
          "{:11.5f}".format(SN.mon_run.vb_model[G.i]), "{:9.5f}".format(mon.vb_model),
          "{:11.5f}".format(SN.mon_run.vb_hdwe[G.i]), "{:9.5f}".format(mon.vb_hdwe),
          "{:11.5f}".format(SN.mon_run.vb_hdwe_f[G.i]), "{:9.5f}".format(mon.vb_hdwe_f),
          "{:11.5f}".format(SN.mon_run.dv_dyn_m[G.i]), "{:8.5f}".format(mon.LoopIbAmp.dv_dyn),
          "{:12.4f}".format(SN.mon_run.ib_wrp_T_m[G.i]), "{:9.4f}".format(mon.LoopIbAmp.WrapErrFilt.dt),
          "{:12.4f}".format(SN.mon_run.ib_wrp_tau_m[G.i]), "{:9.4f}".format(mon.LoopIbAmp.WrapErrFilt.tau),
          "{:12.6f}".format(SN.mon_run.ib_wrp_rate_m[G.i]), "{:9.6f}".format(mon.LoopIbAmp.WrapErrFilt.rate),
          "{:12d}".format(bool(SN.mon_run.ib_wrp_reset_m[G.i])), "{:9d}".format(bool(mon.LoopIbAmp.WrapErrFilt.reset)),
          "{:12.6f}".format(SN.mon_run.ib_wrp_state_m[G.i]), "{:9.6f}".format(mon.LoopIbAmp.WrapErrFilt.state),
          "{:11.5f}".format(SN.mon_run.voc[G.i]), "{:9.5f}".format(mon.voc),
          "{:11.5f}".format(SN.mon_run.voc_soc[G.i]), "{:9.5f}".format(mon.voc_soc),
          "{:11.5f}".format(SN.mon_run.e_wrap_m[G.i]), "{:8.5f}".format(mon.e_wrap_m),
          "{:11.5f}".format(SN.mon_run.e_wrap_m_filt[G.i]), "{:8.5f}".format(mon.e_wrap_m_filt),
          "{:5.0f}".format(SN.mon_run.disable_amp_fault[G.i]),  "{:2.0f}".format(mon.disable_amp_fault),  "{:2.0f}".format(mon.ib_amp_lo),"{:2.0f}".format(mon.ib_noa_lo),
          "{:26.0f}".format(SN.mon_run.e_wrap_m_reset[G.i]), "{:2d}".format(mon.e_wrap_m_reset),
          # "{:11.5f}".format(SN.mon_run.e_wrap_m_filt[G.i]), "{:8.5f}".format(mon.e_wrap_m_filt), "{:8.5f}".format(SN.e_wrap_m_filt_init),
          "{:16.5f}".format(SN.mon_run.e_wrap_m_trim[G.i]), "{:8.5f}".format(mon.e_wrap_m_trim),
          "{:14.6f}".format(SN.mon_run.ib_dyn_n[G.i]), "{:12.6f}".format(mon.LoopIbNoa.ib_dyn),
          "{:9.4f}".format(SN.mon_run.ib_dyn_T_n[G.i]), "{:5.4f}".format(mon.LoopIbNoa.ChargeTransfer.dt),
          "{:11.6f}".format(SN.mon_run.dv_dyn_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.dv_dyn),
          "{:11.5f}".format(SN.mon_run.e_wrap_n[G.i]), "{:8.5f}".format(mon.e_wrap_n),
          "{:11.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:14.6f}".format(SN.mon_run.ib_dyn_n[G.i]), "{:12.6f}".format(mon.LoopIbNoa.ib_dyn),
          "{:14.6f}".format(SN.mon_run.ib_dyn[G.i]), "{:12.6f}".format(mon.ib_dyn),
          "{:9.4f}".format(SN.mon_run.ib_dyn_T_n[G.i]), "{:5.4f}".format(mon.LoopIbNoa.ChargeTransfer.dt),
          "{:15.6f}".format(SN.mon_run.ib_dyn_rstate_n[G.i]), "{:13.6f}".format(mon.LoopIbNoa.ChargeTransfer.rstate),
          "{:15.6f}".format(SN.mon_run.ib_dyn_lstate_n[G.i]), "{:13.6f}".format(mon.LoopIbNoa.ChargeTransfer.state),
          "{:11.5f}".format(SN.mon_run.dv_dyn_n[G.i]), "{:9.5f}".format(mon.LoopIbNoa.dv_dyn),
          "{:12.4f}".format(SN.mon_run.ib_wrp_T_n[G.i]), "{:9.4f}".format(mon.LoopIbNoa.WrapErrFilt.dt),
          "{:12.4f}".format(SN.mon_run.ib_wrp_tau_n[G.i]), "{:9.4f}".format(mon.LoopIbNoa.WrapErrFilt.tau),
          "{:12.6f}".format(SN.mon_run.ib_wrp_rate_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.rate),
          "{:12.6f}".format(SN.mon_run.ib_wrp_state_n[G.i]), "{:9.6f}".format(mon.LoopIbNoa.WrapErrFilt.state),
          "{:16.5f}".format(SN.mon_run.e_wrap_n_trim[G.i]), "{:8.5f}".format(mon.e_wrap_n_trim),
          "{:12.6f}".format(SN.mon_run.e_wrap_n_trimmed[G.i]), "{:9.6f}".format(mon.LoopIbNoa.e_wrap_trimmed),
          "{:11.5f}".format(SN.mon_run.e_wrap_n[G.i]), "{:8.5f}".format(mon.e_wrap_n),
          "{:11.5f}".format(SN.mon_run.e_wrap_n_filt[G.i]), "{:8.5f}".format(mon.e_wrap_n_filt),
          "{:14.6f}".format(SN.mon_run.ib[G.i]), "{:12.6f}".format(mon.ib),
          "{:11.5f}".format(SN.mon_run.e_wrap[G.i]), "{:8.5f}".format(mon.e_wrap),
          "{:11.5f}".format(SN.mon_run.e_wrap_filt[G.i]), "{:8.5f}".format(mon.e_wrap_filt),
          "{:6d}".format(bool(SN.mon_run.ib_dyn_r[G.i])), "{:2d}".format(mon.ib_dyn_r),
          "{:15.6f}".format(SN.mon_run.ib_dyn_rstate[G.i]), "{:13.6f}".format(mon.ib_dyn_in),
          "{:15.6f}".format(SN.mon_run.ib_dyn_T[G.i]), "{:13.6f}".format(mon.ib_dyn_T),
          "{:15.6f}".format(SN.mon_run.ib_dyn_rstate[G.i]), "{:13.6f}".format(mon.ib_dyn_rstate),
          "{:15.6f}".format(SN.mon_run.ib_dyn_lstate[G.i]), "{:13.6f}".format(mon.ib_dyn_lstate),
          "{:15.6f}".format(SN.mon_run.ib_dyn[G.i]), "{:12.6f}".format(mon.ib_dyn),
          "{:14.6f}".format(SN.mon_run.dv_dyn[G.i]), "{:10.6f}".format(mon.dv_dyn),
          "{:12.6f}".format(SN.mon_run.dv_hys[G.i]), "{:10.6f}".format(mon.dv_hys),
          "{:13.7f}".format(SN.mon_run.soc[G.i]), "{:10.7f}".format(mon.soc),
          "{:9.4f}".format(SN.mon_run.dt[G.i]), "{:5.4f}".format(mon.dt),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:14.7f}".format(SN.mon_run.Tb_f[G.i]), "{:10.7f}".format(mon.Tb_f),
          "{:11.5f}".format(SN.mon_run.voc_soc[G.i]), "{:9.5f}".format(mon.voc_soc),
          "{:11.5f}".format(SN.mon_run.voc[G.i]), "{:9.5f}".format(mon.voc),
          "{:11.5f}".format(SN.mon_run.voc_stat[G.i]), "{:9.5f}".format(mon.voc_stat),
          "{:11.5f}".format(SN.sim_run.voc_stat_s[G.i]), "{:9.5f}".format(sim.voc_stat),
          "{:11.5f}".format(SN.mon_run.z[i_ekf]), "{:9.5f}".format(mon.voc_stat_f),
          "{:11.5f}".format(SN.mon_run.soc_ekf[G.i]), "{:9.5f}".format(mon.soc_ekf),
          "{:11.5f}".format(SN.mon_run.y_ekf[G.i]), "{:9.5f}".format(mon.y_ekf),
          "{:11.5f}".format(SN.mon_run.y_ekf_f[G.i]), "{:9.5f}".format(mon.y_ekf_f),
          "{:8d}".format(SN.mon_run.fltw[G.i]), "{:4d}".format(SN.mon_run.falw[G.i]),
          "{:11.5f}".format(SN.mon_run.soc_min[G.i]), "{:9.5f}".format(mon.soc_min),
          )
    print(Colors.reset, end='')
    return hdr

# 8
# noinspection PyPep8Naming
def print_vb_wrap_RunSim(SN, i_temp, i_ekf, t, mon, sim, calc_temp, calc_ekf):
    global count_since_last_header, vv_warning_printed
    if not hasattr(SN.mon_run, 'voltage_low') or SN.mon_run.voltage_low is None:
        if not vv_warning_printed:
            print(Colors.fg.red, end='')
            print(f"\n**********\nLikely a vv1-vv3 run.  Not printing print_vb_wrap_RunSim  (request_hist_in=8)\n*************\n")
            vv_warning_printed = True
            print(Colors.reset, end='')
        return None
    hdr = "  i   time     r       rt   rk   it   ct      re   ie  ce    reset  reset_temp     reset_all_faults   soft_reset  soft_reset_sim  init_mon     init_sim     sa      bms_off     voltage_low   bms_off_s   voltage_low_s  dt                vb                           ib_amp                      vb_m                      voc_m                   voc_soc_m                 voc_soc                   e_wrap_m                  e_wrap_trim          e_wrap_trimmed         e_wrap_m_filt      ib_diff    wrap_m_and_n_fa    wrap_lo_m_fa       wrap_lo_n_fa       wrap_lo_fa         wrap_hi_m_fa       wrap_hi_n_fa       wrap_hi_fa         ib_is_functional   wrap_vb_faj        ib_quiet          ib_really_quiet"
    if (calc_temp or calc_ekf) and count_since_last_header > HDR_SPREAD:
        print(hdr)
        count_since_last_header = 0
    if G.i > 0:
        count_since_last_header += 1
    if mon.reset:
        print(Colors.fg.red, end='')
    elif mon.reset_temp:
        print(Colors.fg.orange, end='')
    print("{:4d}".format(G.i), "{:8.3f}".format(t[G.i]), "{:2.0f}".format(mon.reset),
          "{:7d}".format(mon.reset_temp), "{:4d}".format(mon.reset_kf), "{:4d}".format(i_temp),
          "{:4d}".format(calc_temp),
          "{:7d}".format(mon.reset_ekf), "{:4d}".format(i_ekf), "{:4d}".format(calc_ekf),
          "{:7d}".format(bool(SN.mon_run.reset[G.i])),
          "{:7d}".format(bool(SN.mon_run.reset_temp[G.i])),
          "{:14d}".format(bool(SN.mon_run.reset_all_faults[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset[G.i])),
          "{:15d}".format(bool(SN.mon_run.soft_reset_sim[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_mon[G.i])),
          "{:15d}".format(bool(SN.mon_run.init_sim[G.i])),
          "{:4.0f}".format(SN.mon_run.sat[G.i]), "{:2.0f}".format(mon.sat),
          "{:7d}".format(bool(SN.mon_run.bms_off[G.i])), "{:4d}".format(bool(mon.bms_off)),
          "{:7d}".format(bool(SN.mon_run.voltage_low[G.i])), "{:4d}".format(bool(mon.voltage_low)),
          "{:7d}".format(bool(SN.sim_run.bms_off_s[G.i])), "{:4d}".format(bool(sim.bms_off)),
          "{:7d}".format(bool(SN.sim_run.voltage_low_s[G.i])), "{:4d}".format(bool(sim.voltage_low)),
          "{:9.4f}".format(SN.mon_run.dt[G.i]), "{:7.4f}".format(mon.dt),
          "{:13.7f}".format(SN.mon_run.vb[G.i]), "{:11.7f}".format(mon.vb),
          "{:15.6f}".format(SN.mon_run.ib_amp[G.i]), "{:13.6f}".format(mon.ib_amp),
          "{:13.6f}".format(SN.mon_run.vb_model[G.i]), "{:11.6f}".format(mon.LoopIbAmp.vb),
          "{:13.6f}".format(SN.mon_run.voc_m[G.i]), "{:11.6f}".format(mon.LoopIbAmp.voc),
          "{:11.6f}".format(SN.mon_run.voc_soc_m[G.i]), "{:11.6f}".format(mon.LoopIbAmp.voc_soc),
          "{:13.6f}".format(SN.mon_run.voc_soc[G.i]), "{:9.6f}".format(mon.voc_soc),
          "{:13.5f}".format(SN.mon_run.e_wrap_m[G.i]), "{:8.5f}".format(mon.e_wrap_m),
          "{:16.5f}".format(SN.mon_run.e_wrap_m_trim[G.i]), "{:8.5f}".format(mon.e_wrap_m_trim),
          "{:12.6f}".format(SN.mon_run.e_wrap_m_trimmed[G.i]), "{:9.6f}".format(mon.LoopIbAmp.e_wrap_trimmed),
          "{:11.5f}".format(SN.mon_run.e_wrap_m_filt[G.i]), "{:8.5f}".format(mon.e_wrap_m_filt),
          "{:4d}".format(SN.mon_run.ib_diff_fa[G.i]),
          "{:10d}".format(SN.mon_run.wrap_m_and_n_fa[G.i]),
          "{:18d}".format(SN.mon_run.wrap_lo_m_fa[G.i]),
          "{:18d}".format(SN.mon_run.wrap_lo_n_fa[G.i]),
          "{:18d}".format(SN.mon_run.wrap_lo_fa[G.i]),
          "{:18d}".format(SN.mon_run.wrap_hi_m_fa[G.i]),
          "{:18d}".format(SN.mon_run.wrap_hi_n_fa[G.i]),
          "{:18d}".format(SN.mon_run.wrap_hi_fa[G.i]),
          "{:18d}".format(SN.mon_run.ib_is_functional[G.i]),
          "{:18d}".format(SN.mon_run.wv_fa[G.i]),
          "{:18d}".format(bool(SN.mon_run.ib_quiet[G.i])),
          "{:18d}".format(bool(SN.mon_run.ib_really_quiet[G.i])),
          )
    print(Colors.reset, end='')
    return hdr


def save_clean_file(mon_ver, csv_file, unit_key):
    if mon_ver is None:
        print("save_clean_file: mon_ver is None (broke early due to skip), skipping save.")
        return
    default_header_str = "unit,               hm,                  cTime,        dt,       sat,sel,mod,\
      Tb,Tb_rap,Tb_f,Tb_f,Tb_f_rate,Tb_f_rate_rap, vb,  ib,  ib_dyn, ioc,  voc_soc,    vsat,dv_dyn,voc_stat,voc_stat_f,voc_ekf,     y,    soc_s,soc_ekf,soc,ib_lag,voc_soc_new,"
    n = len(mon_ver.time)
    date_time_start = datetime.now()
    with open(csv_file, "w") as output:
        output.write(default_header_str + "\n")
        for i in range(n):
            s = unit_key + ','
            dt_dt = timedelta(seconds=mon_ver.time[i]-mon_ver.time[0])
            time_stamp = date_time_start + dt_dt
            s += time_stamp.strftime("%Y-%m-%dT%H:%M:%S,")
            s += "{:7.4f},".format(mon_ver.time[i] + mon_ver.time_run_start)
            s += "{:7.4f},".format(mon_ver.dt[i])
            s += "{:1.0f},".format(mon_ver.sat[i])
            # s += "{:1.0f},".format(mon_ver.sel[i])
            s += "{:1.0f},".format(mon_ver.mod_data[i])
            s += "{:7.6f},".format(mon_ver.Tb[i])
            s += "{:7.6f},".format(mon_ver.Tb_f[i])
            s += "{:7.6f},".format(mon_ver.Tb_f[i])
            s += "{:7.6f},".format(mon_ver.Tb_f_rate[i])
            s += "{:7.3f},".format(mon_ver.vb[i])
            s += "{:7.3f},".format(mon_ver.ib[i])
            s += "{:7.3f},".format(mon_ver.ib_dyn[i])
            s += "{:7.3f},".format(mon_ver.ioc[i])
            s += "{:7.3f},".format(mon_ver.voc_soc[i])
            s += "{:7.3f},".format(mon_ver.vsat[i])
            s += "{:7.3f},".format(mon_ver.dv_dyn[i])
            s += "{:7.3f},".format(mon_ver.voc_stat[i])
            s += "{:7.3f},".format(mon_ver.voc_ekf[i])
            s += "{:7.3f},".format(mon_ver.y[i])
            s += "{:7.3f},".format(mon_ver.soc_s[i])
            s += "{:7.3f},".format(mon_ver.soc_ekf[i])
            s += "{:7.3f},".format(mon_ver.soc[i])
            s += "{:7.5f},".format(mon_ver.ib_lag[i])
            s += "{:7.3f},".format(mon_ver.voc_soc_new[i])
            s += "\n"
            output.write(s)
        print("Wrote(save_clean_file):", csv_file)

def save_clean_file_sim(sim_ver, csv_file, unit_key):
    header_str = "unit_m,c_time,Tb_s,vsat_s,voc_stat_s,dv_dyn_s,vb_s,ib_s,sat_s,delta_q_s,\
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

def save_fault_coverage(mon_run, csv_file, unit_key):
    hdr_list = ['unit_fault', 'hm']
    flt_list = ['fltw', 'falw', 'ccd_fa', 'ib_diff_flt', 'ib_diff_fa',
                'wrap_hi_flt', 'wrap_lo_flt', 'vc_flt', 'wrap_hi_m_flt', 'wrap_lo_m_flt', 'wrap_hi_n_flt',
                'wrap_lo_n_flt', 'wrap_m_and_n_flt', 'red_loss', 'wrap_hi_fa', 'wrap_lo_fa', 'wv_fa', 'vc_fa',
                'wrap_hi_m_fa', 'wrap_lo_m_fa', 'wrap_hi_n_fa', 'wrap_lo_n_fa', 'wrap_m_and_n_fa', 'ib_sel',
                'ib_noa_bare_flt', 'ib_amp_bare_flt', 'ib_dscn_flt', 'ib_dscn_fa', 'ib_noa_flt', 'ib_noa_fa',
                'ib_amp_flt', 'ib_amp_fa', 'vb_flt', 'vb_fa_lt', 'Tb_flt', 'Tb_fa', 'bms_off', 'sat', 'red_loss']
    default_header_str = ''
    import numpy as np
    m = 0
    flt_data = []
    mon_run.wrap_m_and_n_flt = ( (np.bool(mon_run.wrap_lo_n_flt) & np.bool(mon_run.wrap_lo_m_flt)) |
                                 (np.bool(mon_run.wrap_hi_n_flt) & np.bool(mon_run.wrap_hi_m_flt)) )
    mon_run.wrap_m_and_n_fa = ( (np.bool(mon_run.wrap_lo_n_fa) & np.bool(mon_run.wrap_lo_m_fa)) |
                                 (np.bool(mon_run.wrap_hi_n_fa) & np.bool(mon_run.wrap_hi_m_fa)) )
    for flt in hdr_list:
        default_header_str += flt + ','
    for flt in flt_list:
        default_header_str += flt + ','
        flt_data.append(getattr(mon_run, flt))
        m += 1
    n = len(mon_run.time)
    date_time_start = datetime.now()
    with open(csv_file, "w") as output:
        output.write(default_header_str + "\n")
        for i in range(n):
            s = unit_key + ','
            dt_dt = timedelta(seconds=mon_run.time[i] - mon_run.time[0])
            time_stamp = date_time_start + dt_dt
            s += time_stamp.strftime("%Y-%m-%dT%H:%M:%S,")
            for j in range(m):
                s += "{:2d},".format(np.bool(flt_data[j][i]))
            s += "\n"
            output.write(s)
        s = 'covered: '
        for j in range(m):
            if any(flt_data[j][:] == 1):
                s += flt_list[j] + ','
        s += "\n"
        output.write(s)

    print("Wrote(save_fault_coverage):", csv_file)

