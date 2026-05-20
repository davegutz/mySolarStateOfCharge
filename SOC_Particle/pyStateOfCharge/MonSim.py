# MonSim:  Monitor and Simulator replication of Particle Photon Application
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

""" Python model of what's installed on the Particle Photon.  Includes
a monitor object (MON) and a simulation object (SIM).   The monitor is
the EKF and Coulomb Counter.   The SIM is a battery model, that also has a
Coulomb Counter built in."""
from MonSimNomConfig import *  # Global config parameters.   Overwrite in your own calls for studies
from battery_constants import apply_off_nominal_battery
from Battery import BatteryMonitor, BatterySim, is_sat, Retained
from UserOptions import UserOptions
from filter.TFDelay import TFDelay
from MonSimClasses import *
from MonSimPrint import *
# noinspection PyPep8Naming
import Globals as G

def battery_size(mr, sr, scale_in_, unit_cap_rated_):
    if hasattr(mr, 'qcrs'):
        scale_mon_ = mr.qcrs[0] / (unit_cap_rated_*3600)
    else:
        scale_mon_ = unit_cap_rated / unit_cap_rated_
        if scale_in_:
            scale_mon_ *= scale_in_
    if sr is not None and hasattr(sr, 'qcrs_s'):
        scale_sim_ = sr.qcrs_s[0] / (unit_cap_rated_*3600)
    else:
        scale_sim_ = scale_mon_
        if scale_in_:
            scale_sim_ *= scale_in_
    return scale_mon_, scale_sim_


def chm_from_mon_or_sim(mr, sr):
    chem_m = mr.chm
    if sr is not None:
        chem_s = sr.chm_s
    else:
        chem_s = mr.chm
    return chem_m, chem_s

def get_modeling(mr, mod_force=None):
    if mod_force is not None:
        return mod_force * np.ones(len(mr.time))
    if hasattr(mr, 'mod_data'):
        modeling_ = mr.mod_data
    else:
        modeling_ = 255 * np.ones(len(mr.time))
    return modeling_

def sync_to_mon_or_sim(mr, sr, t_mx=None):
    if sr is not None and len(sr.time) < len(mr.time):
        # time = sr.time
        # dtime = sr.dt_s
        time = mr.time
        dtime = mr.dt
    else:
        time = mr.time
        dtime = mr.dt
    if t_mx is not None:
        t_delt = time - time[0]
        time = time[np.where(t_delt <= t_mx)]
        dtime = dtime[np.where(t_delt <= t_mx)]
    return time, dtime


def Tb_from_raw_or_selected(use_raw, mr, sr):
    if use_raw:
        if rp.modeling_tb():
            Tb_ = mr.Tb_model
            Tb_f_ = mr.Tb_model_f
        else:
            Tb_ = mr.Tb_hdwe
            Tb_f_ = mr.Tb_hdwe_f
        Tb_s_ = Tb_
        Tb_f_s_ = Tb_f_
    else:
        if hasattr(mr, 'Tb'):
            Tb_ = mr.Tb
        else:
            Tb_ = mr.Tb_f
        Tb_f_ = mr.Tb_f
        Tb_s_ = sr.Tb_s
        Tb_f_s_ = sr.Tb_f_s

    return Tb_, Tb_f_, Tb_s_, Tb_f_s_


def vb_from_raw_or_selected(use_raw, mr):
    if use_raw:
        vb_ = mr.vb_hdwe
    else:
        if hasattr(mr, 'vb_f'):
            vb_ = mr.vb_f
        else:
            vb_ = mr.vb
    return vb_


#  Replicate the application in its entirety here.
#  There are no 'bank' parameters anywhere in this model.   It is assumed that all inputs from the application have
#  been converted to the single battery unit 12v form, S1P1, lower-case nomenclature.
# noinspection PyPep8Naming
def replicate(OPT: UserOptions):
    """TODO:
    7. Fig. 9 EKF 2a: hx(soc) negative slope?  This needs to be run just below saturation
    9. Run CompareHistSim etc.
    19. Fig 15 sim_s 2a:  vb?   Keep looking for this when run at other op conditions.  Shutdown problem.
    """
    # Options
    if OPT.run_type == 'RunSim':
        print(OPT)

    # time
    t, dt = sync_to_mon_or_sim(OPT.mon_run, OPT.sim_run, t_mx=OPT.max_time)

    # vb
    vb = vb_from_raw_or_selected(OPT.use_vb_raw, OPT.mon_run)

    # Tb
    Tb, Tb_f, Tb_s, Tb_f_s = Tb_from_raw_or_selected(OPT.use_vb_raw, OPT.mon_run, OPT.sim_run)

    # chem
    chm_m, chm_s = chm_from_mon_or_sim(OPT.mon_run, OPT.sim_run)

    t_len = len(t)
    rp = Retained()

    # modeling
    modeling = get_modeling(OPT.mon_run, OPT.mod_force)

    # tweaking
    tweak_test = rp.tweak_test

    # Translate the off-nominal values imported from data stream
    if hasattr(OPT.mon_run, 'Battery_off_dict'):
        apply_off_nominal_battery(Battery, OPT.mon_run.Battery_off_dict)

    # Instantiate sensors after above translation / over-write
    SN = Sensors(OPT, rp, run_type=OPT.run_type)

    # Battery sizing
    scale_mon, scale_sim = battery_size(OPT.mon_run, OPT.sim_run, OPT.scale_batt, Battery.NOM_UNIT_CAP)

    # Make batteries from modified class constants
    sim = BatterySim(SN=SN, OPT=OPT, mod_code=chm_s[0], Tb_f=SN.sim_run.Tb_f_s[0], scale=scale_sim,
                     tweak_test=tweak_test, vsat_add=Battery.sp_vsat_add)
    mon = BatteryMonitor(SN=SN, OPT=OPT, mod_code=chm_m[0], Tb_f=SN.mon_run.Tb_f[0], scale=scale_mon,
                         vsat_add=Battery.sp_vsat_add, tweak_test=tweak_test)
    Is_sat_delay = TFDelay(in_=OPT.mon_run.soc[0] > 0.97, t_true=T_SAT, t_false=T_DESAT, dt=0.1)  # later, dt is changed

    # Time sync
    if hasattr(OPT.mon_run, 'time_run_start'):
        mon.saved.time_run_start = OPT.mon_run.time_run_start
        sim.saved_s.time_run_start = OPT.mon_run.time_run_start
    else:
        mon.saved.time_run_start = 0.
        sim.saved_s.time_run_start = 0.

    # time loop initialization
    now = t[0]
    reset_ekf = True
    G.i = -1
    i_ekf = -1
    i_temp = -1
    T = OPT.mon_run.dt[0]
    Tpast = T
    hdr = None
    sat_s_init = None

    # Print debug information
    # if OPT.request_history is not None and OPT.request_history > 0:
    #     hdr = print_hist(OPT, SN, i_temp, i_ekf, t, mon, True, True, sim)

    # Top of time loop
    while G.i < t_len-1:
        G.i += 1

        # Time
        now = t[G.i]
        SN.update(G.i)
        T_ekf = None
        if G.i != 0:
            candidate_dt = t[G.i] - t[G.i-1]  # update
            if candidate_dt > 1e-6:
                Tpast = T
                T = dt[G.i]

        # Get temperature data
        if hasattr(OPT.mon_run, 'time_t'):
            n = len(OPT.mon_run.time)
            if hasattr(OPT.mon_run, 'mtb') and OPT.mon_run.mtb is not None \
                    and OPT.mon_run.mtb[G.i]>0. and G.i+1 < n:
                calc_temp = (i_temp+1 < len(OPT.mon_run.time_t)) \
                    and (OPT.mon_run.time_t[i_temp+1] <= OPT.mon_run.time[G.i+1])
            else:
                calc_temp = (i_temp+1 < len(OPT.mon_run.time_t)) and \
                            (OPT.mon_run.time_t[i_temp+1] <= OPT.mon_run.time[min(G.i,n-1)])
        else:
            calc_temp = True
        mon, sim = SN.calc_temp_pass_1(OPT, mon, sim, G.i, rp)

        # Input
        rp.modeling = rp.add_modeling(modeling[G.i])
        mon.tweak_test = rp.tweak_test

        # Basic reset model verification is to init to the input data
        # Tried hard not to re-implement solvers in the Python verification  tool
        # Also, BTW, did not implement signal selection or tweak logic
        # if hasattr(OPT.mon_run, 'kfres'):
        #     mon.reset_kf = bool(OPT.mon_run.kfres[G.i])
        reset = None
        if OPT.run_type == 'RunSim':
            # Must call Battery logic at least twice with reset=True to initialized seeded transfer functions correctly
            reset = bool(G.i < 3 or (t[G.i] <= OPT.init_time) or (t[G.i] < 0. and t[0] > OPT.init_time))
            if OPT.mon_run.reset is not None:
                reset = reset or bool(OPT.mon_run.reset[G.i] > 0.) or bool(OPT.mon_run.reset_all_faults[G.i] > 0.)  # TODO:  reset_all_faults needed here?  Resets Sim while app does not
        elif OPT.run_type == 'HistSim':
            reset = True
        prn_soc_debug(OPT, time=now, leader="before sim init:         ", i_temp=i_temp, mon=mon, sim=sim)
        mon.reset_kf = reset

        if reset:
            if hasattr(OPT.sim_run, 'qcrs_s') and OPT.sim_run.qcrs_s is not None:
                sim.q_cap_rated_scaled = OPT.sim_run.qcrs_s[G.i]
            sim.apply_soc(OPT.mon_run.soc_s[G.i], Tb_f_s[G.i], OPT.sim_run.delta_q_s[G.i])  # calculates delta_q
            prn_soc_debug(OPT, time=now, leader="after sim.apply_soc:     ", i_temp=i_temp, mon=mon, sim=sim)
            sim.load(sim.delta_q)
            sim.assign_tb(sim.Tb)
            sim.assign_tb_f(sim.Tb_f)
            sim.apply_delta_q_t(sim.delta_q, Tb_f_s[G.i])
            prn_soc_debug(OPT, time=now, leader="after sm.apply_delta_q_t:", i_temp=i_temp, mon=mon, sim=sim)
            sat_s_init = SN.voc_stat_init > OPT.mon_run.vsat[G.i]
            if OPT.sim_run is not None:
                sat_s_init = OPT.sim_run.sat_s[G.i]
            sim.sat = sat_s_init
            mon.sat = OPT.mon_run.sat[G.i]
        mon = SN.calc_temp_pass_2(OPT.mon_run, mon, Battery, rp, G.i)
        # Models
        SN.update_ib_vb(G.i)

        if OPT.sim_run is not None and not OPT.use_ib_mon:
            ib_in_s = OPT.sim_run.ib_in_s[G.i]
        else:
            if OPT.run_type == 'RunSim':
                ib_in_s = OPT.mon_run.ib[G.i]
            else:
                ib_in_s = OPT.mon_run.ib_f[G.i]

        if OPT.ib_fail_t is not None and t[G.i] > OPT.ib_fail_t:
            ib_ = OPT.ib_fail
        else:
            if OPT.mon_run.ib_sel is not None:
                ib_ = OPT.mon_run.ib_sel[G.i]
            else:
                ib_ = OPT.mon_run.ib[G.i]

        if OPT.use_vb_sim:
            vb_ = sim.vb
        elif OPT.vb_fail_t and t[G.i] >= OPT.vb_fail_t:
            vb_ = OPT.vb_fail
        else:
            vb_ = vb[G.i]

        if OPT.Bsim is None:
            _chm_s = chm_s[G.i]
        else:
            _chm_s = OPT.Bsim

        sim.calculate(_chm_s, Tb_s[G.i], Tb_f_s[G.i], vb_, ib_in_s, SN.dt_s[G.i], reset, None, None, SN, OPT,
                      soc=sim.soc, q_capacity=sim.q_capacity, rp=rp, saturated_init=sat_s_init)

        sim.count_coulombs(OPT, SN, chem=_chm_s, reset_temp=reset, tb_f=Tb_f_s[G.i], charge_curr=sim.ib_charge,
                           sat=False, saturated=False, mon_sat=mon.sat, rp=rp)

        # Charge init
        if reset:
            if hasattr(SN.mon_run, 'qcrs') and SN.mon_run.qcrs is not None:
                mon.q_cap_rated_scaled = SN.mon_run.qcrs[G.i]
            mon.apply_delta_q_t(SN.mon_run.delta_q[G.i], SN.mon_run.Tb_f[G.i])
            prn_soc_debug(OPT, time=now, leader="after mon.apply_delta_q_t", i_temp=i_temp, mon=mon, sim=sim)
            rp.delta_q = mon.delta_q
            mon.load(rp.delta_q)

        # Chemistry
        if OPT.Bmon is None:
            _chm_m = chm_m[G.i]
        else:
            _chm_m = OPT.Bmon

        # Monitor EKF sequencing logic
        if (i_ekf+1 < len(OPT.mon_run.time_e)) and (OPT.mon_run.time_e[i_ekf+1] <= OPT.mon_run.time[G.i]):
            i_ekf += 1
            reset_ekf = i_ekf == 0 or reset or OPT.run_type == 'HistSim'
            if i_ekf < 1:
                T_ekf = OPT.mon_run.dt_ekf[i_ekf]
            else:
                T_ekf = OPT.mon_run.time_e[i_ekf] - OPT.mon_run.time_e[i_ekf-1]  # update
            calc_ekf = True
        else:
            calc_ekf = False
        SN.update_ekf(max(i_ekf, 0))  # z_init and voc_stat_f_lstate_init

        if reset_ekf and calc_ekf:
            mon.init_soc_ekf(OPT.mon_run, G.i, i_ekf)  # when modeling (assumed in python) ekf wants to equal model

        # Monitor calculate
        mon.calculate(_chm_m, Tb[G.i], Tb_f[G.i], vb_, ib_, T, reset, calc_ekf, T_ekf, SN, OPT, rp=rp, reset_ekf=reset_ekf,
                      i=G.i, i_ekf=i_ekf)
        ib_charge = mon.ib_charge

        if OPT.use_sat_mon:
            sat = OPT.mon_run.sat[G.i]
            saturated = OPT.mon_run.saturated[G.i]
        else:
            sat = is_sat(SN.Tb_f_past, mon.chemistry.rated_temp, mon.voc_dead, mon.soc, mon.chemistry.nom_vsat,
                         mon.chemistry.dvoc_dt, mon.chemistry.low_t, vsat_add=Battery.sp_vsat_add)
            saturated = Is_sat_delay.calculate(sat, T_SAT, T_DESAT, min(T, T_SAT / 2.), reset)

        # Monitor count Coulumbs
        mon.count_coulombs(OPT, chem=_chm_m, dt=T, reset=reset, tb_f=Tb_f[G.i], charge_curr=ib_charge, sat=sat,
                           saturated=saturated)
        prn_soc_debug(OPT, time=now, leader="after mn.count_coulombs: ", i_temp=i_temp, mon=mon, sim=sim)
        mon.calc_charge_time(mon.q, mon.q_capacity, ib_charge, mon.soc)
        # Firmware uses monitor's NOMINAL_TB q_capacity for soc_s during Tb_fa; mirror that here so soc_s tracks soc.
        if hasattr(OPT.mon_run, 'Tb_fa') and bool(OPT.mon_run.Tb_fa[G.i]) and mon.q_capacity and mon.q_capacity != 0.:
            sim.soc = (mon.q_capacity + sim.delta_q) / mon.q_capacity
            print(f'{mon.soc=} {mon.q_capacity=} {mon.delta_q=} {sim.q_capacity=} {sim.delta_q=} {sim.soc=}')
        mon.assign_soc_s(sim.soc)

        # Break if data integrity questionable
        if SN.run_type == 'RunSim':
            if (SN.mon_run.skip_ekf[i_ekf] or SN.mon_run.skip_sel[G.i] or SN.mon_run.skip_mon[G.i]
                    or SN.sim_run.skip_sim[G.i]):
                print(f"Broke early due to skip in one of the data files")
                break

        # Save plot info
        mon.save(t[G.i], T, mon.soc, sim.voc, SN, rp, sim)
        sim.save(t[max(G.i-1,0)], Tpast)
        sim.save_s(t[max(G.i-1,0)])
        Tpast = T

        # Print initial
        if G.i == 0 and OPT.verbose:
            print('time=', t[G.i])
            print('mon:  ', str(mon))
            print('time=', t[G.i])
            print('sim:  ', str(sim))

        # History print
        if OPT.request_history is not None and OPT.request_history > 0:
            hdr = print_hist(OPT, SN, i_temp, i_ekf, t, mon, calc_temp, calc_ekf, sim)

        prn_soc_debug(OPT, time=now, leader="end loop:                ", i_temp=i_temp, mon=mon, sim=sim)

        # Finish loop
        # if calc_ekf:
        #     reset_ekf = False

    # Final hdr print
    if OPT.request_history is not None and OPT.request_history > 0:
        print(hdr)
    if SN.run_type == 'RunSim':
        if (SN.mon_run.skip_ekf[i_ekf] or SN.mon_run.skip_sel[G.i] or SN.mon_run.skip_mon[G.i]
                or SN.sim_run.skip_sim[G.i]):
            print(f"\n\n************** Data integrity degraded by skip.  A digit could have been inserted anywhere in data.  Break.")
            print(f"\nCheck for W too short before vv4 or TEMP_INIT_DELAY or TEMP_DELAY too long in SOC_Particle.ino")
            print("   now {:5.3f}".format(now),
                  "   time_end {:5.3f}\n\n".format(t[-1]),
                  )
            print("at time {:5.3f}\n".format(now),
                  f"skip_ekf {bool(SN.mon_run.skip_ekf[i_ekf])}\n"
                  f"skip_temp {bool(SN.mon_run.skip_temp[i_temp])}\n"
                  f"skip_sel {bool(SN.mon_run.skip_sel[G.i])}\n"
                  f"skip_mon {bool(SN.mon_run.skip_mon[G.i])}\n"
                  f"skip_sim {bool(SN.sim_run.skip_sim[G.i])}\n"
                  f"")
            return None, None, None, None, None, None

    # Data
    if OPT.verbose:
        print('   time mr.chm sr.chm sr.ib_in_s sr.dv_hys  mr.ib mr.soc mr.dv_hys   smv.ib_in_s sim.ibs sim.ioc sim.sat sim.dis sim.dv_dot smv.dv_hys  mv.ib  mv.soc mon.ibs  mon.ioc   mon.sat   mon.dis    mon.dv_dot  mv.dv_hys')
        print('time=', now)
        print('mon:  ', str(mon))
        print('sim:  ', str(sim))

    return mon.saved, sim.saved, sim.saved_s, mon, sim, SN.Battery
