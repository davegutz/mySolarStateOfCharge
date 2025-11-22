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
from DataOverModel import SavedData as SavedData
from DataOverModel import SavedDataSim as SavedDataSim
from MonSimNomConfig import *  # Global config parameters.   Overwrite in your own calls for studies
from Battery import Battery, BatteryMonitor, BatterySim, is_sat, Retained
from dataclasses import dataclass
from Battery import overall_batt
from typing import Optional
from TFDelay import TFDelay
from MonSimClasses import *
from MonSimPrint import *
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
        scale_sim_ = unit_cap_rated / unit_cap_rated_
        if scale_in_:
            scale_sim_ *= scale_in_
    return scale_mon_, scale_sim_


def chm_from_mon_or_sim(mr, sr):
    chem_m = mr.chm
    if sr is not None:
        chem_s = sr.chm_s
    else:
        chem_s = mr.chm_m
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
        time = sr.time
        dtime = sr.dt_s
    else:
        time = mr.time
        dtime = mr.dt
    if t_mx is not None:
        t_delt = time - time[0]
        time = time[np.where(t_delt <= t_mx)]
        dtime = dtime[np.where(t_delt <= t_mx)]
    return time, dtime

def vb_from_raw_or_selected(use_raw, mr):
    if use_raw:
        vb_ = mr.vb_h
    else:
        if hasattr(mr, 'vb_f'):
            vb_ = mr.vb_f
        else:
            vb_ = mr.vb
    return vb_

@dataclass
class UserOptions:
    mon_run: SavedData  # Mandatory reference data to be replicated
    run_type: Optional[str] = None  # Either "RunSim" or "HistSim" depending on caller
    sim_run: Optional[SavedDataSim] = None  # Embedded model data
    unit: Optional[str] = None  # Name of the battery instance derived from 'HDWE_UNIT' of configuration include .h file
    Bsim: Optional[int] = None  # sim model code BB=0 (Battleborn), CH=1 (Chins), CHG=2 (Chins in Garage)
    Bmon: Optional[int] = None  # mon model code BB=0 (Battleborn), CH=1 (Chins), CHG=2 (Chins in Garage)
    init_time: Optional[float] = -4.  # The process tries to determine mon_run.init_time when data is loaded by finding
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
    request_history: Optional[int] = 5  # Print simulation history (0 - 5) to check overplot using data in addition
    use_ib_mon: Optional[bool] = False  # Drive BatterySim directly with the BatteryMonitor input, useful when raw sim data not available
    use_sat_mon: Optional[bool] = False  # Drive entire model directly with the run input, useful for HistSim unable to accurately run sliding deadbanc
    use_mon_soc: Optional[bool] = False  # Drive SOC of the model directly with data to focus on modeling that is downstream of SOC
    use_vb_raw: Optional[bool] = False  # Force usage of raw Vb bypassing the signal selection logic
    verbose: Optional[bool] = True  # Lots of 'helpful' information used to provide some quick clues about whatever
    # to or instead of plots
    mod_force: Optional[int] = None  # Force modeling config that cannot be gleaned from input data or other reason
    IB_CHARGE_NOA: Optional[bool] = False  # Force use of ib_noa in coulomb counting but not signal selection or calculate

#  Replicate the application in its entirety here.
#  There are no 'bank' parameters anywhere in this model.   It is assumed that all inputs from the application have
#  been converted to the single battery unit 12v form, S1P1, lower-case nomenclature.
def replicate(OPT: UserOptions):
    """TODO:
    7. Fig. 9 EKF 2a: hx(soc) negative slope?  This needs to be run just below saturation
    9. Run CompareHistSim etc.
    19. Fig 15 sim_s 2a:  vb?   Keep looking for this when run at other op conditions.  Shutdown problem.
    """
    # Options
    print(OPT)

    # time
    t, dt = sync_to_mon_or_sim(OPT.mon_run, OPT.sim_run, t_mx=OPT.max_time)

    # vb
    vb = vb_from_raw_or_selected(OPT.use_vb_raw, OPT.mon_run)

    # chem
    chm_m, chm_s = chm_from_mon_or_sim(OPT.mon_run, OPT.sim_run)

    t_len = len(t)
    rp = Retained()

    # modeling
    modeling = get_modeling(OPT.mon_run, OPT.mod_force)

    # tweaking
    tweak_test = rp.tweak_test()

    SN = Sensors(OPT, run_type=OPT.run_type)

    # Battery sizing
    scale_mon, scale_sim = battery_size(OPT.mon_run, OPT.sim_run, OPT.scale_in, Battery.NOM_UNIT_CAP)

    # Translate the off-nominal values imported from data stream
    if hasattr(OPT.mon_run, 'Battery_off_dict'):
        print("Over-writing pre-existing off-nominal values into Battery class structure")
        for key in dir(Battery):
            if key.isupper() and not key.startswith('__'):
                if key in OPT.mon_run.Battery_off_dict:
                    print(f"Battery.{key} {getattr(Battery, key)} --> ", end='')
                    setattr(Battery, key, OPT.mon_run.Battery_off_dict[key])
                    print(f" {getattr(Battery, key)}")

    # Make batteries from modified class constants
    sim = BatterySim(SN=SN, OPT=OPT, mod_code=chm_s[0], tb_f=SN.Tb0_s, scale=scale_sim, tweak_test=tweak_test)
    mon = BatteryMonitor(SN=SN, OPT=OPT, mod_code=chm_m[0], tb_f=SN.Tb0, scale=scale_mon, tweak_test=tweak_test)
    Is_sat_delay = TFDelay(in_=OPT.mon_run.soc[0] > 0.97, t_true=T_SAT, t_false=T_DESAT, dt=0.1)  # later, dt is changed

    # Time sync
    if hasattr(OPT.mon_run, 'time_run'):
        mon.saved.time_run = OPT.mon_run.time_run
        sim.saved_s.time_run = OPT.mon_run.time_run
    else:
        mon.saved.time_run = 0.
        sim.saved_s.time_run = 0.

    # time loop initialization
    now = t[0]
    dtnow = dt[0]
    reset_ekf = True
    G.i = -1
    i_ekf = -1
    i_temp = -1
    T = OPT.mon_run.dt[0]
    hdr = None
    sat_s_init = None

    # Print debug information
    if OPT.request_history is not None and OPT.request_history > 0:
        hdr = print_hist(OPT, SN, i_temp, i_ekf, t, mon, True, True, sim)

    # Top of time loop
    while G.i < t_len-1:
        G.i += 1

        if G.i >= 206:
            pass  # used for debug breakpoint at i >= <val>

        # Time
        now =t [G.i]
        SN.update(G.i)
        T_ekf = None
        if G.i != 0:
            candidate_dt = t[G.i] - t[G.i-1]  # update
            # print(f"{t[G.i]=} {t[G.i-1]=} {candidate_dt=}")
            if candidate_dt > 1e-6:
                T = dt[G.i]

        # Get temperature data
        if hasattr(OPT.mon_run, 'time_t'):
            calc_temp = (i_temp+1 < len(OPT.mon_run.time_t)) and (OPT.mon_run.time_t[i_temp+1] <= OPT.mon_run.time[G.i])
        else:
            calc_temp = True
        if calc_temp:
            i_temp += 1
            mon, sim = SN.calc_temp_pass_1(OPT, mon, sim, i_temp)

        # Input
        rp.modeling = modeling[G.i]

        # Basic reset model verification is to init to the input data
        # Tried hard not to re-implement solvers in the Python verification  tool
        # Also, BTW, did not implement signal selection or tweak logic
        reset = None
        if OPT.run_type == 'RunSim':
            reset = bool((t[G.i] <= OPT.init_time) or (t[G.i] < 0. and t[0] > OPT.init_time))
            if OPT.mon_run.res is not None:
                reset = reset or bool(OPT.mon_run.res[G.i] > 0.)
        elif OPT.run_type == 'HistSim':
            reset = True
        prn_soc_debug(OPT, time=now, leader="before sim init:         ", i_temp=i_temp, mon=mon, sim=sim)

        if reset:
            sim.apply_soc(OPT.mon_run.soc_s[G.i], SN.Tb_f_past)  # calculates delta_q
            prn_soc_debug(OPT, time=now, leader="after sim.apply_soc:     ", i_temp=i_temp, mon=mon, sim=sim)
            sim.load(sim.delta_q)
            sim.assign_tb(sim.Tb)
            sim.assign_tb_f(sim.Tb_f)
            sim.apply_delta_q_t(sim.delta_q, SN.Tb_f_past)
            prn_soc_debug(OPT, time=now, leader="after sm.apply_delta_q_t:", i_temp=i_temp, mon=mon, sim=sim)
            sat_s_init = SN.voc_stat_init > OPT.mon_run.vsat[0]
            if OPT.sim_run is not None:
                sat_s_init = OPT.sim_run.sat_s[0]
            sim.sat = sat_s_init
            mon.sat = OPT.mon_run.sat[0]

        if calc_temp:
            mon = SN.calc_temp_pass_2(OPT.mon_run, mon, Battery, i_temp)

        # Models
        SN.update_ib_vb(G.i)

        if OPT.sim_run is not None and not OPT.use_ib_mon:
            ib_in_s = OPT.sim_run.ib_in_s[G.i]
        else:
            if OPT.run_type == 'RunSim':
                ib_in_s = OPT.mon_run.ib[G.i]
            else:
                ib_in_s = OPT.mon_run.ib_f[G.i]

        if OPT.Bsim is None:
            _chm_s = chm_s[G.i]
        else:
            _chm_s = OPT.Bsim

        sim.calculate(_chm_s, None, ib_in_s, SN.dt_s[G.i], reset, None, None, SN, OPT,
                      soc=sim.soc, q_capacity=sim.q_capacity, rp=rp, sat_init=sat_s_init)

        sim.count_coulombs(OPT, SN, chem=_chm_s, reset_temp=reset, tb_f=sim.Tb_f, charge_curr=sim.ib_charge, sat=False,
                           mon_sat=mon.sat)

        # EKF
        if reset:
            mon.apply_delta_q_t(SN.delta_q[G.i], SN.Tb_f_rap[G.i])
            prn_soc_debug(OPT, time=now, leader="after mon.apply_delta_q_t", i_temp=i_temp, mon=mon, sim=sim)
            rp.delta_q = mon.delta_q
            mon.load(rp.delta_q)

        # Chemistry
        if OPT.Bmon is None:
            _chm_m = chm_m[G.i]
        else:
            _chm_m = OPT.Bmon

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
        SN.update_ekf(i_ekf)

        if reset_ekf and calc_ekf:
            mon.init_soc_ekf(OPT.mon_run, G.i, i_ekf)  # when modeling (assumed in python) ekf wants to equal model

        # Monitor calculate
        mon.calculate(_chm_m, vb_, ib_, T, reset, calc_ekf, T_ekf, SN, OPT, rp=rp, reset_ekf=reset_ekf, i=G.i)
        ib_charge = mon.ib_charge

        if OPT.use_sat_mon:
            saturated = OPT.mon_run.sat[G.i]
        else:
            sat = is_sat(SN.Tb_f_past, mon.chemistry.rated_temp, mon.voc_dead, mon.soc, mon.chemistry.nom_vsat,
                         mon.chemistry.dvoc_dt, mon.chemistry.low_t)
            saturated = Is_sat_delay.calculate(sat, T_SAT, T_DESAT, min(T, T_SAT / 2.), reset)

        # Monitor count Coulumbs
        mon.count_coulombs(OPT, chem=_chm_m, dt=T, reset=reset, tb_f=SN.Tb_f_past, charge_curr=ib_charge, sat=saturated)
        prn_soc_debug(OPT, time=now, leader="after mn.count_coulombs: ", i_temp=i_temp, mon=mon, sim=sim)
        mon.calc_charge_time(mon.q, mon.q_capacity, ib_charge, mon.soc)
        mon.assign_soc_s(sim.soc)

        # Break if data integrity questionable
        if SN.skip_e[i_ekf] or SN.skip_t[i_temp] or SN.skip_sel[G.i] or SN.skip_rap[G.i] or SN.skip_s[G.i]:
            break

        # Save plot info
        mon.save(t[G.i], T, mon.soc, sim.voc)
        sim.save(t[G.i], T)
        sim.save_s(t[G.i])

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

        # pick a pass to run debugger to a time
        if G.i >= 206:
            pass  # used for debug breakpoint at i >= <val>
        if now > 2:
            pass  # used for debug breakpoint at now > <val>
        else:
            pass

        # Finish loop
        # if calc_ekf:
        #     reset_ekf = False

    # Final hdr print
    if OPT.request_history is not None and OPT.request_history > 0:
        print(hdr)
    if SN.skip_e[i_ekf] or SN.skip_t[i_temp] or SN.skip_sel[G.i] or SN.skip_rap[G.i] or SN.skip_s[G.i]:
        print(f"\n\n************** Data integrity degraded by skip.  A digit could have been inserted anywhere in data.  Break.")
        print("   now {:5.3f}".format(now),
              "   time_end {:5.3f}\n\n".format(t[-1]),
              )

    # Data
    if OPT.verbose:
        print('   time mr.chm sr.chm sr.ib_in_s sr.dv_hys  mr.ib mr.soc mr.dv_hys   smv.ib_in_s sim.ibs sim.ioc sim.sat sim.dis sim.dv_dot smv.dv_hys  mv.ib  mv.soc mon.ibs  mon.ioc   mon.sat   mon.dis    mon.dv_dot  mv.dv_hys')
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
        mon_run_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)

        # Load sel (old)
        sel_file_clean = write_clean_file(data_file_old_txt, type_='_sel', hdr_key=hdr_key_sel,
                                          unit_key=unit_key_sel, skip=skip)
        sel_old_raw = None
        if sel_file_clean:
            sel_old_raw = np.genfromtxt(sel_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
        mon_run = SavedData(rap=mon_run_raw, sel=sel_old_raw, time_end=time_end, zero_zero=zero_zero_in,
                            zero_thr=zero_thr_in, init_time_in=init_time_in)

        # Load _m v24 portion of real-time run (old)
        data_file_sim_clean = write_clean_file(data_file_old_txt, type_='_sim', hdr_key=hdr_key_sim,
                                               unit_key=unit_key_sim, skip=skip)
        if data_file_sim_clean:
            sim_run_raw = np.genfromtxt(data_file_sim_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
            sim_run = SavedDataSim(time_run=mon_run.time_run, data=sim_run_raw, time_end=time_end)
        else:
            sim_run = None

        # New run
        mon_file_save = data_file_clean.replace(".csv", "_rep.csv")

        replicateOptions = UserOptions(mon_run=mon_run, sim_run=sim_run, Bmon=Bmon_in, Bsim=Bsim_in,
                                       init_time=mon_run.init_time, use_ib_mon=use_ib_mon_in,
                                       use_vb_raw=use_vb_raw, add_voc_sim=dvoc_sim_in,
                                       add_voc_mon=dvoc_mon_in, slr_r_ss=scale_r_ss_in,
                                       scale_in=scale_in, ib_fail_t=ib_fail_t)
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
