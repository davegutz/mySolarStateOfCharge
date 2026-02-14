# SavedData - data structures
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

""" General data-over-model data structure classes
Dependencies:
    - SavedData  (structures)
"""
from Battery import load_off_nominal_battery, apply_off_nominal_battery
from filter.myFilters import LagExp
from Colors import Colors
import Chemistry_BMS
import numpy as np


class SavedData:
    def __init__(self, battery=None, rap=None, sel=None, ekf=None, temp=None, shunt=None,
                 time_end=None, zero_zero=False, zero_thr=0.02, sync_cTime=None, init_time_in=None, time_shift_in=None,
                 str_=None):
        self.str = str_
        i_end = 0
        n = None
        ib_lag = None
        self.time_shift = time_shift_in

        # Load off-nominal Battery values
        if battery is not None:
            # Scroll through all off-nominals make dictionary
            self.Battery_off_dict = load_off_nominal_battery(Battery_to_add=battery)

        if rap is None:
            IbLag = None
            self.skip_rap = None
            self.i = 0
            self.time = None
            self.reset = None
            self.reset_all_faults = None
            self.soft_reset = None
            self.reset_temp = None
            self.soft_reset_sim = None
            self.init_mon = None
            self.init_sim = None
            self.time_min = None
            self.time_day = None
            self.dt = None  # Update time, s
            self.unit = None  # text title
            self.hm = None  # hours, minutes
            self.cTime = None  # Control time, s
            self.ib = None  # Bank current, A
            self.ib_f = None  # Bank current filtered, A
            self.ioc = None  # Hys indicator current, A
            self.voc = None
            self.voc_soc = None
            # self.ib_past = None  # Past bank current, A
            self.ib_charge = None  # BMS switched current, A
            self.vb = None  # Bank voltage, V
            self.chm = None  # Battery chemistry code
            self.qcrs = None  # Unit capacity rated scaled, Coulombs
            self.d_delta_q = None  # Change in the charge for update, Coulombs
            self.delta_q = None  # Change in the charge for update, Coulombs
            self.q_capacity = None  # Charge capacity at instant, Coulombs
            self.sat = None  # Indication that battery is saturated, T=saturated
            self.ib_lag = None  # Lagged indication that battery is saturated, 1=saturated
            self.sel = None  # Current source selection, 0=amp, 1=no amp
            self.mod = None  # Configuration control code, 0=all hardware, 7=all simulated, +8 tweak test
            self.bms_off = None  # Battery management system off, T=off
            self.Tb_rap = None  # Battery bank temperature, deg C
            self.Tb_f_rap = None  # Battery bank filtered temperature, deg C
            self.Tb_f_rate_rap = None  # Battery bank filtered temperature, deg C
            self.vsat = None  # Monitor Bank saturation threshold at temperature, deg C
            self.dv_dyn = None  # Monitor Bank current induced back emf, V
            self.dv_hys = None  # Drop across hysteresis, V
            self.voc_stat = None  # Monitor Static bank open circuit voltage, V
            self.voc = None  # Bank VOC estimated from vb and RC model, V
            self.voc_ekf = None  # Monitor bank solved static open circuit voltage, V
            self.y_ekf = None  # Monitor single battery solver error, V
            self.y_ekf_f = None  # Monitor single battery solver filtered error, V
            self.soc_s = None  # Simulated state of charge, fraction
            self.soc_ekf = None  # Solved state of charge, fraction
            self.soc = None  # Coulomb Counter fraction of saturation charge (q_capacity_) available (0-1)
            self.time_run_start = 0.  # Adjust time for start of ib input
            self.voc_soc_new = None  # For studies
            self.init_time = None
            self.ib_dyn_r = None
            self.ib_dyn_T = None
            self.ib_dyn_lstate = None
            self.ib_dyn_rstate = None
        else:
            self.skip_rap = np.bool(np.array(rap.skip_mon))
            self.i = 0
            self.cTime = np.array(rap.cTime)
            self.time = np.array(rap.cTime)
            self.reset = np.array(rap.reset)
            self.reset_all_faults = np.array(rap.reset_all_faults)
            self.soft_reset = np.array(rap.soft_reset)
            self.soft_reset_sim = np.array(rap.soft_reset_sim)
            self.init_mon = np.array(rap.init_mon)
            self.init_sim = np.array(rap.init_sim)
            self.reset_temp = np.array(rap.reset_temp)
            self.ib = np.array(rap.ib)
            # manage data shape
            # Find first non-zero ib and use to adjust time
            # Ignore initial run of non-zero ib because resetting from previous run
            if zero_zero:
                self.zero_end = 0
            elif sync_cTime is not None:
                self.zero_end = np.where(self.cTime < sync_cTime[0])[0][-1] + 2
            else:
                try:
                    self.zero_end = 0
                    # stop after first non-zero
                    while self.zero_end < len(self.ib) and abs(self.ib[self.zero_end]) < zero_thr:
                        self.zero_end += 1
                    self.zero_end -= 1  # backup one
                    if self.zero_end == len(self.ib) - 1:
                        print(Colors.fg.red, f"\n\nLikely ib is zero throughout the data.  Check setup and retry\n\n",
                              Colors.reset)
                        self.zero_end = 0
                    elif self.zero_end == -1:
                        print(Colors.fg.red, f"\n\nLikely ib is noisy throughout the data.  Check setup and retry\n\n",
                              Colors.reset)
                        self.zero_end = 0
                except IOError:
                    self.zero_end = 0
            self.time_run_start = self.time[self.zero_end]
            self.time -= self.time_run_start
            self.time_min = self.time / 60.
            self.time_day = self.time / 3600. / 24.

            # Truncate
            i_end = None
            i_end_sel =  None
            i_end_shunt = None
            if time_end is None:
                if temp is None:
                    print(Colors.fg.red, end='')
                    print(f"\n**********\nRun too short, no temp.csv data for {self.time[-1]} s run\n*************\n")
                    print(Colors.reset, end='')
                    exit(0)
                if temp is not None:
                    time_t = np.atleast_1d(np.array(np.array(temp.c_time) - self.time_run_start))
                    Tt = np.atleast_1d(np.array(temp.Tt))
                    if len(Tt) <= 1:
                        print(Colors.fg.red, end='')
                        print(f"\n**********\nRun too short, length Tt = {len(Tt)} for {self.time[-1]} s run.  Need at least 2 samples (asynchronous so time not definitive).\n*************\n")
                        print(Colors.reset, end='')
                        exit(0)
                    time_end = time_t[-1] + Tt[-1]
                    i_end = np.where(self.time <= time_end)[0][-1] + 1
                else:
                    i_end = len(self.time)
                if sel is not None:
                    self.c_time_s = np.array(sel.c_time) - self.time_run_start
                    i_end = min(i_end, len(self.c_time_s))
                if ekf is not None:
                    self.time_e = np.array(np.atleast_1d(ekf.c_time) - self.time_run_start)
                if shunt is not None:
                    self.c_time_shunt = np.array(np.atleast_1d(shunt.c_time) - self.time_run_start)
                    i_end = min(i_end, len(self.c_time_shunt))
            else:
                if temp is not None:
                    time_t = np.atleast_1d(np.array(np.array(temp.c_time) - self.time_run_start))
                    Tt = np.atleast_1d(np.array(temp.T_t))
                    i_end = np.where(self.time <= time_end)[0][-1] + 1
                else:
                    i_end = len(self.time)
                if sel is not None:
                    self.c_time_s = np.array(sel.c_time) - self.time_run_start
                    i_end_sel = np.where(self.c_time_s <= time_end)[0][-1] + 1
                    i_end = np.minimum(i_end, i_end_sel)
                    self.zero_end = np.minimum(self.zero_end, i_end-1)
                if ekf is not None:
                    self.time_e = np.array(np.atleast_1d(ekf.c_time) - self.time_run_start)
                if shunt is not None:
                    self.c_time_shunt = np.array(shunt.c_time) - self.time_run_start
                    i_end_shunt = np.where(self.c_time_shunt <= time_end)[0][-1] + 1
                    i_end = np.minimum(i_end, i_end_shunt)
                    self.zero_end = np.minimum(self.zero_end, i_end-1)


            self.cTime = self.cTime[:i_end]
            self.dt = np.array(rap.dt[:i_end])
            self.time = np.array(self.time[:i_end])
            if self.time_shift:
                self.time += self.time_shift
            self.reset = np.array(rap.reset[:i_end])
            self.reset_all_faults = np.array(rap.reset_all_faults[:i_end])
            self.reset_temp = np.array(rap.reset_temp[:i_end])
            self.soft_reset = np.array(rap.soft_reset[:i_end])
            self.soft_reset_sim = np.array(rap.soft_reset_sim[:i_end])
            self.init_mon = np.array(rap.init_mon[:i_end])
            self.init_sim = np.array(rap.init_sim[:i_end])
            self.ib = np.array(rap.ib[:i_end])
            self.ioc = np.array(rap.ib[:i_end])
            self.voc_soc = np.array(rap.voc_soc[:i_end])
            self.vb = np.array(rap.vb[:i_end])
            self.chm = np.array(rap.chm[:i_end])
            if hasattr(rap, 'qcrs'):
                self.qcrs = rap.qcrs[:i_end]
            if hasattr(rap, 'd_delta_q'):
                self.d_delta_q = rap.d_delta_q[:i_end]
            if hasattr(rap, 'delta_q'):
                self.delta_q = rap.delta_q[:i_end]
            if hasattr(rap, 'qcap'):
                self.q_capacity = rap.qcap[:i_end]
            self.sat = np.array(rap.sat[:i_end])
            # Lag for saturation
            n = len(self.cTime)
            ib_lag = Chemistry_BMS.ib_lag(self.chm[0])
            IbLag = LagExp(1., ib_lag, -100., 100.)
            self.ib_lag = np.zeros(n)
            self.sel = np.array(rap.sel[:i_end])
            self.mod_data = np.array(rap.mod[:i_end])
            self.bms_off = np.array(rap.bmso[:i_end])
            # not_bms_off = self.bms_off < 1
            # bms_off_and_not_charging = self.bms_off * not_bms_off
            # self.ib_charge = self.ib * (bms_off_and_not_charging < 1)
            self.ib_charge = np.array(rap.ib_charge[:i_end])
            self.Tb_rap = np.array(rap.Tb_rap[:i_end])
            self.Tb_f_rap = np.array(rap.Tb_f_rap[:i_end])
            self.Tb_f_rate_rap = np.array(rap.Tb_f_rate_rap[:i_end])
            self.vsat = np.array(rap.vsat[:i_end])
            self.dv_dyn = np.array(rap.dv_dyn[:i_end])
            if hasattr(rap, 'ib_dyn_T'):
                self.ib_dyn_T = np.array(rap.ib_dyn_T[:i_end])
            else:
                self.ib_dyn_T = self.vsat*0.
            if hasattr(rap, 'ib_dyn_r'):
                self.ib_dyn_r = np.bool(np.array(rap.ib_dyn_r[:i_end]))
            else:
                self.ib_dyn_r = np.bool(self.vsat*0.)
            if hasattr(rap, 'ib_dyn_lstate'):
                self.ib_dyn_lstate = np.array(rap.ib_dyn_lstate[:i_end])
            else:
                self.ib_dyn_lstate = self.vsat*0.
            if hasattr(rap, 'ib_dyn_rstate'):
                self.ib_dyn_rstate = np.array(rap.ib_dyn_rstate[:i_end])
            else:
                self.ib_dyn_rstate = self.vsat*0.
            self.ib_dyn = np.array(rap.ib_dyn[:i_end])
            self.voc_stat = np.array(rap.voc_stat[:i_end])
            self.voc = self.vb - self.dv_dyn
            self.dv_hys = np.array(rap.dv_hys[:i_end])
            self.voc_ekf = np.array(rap.voc_ekf[:i_end])
            self.y_ekf = np.array(rap.y_ekf[:i_end])
            self.soc_s = np.array(rap.soc_s[:i_end])
            self.soc_ekf = np.array(rap.soc_ekf[:i_end])
            self.soc = np.array(rap.soc[:i_end])
            self.voc_soc_new = None



        if sel is None:
            pass
        else:
            # Load
            self.assign_all_from(sel, i_end)
            # Specials
            falw = np.array(sel.falw, dtype=np.uint32)
            fltw = np.array(sel.fltw, dtype=np.uint32)
            dispw = np.array(sel.dispw, dtype=np.uint32)
            self.c_time_s = np.array(sel.c_time) - self.time_run_start
            self.ccd_fa = np.bool_(np.array(falw) & 2**4)
            self.ib_diff_flt = np.bool_((np.array(fltw) & 2**8) | (np.array(fltw) & 2**9))
            self.ib_diff_fa = np.bool_((np.array(falw) & 2**8) | (np.array(falw) & 2**9))
            if not hasattr(sel, 'vb_hdwe'):
                self.vb_hdwe = np.array(sel.vb[:i_end])
            if not hasattr(sel, 'vb_hdwe_f'):
                self.vb_hdwe_f = np.array(sel.vb_hdwe[:i_end])
            self.wrap_hi_flt = np.bool_(np.array(fltw) & 2**5)
            self.wrap_lo_flt = np.bool_(np.array(fltw) & 2**6)
            self.wrap_hi_m_flt = np.bool_(np.array(fltw) & 2**14)
            self.wrap_lo_m_flt = np.bool_(np.array(fltw) & 2**15)
            self.wrap_hi_n_flt = np.bool_(np.array(fltw) & 2**16)
            self.wrap_lo_n_flt = np.bool_(np.array(fltw) & 2**17)
            self.wrap_m_and_n_flt = (self.wrap_lo_n_flt & self.wrap_lo_m_flt) | (self.wrap_hi_n_flt & self.wrap_hi_m_flt)
            self.fltw = np.array(fltw)
            self.falw = np.array(falw)
            self.red_loss = np.bool_(np.array(fltw) & 2**7)
            self.wrap_hi_fa = np.bool_(np.array(falw) & 2**5)
            self.wrap_lo_fa = np.bool_(np.array(falw) & 2**6)
            self.wv_fa = np.bool_(np.array(falw) & 2**7)
            self.wrap_hi_m_fa = np.bool_(np.array(falw) & 2**14)
            self.wrap_lo_m_fa = np.bool_(np.array(falw) & 2**15)
            self.wrap_hi_n_fa = np.bool_(np.array(falw) & 2**16)
            self.wrap_lo_n_fa = np.bool_(np.array(falw) & 2**17)
            self.wrap_m_and_n_fa = (self.wrap_lo_n_fa & self.wrap_lo_m_fa) | (self.wrap_hi_n_fa & self.wrap_hi_m_fa)
            self.ib_sel = np.array(sel.ib[:i_end])
            self.dscn_flt = np.bool_(np.array(fltw) & 2**10)
            self.dscn_fa = np.bool_(np.array(falw) & 2**10)
            self.vb_flt = np.bool_(np.array(fltw) & 2**1)
            self.vb_fa = np.bool_(np.array(falw) & 2**1)
            self.tb_flt = np.bool_(np.array(fltw) & 2**0)
            self.tb_fa = np.bool_(np.array(falw) & 2**0)
            self.time_long = np.bool_(np.array(dispw) & 2**11)
            self.accy = np.bool_(np.array(dispw) & 2**10)
            self.off = np.bool_(np.array(dispw) & 2**9)
            self.SAT = np.bool_(np.array(dispw) & 2**8)
            self.flt_ekf = np.bool_(np.array(dispw) & 2**7)
            self.flt_tb = np.bool_(np.array(dispw) & 2**6)
            self.fail_vb = np.bool_(np.array(dispw) & 2**5)
            self.fail_ibm = np.bool_(np.array(dispw) & 2**4)
            self.fail_ib = np.bool_(np.array(dispw) & 2**3)
            self.red_loss = np.bool_(np.array(dispw) & 2**2)
            self.diff_ib = np.bool_(np.array(dispw) & 2**1)
            self.conn = np.bool_(np.array(dispw) & 2**0)
            self.ib_is_functional = np.bool_(np.array(self.ib_is_functional))

        if shunt is None:
            pass
        else:
            #Load
            self.assign_all_from(shunt, i_end)
            # Special handling
            self.c_time_shunt = np.array(shunt.c_time[:i_end]) - self.time_run_start

        if ekf is None:
            pass
        else:
            # Load
            self.assign_all_from_frame(ekf, i_end)
            # Special handling
            self.time_e = np.array(np.atleast_1d(ekf.c_time)[:i_end] - self.time_run_start)

        if temp is None:
            pass
        else:
            # Load
            self.assign_all_from_frame(temp, i_end)
            # Specials
            self.time_t = np.array(np.atleast_1d(temp.c_time)[:i_end]) - self.time_run_start

        # Workarounds for incomplete data sets e.g. vv1, vv2, vv3
        if self.dv_dyn_m is None:
            self.dv_dyn_m = np.copy(self.dv_dyn)
        if self.dv_dyn_n is None:
            self.dv_dyn_n = np.copy(self.dv_dyn)
        if self.ib_amp_hdwe is None:
            self.ib_amp_hdwe = np.copy(self.ib)
        if self.ib_noa_hdwe is None:
            self.ib_noa_hdwe = np.copy(self.ib)
        if self.ib_amp_model is None:
            self.ib_amp_model = np.copy(self.ib)
        if self.ib_noa_model is None:
            self.ib_noa_model = np.copy(self.ib)
        if self.ib_dyn_m is None:
            self.ib_dyn_m = np.copy(self.ib_dyn)
        if self.ib_dyn_lstate_m is None:
            self.ib_dyn_lstate_m = np.copy(self.ib_dyn)
        if self.ib_dyn_lstate_n is None:
            self.ib_dyn_lstate_n = np.copy(self.ib_dyn)
        if self.ib_dyn_rstate_m is None:
            self.ib_dyn_rstate_m = np.copy(self.ib)
        if self.ib_dyn_rstate_n is None:
            self.ib_dyn_rstate_n = np.copy(self.ib)
        if self.ib_dyn_T_m is None:
            self.ib_dyn_T_m = np.copy(self.dt)
        if self.ib_dyn_T_n is None:
            self.ib_dyn_T_n = np.copy(self.dt)
        if self.ib_dyn_tau_m is None:
            self.ib_dyn_tau_m = np.copy(self.dt) * 0. + 10.
        if self.ib_dyn_tau_n is None:
            self.ib_dyn_tau_n = np.copy(self.dt) * 0. + 10.
        if self.ib_dyn_n is None:
            self.ib_dyn_n = np.copy(self.ib_dyn)
        if self.ib_dec is None:
            self.ib_dec = np.copy(self.ib) * 0
        if self.ib_sel is None:
            self.ib_sel = np.copy(self.ib)
        if self.ib_sel_stat is None:
            self.ib_sel_stat = np.copy(self.ib) * 0
        if self.ib_choice is None:
            self.ib_choice = np.copy(self.ib) * 0
        if self.ib_h is None:
            self.ib_h = np.copy(self.ib)
        if self.ib_s is None:
            self.ib_s = np.copy(self.ib)
        if self.ib_wrp_reset_m is None:
            self.ib_wrp_reset_m = np.copy(self.dt) * 0
        if self.ib_wrp_rate_m is None:
            self.ib_wrp_rate_m = np.copy(self.dt) * 0.
        if self.ib_wrp_state_m is None:
            self.ib_wrp_state_m = np.copy(self.dt) * 0.
        if self.ib_wrp_T_m is None:
            self.ib_wrp_T_m = np.copy(self.dt)
        if self.ib_wrp_tau_m is None:
            self.ib_wrp_tau_m = np.copy(self.dt) * 0. + 10.
        if self.ib_wrp_rate_n is None:
            self.ib_wrp_rate_n = np.copy(self.dt) * 0.
        if self.ib_wrp_state_n is None:
            self.ib_wrp_state_n = np.copy(self.dt) * 0.
        if self.ib_wrp_T_n is None:
            self.ib_wrp_T_n = np.copy(self.dt)
        if self.ib_wrp_tau_n is None:
            self.ib_wrp_tau_n = np.copy(self.dt) * 0. + 10.
        if self.e_wrap_m is None:
            self.e_wrap_m = np.copy(self.ib) * 0.
        if self.e_wrap_m_filt is None:
            self.e_wrap_m_filt = np.copy(self.ib) * 0.
        if self.e_wrap_m_reset is None:
            self.e_wrap_m_reset = np.copy(self.ib) * 0
        if self.e_wrap_m_trim is None:
            self.e_wrap_m_trim = np.copy(self.ib) * 0.
        if self.ib_amp is None:
            self.ib_amp = np.copy(self.ib) * 0.
        if self.e_wrap_n is None:
            self.e_wrap_n = np.copy(self.ib) * 0.
        if self.e_wrap_n_filt is None:
            self.e_wrap_n_filt = np.copy(self.ib) * 0.
        if self.e_wrap is None:
            self.e_wrap = np.copy(self.ib) * 0.
        if self.e_wrap_filt is None:
            self.e_wrap_filt = np.copy(self.ib) * 0.
        if self.mvb is None:
            self.mvb = np.bool(np.copy(self.mod_data))
        if self.Tb is None:
            self.Tb = np.copy(self.Tb_rap)
        if self.Tb_f is None:
            self.Tb_f = np.copy(self.Tb_f_rap)
        if self.Tb_f_rate is None:
            self.Tb_f_rate = np.copy(self.Tb_f_rate_rap)
        if self.Tb_hdwe is None:
            self.Tb_hdwe = np.copy(self.Tb_rap)
        if self.Tb_hdwe_filt_rate is None:
            self.Tb_hdwe_filt_rate = np.copy(self.Tb_f_rate_rap)
        if self.Tb_model_filt_rate is None:
            self.Tb_model_filt_rate = np.copy(self.Tb_f_rate_rap)
        if self.Tb_hdwe_filt is None:
            print(f"Using Tb_f_rap to initialize Tb_hdwe_filt")
            self.Tb_hdwe_filt = np.copy(self.Tb_f_rap)
        if self.Tb_model is None:
            self.Tb_model = np.copy(self.Tb_rap)
        if self.Tb_model_filt is None:
            print(f"Using Tb_f_rap to initialize Tb_model_filt")
            self.Tb_model_filt = np.copy(self.Tb_f_rap)
        if self.dt_ekf is None:
            self.dt_ekf = np.copy(self.dt)
        if self.vb_hdwe is None:
            self.vb_hdwe = np.copy(self.vb)
        if self.x is None:
            self.x = np.copy(self.soc_ekf)
        if self.x_prior is None:
            self.x_prior = np.copy(self.soc_ekf)
        if self.x_post is None:
            self.x_post = np.copy(self.soc_ekf)
        if self.y_ekf is None:
            self.y_ekf = np.copy(self.voc_stat) * 0.
        if self.y_ekf_f is None:
            self.y_ekf_f = np.copy(self.voc_stat) * 0.
        if self.z is None:
            self.z = np.copy(self.voc_stat)
        if self.H is None:
            self.H = np.copy(self.voc_stat)
        if self.hx is None:
            self.hx = np.copy(self.voc_stat)
        if self.K is None:
            self.K = np.copy(self.x) * 0.
        if self.P is None:
            self.P = np.copy(self.x) * 0.
        if self.P_post is None:
            self.P_post = np.copy(self.x) * 0.
        if self.P_prior is None:
            self.P_prior = np.copy(self.x) * 0.
        if self.Q is None:
            self.Q = np.copy(self.x) * 0.
        if self.S is None:
            self.S = np.copy(self.x) * 0.
        if self.tb_f_for_hx is None:
            self.tb_f_for_hx = np.copy(self.Tb_f)
        if self.x_for_hx is None:
            self.x_for_hx = np.copy(self.x)
        if self.disable_amp_fault is None:
            self.disable_amp_fault = np.copy(self.ib) * 0
        if self.time_e is None:
            self.time_e = np.copy(self.dt)
        if self.time_t is None:
            self.time_t = np.copy(self.dt)

        # Initialization time logic
        if init_time_in:
            self.init_time = init_time_in
        else:
            if self.time[0] == 0.:  # no initialization flat detected at beginning of recording
                self.init_time = 1.
            else:
                self.init_time = -4.

        for i in range(n):
            if self.time[i] <= self.init_time:
                lag_reset = True
                if i < n-1:
                    T_lag = self.cTime[i+1] - self.cTime[i]
                else:
                    T_lag = self.cTime[i] - self.cTime[i-1]
            else:
                lag_reset = False
                T_lag = self.cTime[i] - self.cTime[i-1]
            self.ib_lag[i] = IbLag.calculate_tau(float(self.ib[i]), lag_reset, T_lag, ib_lag)

    def assign_all_from(self, x=None, i_end=None):
        """
        Iterates over members of a dataset x, assigns values to numpy.ndarray members
        """
        for name in list(x.dtype.names):
            if i_end is None:
                setattr(self, name, x[name])
            else:
                setattr(self, name, getattr(x, name)[:i_end])

    def assign_all_from_frame(self, x=None, i_end=None):
        """
        Iterates over members of a dataset x, assigns values to numpy.ndarray members
        """
        # self.Fx = np.array(np.atleast_1d(ekf.Fx_)[:i_end])

        for name in list(x.dtype.names):
            if i_end is None:
                setattr(self, name, np.array(np.atleast_1d(x[name])))
            else:
                setattr(self, name, np.array(np.atleast_1d(getattr(x, name)[:i_end])))

    def truncate(self, i_end=None, key_attr='time'):
        """
        Iterates over members of a self, assigns values to numpy.ndarray members
        up to i_end.
        """
        for attr_name in dir(self):
            # Filter out built-in attributes and methods
            if not attr_name.startswith('__') and not callable(getattr(self, attr_name)):
                member = getattr(self, attr_name)
                if isinstance(member, np.ndarray):
                    # Ensure the slice doesn't exceed the bounds of rap_self.ib
                    end_index = min(i_end, len(getattr(self, key_attr)))

                    # Assign the slice to the numpy.ndarray member
                    # If the target array has a different shape, direct assignment
                    # might fail or reshape the array. Using np.array() ensures
                    # a new array is created with the correct slice.
                    setattr(self, attr_name, getattr(self, attr_name)[:end_index])

    def __str__(self):
        s = "{},".format(self.unit[self.i])
        s += "{},".format(self.hm[self.i])
        # s += "{:13.3f},".format(self.cTime[self.i])
        s += "{:8.3f},".format(self.ib[self.i])
        s += "{:7.2f},".format(self.vsat[self.i])
        s += "{:5.2f},".format(self.dv_dyn[self.i])
        s += "{:5.2f},".format(self.voc_stat[self.i])
        s += "{:5.2f},".format(self.voc_ekf[self.i])
        s += "{:10.6f},".format(self.y_ekf[self.i])
        s += "{:7.3f},".format(self.soc_s[self.i])
        s += "{:5.3f},".format(self.soc_ekf[self.i])
        s += "{:5.3f},".format(self.soc[self.i])
        return s

    def mod(self):
        return self.mod_data[self.zero_end]


class SavedDataSim:
    def __init__(self, time_run_start, data=None, time_end=None, fake=False, mon_for_fake=None, str_=None):
        self.str = str_
        if data is None:
            pass
        else:
            self.cTime = np.array(data.c_time)
            self.time = self.cTime  - time_run_start
            if time_end is None:
                i_end = len(self.time)
            else:
                i_end = np.where(self.time <= time_end)[0][-1] + 1
            self.time_min = self.time / 60.
            self.time_day = self.time / 3600. / 24.
            self.i = 0
            self.assign_all_from(data, i_end)

            # Auxiliary parameters
            self.voc_s = self.vb_s - self.dv_dyn_s

            # Truncate
            self.truncate(i_end=i_end)

        if fake:
            self.ib_in_s = np.copy(mon_for_fake.ib)
            self.ib_dyn_s = np.copy(mon_for_fake.ib_dyn)
            self.time = np.copy(mon_for_fake.time)
            self.dv_dyn_s = np.copy(mon_for_fake.dv_dyn)
            self.dv_hys_s = np.copy(mon_for_fake.dv_hys)
            self.Tb_hdwe = np.copy(mon_for_fake.Tb_rap)
            self.delta_q_s = np.copy(mon_for_fake.delta_q)
            self.delta_q_s = np.copy(mon_for_fake.delta_q)
            self.voc_stat_s = np.copy(mon_for_fake.voc_stat)
            self.qcrs_s = np.copy(mon_for_fake.qcrs)
            self.chm_s = np.copy(mon_for_fake.chm)
            self.sat_s = np.copy(mon_for_fake.sat)
            self.soc_s = np.copy(mon_for_fake.soc_s)
            self.dt_s = np.copy(mon_for_fake.dt)
            self.bms_off_s = np.copy(mon_for_fake.bms_off)
            self.mod_tb = np.bool(np.copy(mon_for_fake.mod_data))

    def assign_all_from(self, x=None, i_end=None):
        """
        Iterates over members of a dataset x, assigns values to numpy.ndarray members
        """
        for name in list(x.dtype.names):
            if i_end is None:
                setattr(self, name, x[name])
            else:
                setattr(self, name, getattr(x, name)[:i_end])

    def truncate(self, i_end=None, key_attr='time'):
        """
        Iterates over members of a self, assigns values to numpy.ndarray members
        up to i_end.
        """
        for attr_name in dir(self):
            # Filter out built-in attributes and methods
            if not attr_name.startswith('__') and not callable(getattr(self, attr_name)):
                member = getattr(self, attr_name)
                if isinstance(member, np.ndarray):
                    # Ensure the slice doesn't exceed the bounds of rap_self.ib
                    end_index = min(i_end, len(getattr(self, key_attr)))

                    # Assign the slice to the numpy.ndarray member
                    # If the target array has a different shape, direct assignment
                    # might fail or reshape the array. Using np.array() ensures
                    # a new array is created with the correct slice.
                    setattr(self, attr_name, getattr(self, attr_name)[:end_index])

    def __str__(self):
        s = "{},".format(self.unit[self.i])
        # s += "{:13.3f},".format(self.cTime[self.i])
        # s += "{:5.2f},".format(self.Tb_s[self.i])
        s += "{:8.3f},".format(self.vsat_s[self.i])
        s += "{:5.2f},".format(self.voc_stat_s[self.i])
        s += "{:5.2f},".format(self.dv_dyn_s[self.i])
        s += "{:5.2f},".format(self.vb_s[self.i])
        s += "{:8.3f},".format(self.ib_s[self.i])
        s += "{:8.3f},".format(self.ib_dyn_s[self.i])
        s += "{:7.3f},".format(self.sat_s[self.i])
        # s += "{:5.3f},".format(self.ddq_s[self.i])
        s += "{:5.3f},".format(self.delta_q_s[self.i])
        # s += "{:5.3f},".format(self.qcap_s[self.i])
        s += "{:7.3f},".format(self.soc_s[self.i])
        s += "{:d},".format(self.reset_s[self.i])
        return s
