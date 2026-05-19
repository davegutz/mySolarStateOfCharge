# SavedData - data structures
# Copyright (C) 2026 Dave Gutz
#
# noinspection PyAttributeOutsideInit,PyUnresolvedReferences,PyPep8Naming
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

""" General data-over-model data structure classes for importing data from Particle Photon runs and simulations,
and for managing data for plotting.
Dependencies:
    - SavedData  (structures)
"""
from battery_constants import load_off_nominal_battery
from filter.myFilters import LagExp
from Colors import Colors
import Chemistry_BMS
import numpy as np


# type: ignore
class SavedData:
    # noinspection PyPep8Naming
    def __init__(self, battery=None, rap=None, sel=None, ekf=None, shunt=None, time_end=None, zero_zero=False,
                 zero_thr=0.02, sync_cTime=None, init_time=None, time_shift=None, str_=None):
        self.str = str_
        i_end = 0
        n = 0
        ib_lag = 0.
        IbLag = None
        self.time_shift = time_shift

        # Load off-nominal Battery values
        if battery is not None:
            # Scroll through all off-nominals make dictionary
            self.Battery_off_dict = load_off_nominal_battery(Battery_to_add=battery)

        if rap is None:
            pass
        else:
            # Load
            self.assign_all_from(rap)

            # Specials
            self.i = 0
            self.time = np.array(self.cTime)

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
                        self.zero_end = 2
                    elif self.zero_end == -1:
                        print(Colors.fg.red, f"\n\nLikely ib is noisy throughout the data.  Check setup and retry\n\n",
                              Colors.reset)
                        self.zero_end = 0
                except IOError:
                    self.zero_end = 0
            self.time_run_start = self.time[self.zero_end]
            self.time -= self.time_run_start

            # Truncate
            if time_end is None:
                i_end = len(self.time)
                if sel is not None:
                    self.c_time_sel = np.array(sel.c_time_sel) - self.time_run_start
                    i_end = min(i_end, len(self.c_time_sel))
                if ekf is not None:
                    self.time_e = np.array(np.atleast_1d(ekf.c_time_e) - self.time_run_start)
                if shunt is not None:
                    self.c_time_shunt = np.array(np.atleast_1d(shunt.c_time) - self.time_run_start)
                    i_end = min(i_end, len(self.c_time_shunt))
            else:
                i_end = len(self.time)
                if sel is not None:
                    self.c_time_sel = np.array(sel.c_time_sel) - self.time_run_start
                    i_end_sel = np.where(self.c_time_sel <= time_end)[0][-1] + 1
                    i_end = np.minimum(i_end, i_end_sel)
                    self.zero_end = np.minimum(self.zero_end, i_end-1)
                if ekf is not None:
                    self.time_e = np.array(np.atleast_1d(ekf.c_time_e) - self.time_run_start)
                if shunt is not None:
                    self.c_time_shunt = np.array(shunt.c_time_shunt) - self.time_run_start
                    i_end_shunt = np.where(self.c_time_shunt <= time_end)[0][-1] + 1
                    i_end = np.minimum(i_end, i_end_shunt)
                    self.zero_end = np.minimum(self.zero_end, i_end-1)

            # Load again with new i_end
            self.assign_all_from(rap, i_end=i_end)
            self.time = np.array(self.cTime) - self.time_run_start
            self.time_min = self.time / 60.
            self.time_day = self.time / 3600. / 24.
            if self.time_shift:
                self.time += self.time_shift
            self.ioc = np.array(rap.ib[:i_end])
            if hasattr(rap, 'qcap'):
                self.q_capacity = rap.qcap[:i_end]
            # Lag for saturation
            n = len(self.cTime)
            ib_lag = Chemistry_BMS.ib_lag(self.chm[0])
            IbLag = LagExp(1., ib_lag, -100., 100.)
            self.ib_lag = np.zeros(n)
            # self.sel = np.array(rap.sel[:i_end])
            self.mod_data = np.array(rap.mod[:i_end])
            self.bms_off = np.array(rap.bmso[:i_end])
            if not hasattr(rap, 'ib_dyn_T'):
                self.ib_dyn_T = self.vsat*0.
            if not hasattr(rap, 'ib_dyn_r'):
                self.ib_dyn_r = np.bool(self.vsat*0.)
            if not hasattr(rap, 'ib_dyn_lstate'):
                self.ib_dyn_lstate = self.vsat*0.
            if not hasattr(rap, 'ib_dyn_rstate'):
                self.ib_dyn_rstate = self.vsat*0.
            self.voc = self.vb - self.dv_dyn
            self.voc_soc_new = None


        if sel is None:
            pass
        else:
            # Load
            self.assign_all_from(sel, i_end)
            # Specials
            falw = np.array(self.falw, dtype=np.uint32)
            fltw = np.array(self.fltw, dtype=np.uint32)
            dispw = np.array(self.dispw, dtype=np.uint32)
            self.c_time_sel = np.array(self.c_time_sel) - self.time_run_start
            self.ccd_fa = np.bool_(np.array(falw) & 2**4)
            self.ib_diff_flt = np.bool_((np.array(fltw) & 2**8) | (np.array(fltw) & 2**9))
            self.ib_diff_fa = np.bool_((np.array(falw) & 2**8) | (np.array(falw) & 2**9))
            if not hasattr(sel, 'vb_hdwe'):
                self.vb_hdwe = np.array(self.vb)
            if not hasattr(sel, 'vb_hdwe_f'):
                self.vb_hdwe_f = np.array(self.vb_hdwe)
            self.wrap_hi_flt = np.bool_(np.array(fltw) & 2**5)
            self.wrap_lo_flt = np.bool_(np.array(fltw) & 2**6)
            self.vc_flt = np.bool_(np.array(fltw) & 2**13)
            self.wrap_hi_m_flt = np.bool_(np.array(fltw) & 2**14)
            self.wrap_lo_m_flt = np.bool_(np.array(fltw) & 2**15)
            self.wrap_hi_n_flt = np.bool_(np.array(fltw) & 2**16)
            self.wrap_lo_n_flt = np.bool_(np.array(fltw) & 2**17)
            self.Tb_flt = np.bool_(fltw & 2 ** 19)
            self.wrap_m_and_n_flt = (self.wrap_lo_n_flt & self.wrap_lo_m_flt) | (self.wrap_hi_n_flt & self.wrap_hi_m_flt)
            self.fltw = np.array(fltw)
            self.falw = np.array(falw)
            self.dispw = np.array(dispw)
            self.red_loss = np.bool_(np.array(fltw) & 2**7)
            self.wrap_hi_fa = np.bool_(np.array(falw) & 2**5)
            self.wrap_lo_fa = np.bool_(np.array(falw) & 2**6)
            self.wv_fa = np.bool_(np.array(falw) & 2**7)
            self.vc_fa = np.bool_(np.array(fltw) & 2**13)
            self.wrap_hi_m_fa = np.bool_(np.array(falw) & 2**14)
            self.wrap_lo_m_fa = np.bool_(np.array(falw) & 2**15)
            self.wrap_hi_n_fa = np.bool_(np.array(falw) & 2**16)
            self.wrap_lo_n_fa = np.bool_(np.array(falw) & 2**17)
            self.wrap_m_and_n_fa = (self.wrap_lo_n_fa & self.wrap_lo_m_fa) | (self.wrap_hi_n_fa & self.wrap_hi_m_fa)
            self.ib_sel = np.array(self.ib)
            self.ib_noa_bare_flt = np.bool_(np.array(fltw) & 2**12)
            self.ib_amp_bare_flt = np.bool_(np.array(fltw) & 2**11)
            self.ib_dscn_flt = np.bool_(np.array(fltw) & 2**10)
            self.ib_dscn_fa = np.bool_(np.array(falw) & 2**10)
            self.ib_noa_flt = np.bool_(fltw & 2 ** 3)
            self.ib_noa_fa = np.bool_(falw & 2 ** 3)
            self.ib_amp_flt = np.bool_(fltw & 2 ** 2)
            self.ib_amp_fa = np.bool_(falw & 2 ** 2)
            self.vb_flt = np.bool_(np.array(fltw) & 2**1)
            self.vb_fa_lt = np.bool_(np.array(falw) & 2**1)
            self.Tb_flt = np.bool_(np.array(fltw) & 2**0)
            self.Tb_fa = np.bool_(np.array(falw) & 2**0)
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
            self.c_time_shunt = np.array(shunt.c_time_shunt[:i_end]) - self.time_run_start

        if ekf is None:
            pass
        else:
            # Load
            self.assign_all_from_frame(ekf, i_end)
            # Special handling
            self.time_e = np.array(np.atleast_1d(ekf.c_time_e)[:i_end] - self.time_run_start)

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
        if self.e_wrap_n_trim is None:
            self.e_wrap_n_trim = np.copy(self.ib) * 0.
        if self.e_wrap is None:
            self.e_wrap = np.copy(self.ib) * 0.
        if self.e_wrap_filt is None:
            self.e_wrap_filt = np.copy(self.ib) * 0.
        if self.mvb is None:
            self.mvb = np.bool(np.copy(self.mod_data))
        if self.Tb_model_f is None:
            print(f"Using Tb_f to initialize Tb_model_f")
            self.Tb_model_f = np.copy(self.Tb_f)
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
        if hasattr(self, 'y_ekf_f') and self.y_ekf_f is None:
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

        # Initialization time logic
        if init_time:
            self.init_time = init_time
        else:
            if self.time[0] == 0.:  # no initialization flat detected at beginning of recording
                self.init_time = 1.
            else:
                self.init_time = -4.

        if IbLag is not None:
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
        s += "{:10.6f},".format(self.y[self.i])
        s += "{:7.3f},".format(self.soc_s[self.i])
        s += "{:5.3f},".format(self.soc_ekf[self.i])
        s += "{:5.3f},".format(self.soc[self.i])
        return s

    def mod(self):
        return self.mod_data[self.zero_end]


# type: ignore
class SavedDataSim:
    def __init__(self, time_run_start, data=None, time_end=None, fake=False, mon_for_fake=None, str_=None):
        self.str = str_
        if data is None:
            pass
        else:
            self.cTime = np.array(data.c_time_sim)
            self.time = self.cTime  - time_run_start
            if time_end is None:
                i_end = len(self.time)
            else:
                i_end = np.where(self.time <= time_end)[0][-1] + 1
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
