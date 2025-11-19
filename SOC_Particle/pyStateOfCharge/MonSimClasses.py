# MonSimClasses:  Subclasses used to support replicate()
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

import numpy as np
import Battery
from Battery import Battery, calculate_capacity
from myFilters import LagExp
from pyDAGx import myTables

class MutableInt:
    def __init__(self, value):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self

    def __repr__(self):
        return str(self.value)


class ProArray:
    def __init__(self, data, mutable=False):
        # Initialize a NumPy array, marked as "internal" with a leading underscore
        if not isinstance(data, (list, np.ndarray)):
            raise TypeError("Data must be a list or NumPy array.")
        if not mutable:
            self._data = np.array(data)
        else:
            self._data = data

    @property
    def data(self):
        """
        Getter for the array data. Returns a copy to prevent external modification
        of the internal array.
        """
        return self._data.copy()

    def __getitem__(self, index=None):
        return self._data[max(min(index, len(self._data)-1), 0)]

    def __setitem__(self, index, value):
        self._data[index] = value
        return None

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return f"ProArray({self._data})"


class SensorLooparound:
    """Collect Looparound sense parameters to create proper delays in data feed and connections to model"""

    def __init__(self, ib, ib_dyn, e_wrap_trim, e_wrap_filt):
        self.ib = ib
        self.ib_init = self.ib[0]
        self.ib_dyn = ib_dyn
        # self.ib_dyn_init = self.ib_dyn[0]
        self.e_wrap_trim = e_wrap_trim
        self.e_wrap_trim_init = self.e_wrap_trim[0]
        self.e_wrap_filt = e_wrap_filt
        self.e_wrap_filt_init = self.e_wrap_filt[0]

    def update(self, i):
        self.ib_init = self.ib[max(i - 1, 0)]
        # self.ib_dyn_init = self.ib_dyn[i]
        self.e_wrap_trim_init = self.e_wrap_trim[i]
        self.e_wrap_filt_init = self.e_wrap_filt[i]


class Sensors:
    """Collect various sense parameters to create proper delays in data feed and connections to model"""

    def __init__(self, OPT, run_type=None):
        self.mon_run = OPT.mon_run
        self.sim_run = OPT.sim_run
        if run_type == 'RunSim':
            if hasattr(self.mon_run, 'mtb'):
                self.mod_tb = self.mon_run.mtb
            else:
                self.mod_tb = self.mon_run.Tb_f.copy()*0.
            self.Tb0 = self.mon_run.Tb_f[0]
            self.Tb0_s = self.mon_run.Tb_mod[0]
            self.lut_dTb = None
            self.dTb = 0.
            if OPT.add_Tb_in is not None:
                self.add_Tb_in = np.array(OPT.add_Tb_in)
                self.Tb0 += OPT.add_Tb_in[1, 0]
                self.lut_dTb = myTables.TableInterp1D(np.array(OPT.add_Tb_in[0, :]), np.array(OPT.add_Tb_in[1, :]))
                self.dTb = self.lut_dTb.interp(self.mon_run.t[0])
            self.Tb = self.mon_run.Tb[0]
            self.Tb_f = self.mon_run.Tb_f[0]
            self.Tb_f_rate = self.mon_run.Tb_f_rate[0]
            self.Tb_past = self.mon_run.Tb_rap[0] + self.dTb
            self.Tb_f_past = self.mon_run.Tb_f_rap[0] + self.dTb
            self.Tb_f_rate_past = self.mon_run.Tb_f_rate_rap[0]
            self.TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)
            self.LoopAmp = SensorLooparound(self.mon_run.ibmh, self.mon_run.ib_dyn_m, self.mon_run.e_wrap_m_trim,
                                            self.mon_run.e_wrap_m_filt)
            self.LoopNoa = SensorLooparound(self.mon_run.ibnh, self.mon_run.ib_dyn_n, self.mon_run.e_wrap_m_trim * 0.,
                                            self.mon_run.e_wrap_n_filt)
            self.ib_amp = 0.
            self.ib_noa = 0.
            self.ib_dyn = ProArray(self.mon_run.ib_dyn, mutable=True)
            # self.ib_dyn_init = self.ib_dyn[0]
            self.z = self.mon_run.z
            # self.z_init = self.z[0]
            self.ib_in_s = self.sim_run.ib_in_s
            self.ib_in_s_init = self.ib_in_s[0]
            self.ib_dyn_s = self.sim_run.ib_dyn_s
            # self.soc_s_init = self.mon_run.soc_s[0]
            # self.ib_dyn_s_init = self.ib_dyn_s[0]
            self.dv_dyn_s = self.sim_run.dv_dyn_s
            self.dt_s = self.sim_run.dt_s
            self.dv_dyn_s_init = self.dv_dyn_s[0]
            self.d_delta_q_s_init = 0.
            # self.ib_s_init = self.ib_in_s_init
            # self.ib_fut_s_init = self.ib_in_s_init
            # self.ib_charge_s_init = self.ib_in_s_init
            # self.ioc_s_init = self.ib_in_s_init
            # self.vb_s_init = self.mon_run.vb[0]
            # self.voc_stat_init = self.mon_run.voc_stat[0]
            # self.voc_s_init = self.sim_run.voc_stat_s[0]  # is this right?
            self.Tb_hdwe_init = self.mon_run.Tb_hdwe[0]
            self.Tb_hdwe_filt_init = self.mon_run.Tb_hdwe_filt[0]
            self.Tb_hdwe_filt_rate_init = self.mon_run.Tb_hdwe_filt_rate[0]
            self.e_wrap_init = self.mon_run.e_wrap[0]
            self.e_wrap_filt_init = self.mon_run.e_wrap_filt[0]
            self.e_wrap_m_init = self.mon_run.e_wrap_m[0]
            self.e_wrap_m_filt_init = self.mon_run.e_wrap_m_filt[0]
            self.e_wrap_m_trim_init = self.mon_run.e_wrap_m_trim[0]
            self.e_wrap_n_init = self.mon_run.e_wrap_n[0]
            self.e_wrap_n_filt_init = self.mon_run.e_wrap_n_filt[0]
            self.e_wrap_n_trim_init = 0.
            self.voc_soc_init = self.mon_run.voc_soc[0]
            self.vb_s_init = self.mon_run.vb[0]
            # self.Tb_init = self.mon_run.Tb[0]
            self.Tb_f_init = self.mon_run.Tb_f[0]
            self.Tb_f_rate_init = self.mon_run.Tb_f_rate[0]
            self.lut_dTb = None
            self.dTb = 0.
            if OPT.add_Tb_in is not None:
                self.add_Tb_in = np.array(OPT.add_Tb_in)
                self.Tb0 += OPT.add_Tb_in[1, 0]
                self.lut_dTb = myTables.TableInterp1D(np.array(OPT.add_Tb_in[0, :]), np.array(OPT.add_Tb_in[1, :]))
                self.dTb = self.lut_dTb.interp(self.mon_run.t[0])
            self.Tb_f_rap = self.mon_run.Tb_f_rap
            self.Tb_rap_init = self.mon_run.Tb_rap[0] + self.dTb
            self.Tb_f_rap_init = self.mon_run.Tb_f_rap[0] + self.dTb
            self.Tb_f_rate_rap_init = self.mon_run.Tb_f_rate_rap[0]
            self.ib_init = self.mon_run.ib[0]
            self.ib_charge_init = self.mon_run.ib_charge[0]
            self.vb_init = self.mon_run.vb[0]
            # self.soc_init = self.mon_run.soc[0]
            # self.reset_init = True
            # self.sat_init = self.mon_run.sat[0]
            # self.reset_ekf_init = True
            # self.voc_ekf_init = self.mon_run.hx[0]
            self.voc_stat_init = self.mon_run.voc_stat[0]
            # self.x_init = self.mon_run.x[0]
            # self.x_prior_init = self.mon_run.x_prior[0]
            # self.hx_init = self.mon_run.hx[0]
            # self.soc_ekf_init = self.mon_run.soc_ekf[0]
            # self.z_ekf_init = self.mon_run.z[0]
            # self.z_init = self.mon_run.z[0]

        elif run_type == 'HistSim':

            if not hasattr(self.mon_run, 'e_wrap_f'):
                self.mon_run.e_wrap_f = np.copy(self.mon_run.e_w_f)
            if not hasattr(self.mon_run, 'e_wrap_m_filt'):
                self.mon_run.e_wrap_m_filt = np.copy(self.mon_run.e_wm_f)
            if not hasattr(self.mon_run, 'e_wrap_m_trim'):
                self.mon_run.e_wrap_m_trim = np.copy(self.mon_run.e_wm_t)
            if not hasattr(self.mon_run, 'e_wrap_n_filt'):
                self.mon_run.e_wrap_n_filt = np.copy(self.mon_run.e_wn_f)
            if not hasattr(self.mon_run, 'e_wrap_n_trim'):
                self.mon_run.e_wrap_n_trim = np.copy(self.mon_run.e_wm_t) * 0.
            if not hasattr(self.mon_run, 'ib_dyn_m'):
                self.mon_run.ib_dyn_m = np.copy(self.mon_run.ibmh_f)
            if not hasattr(self.mon_run, 'ib_dyn_n'):
               self.mon_run.ib_dyn_n = np.copy(self.mon_run.ibnh_f)

            self.Tb_hdwe_init = self.mon_run.Tb_h_f[0]
            self.Tb_hdwe_filt_init = self.mon_run.Tb_h_f[0]
            self.Tb_hdwe_filt_rate_init = 0.
            self.e_wrap_init = self.mon_run.e_wrap[0]
            self.e_wrap_filt_init = self.mon_run.e_wrap_f[0]
            self.e_wrap_m_init = self.mon_run.e_wrap[0]
            self.e_wrap_m_filt_init = self.mon_run.e_wrap_m_filt[0]
            self.e_wrap_m_trim_init = 0.
            self.e_wrap_n_init = self.mon_run.e_wrap[0]
            self.e_wrap_n_filt_init = self.mon_run.e_wrap_n_filt[0]
            self.e_wrap_n_trim_init = 0.
            self.voc_soc_init = self.mon_run.voc_soc[0]
            self.voc_stat_init = self.mon_run.voc_stat_f[0]
            self.vb_s_init = self.mon_run.vb_f[0]
            self.Tb0 = self.mon_run.Tb_f[0]
            self.Tb_f_init = self.mon_run.Tb_f[0]
            self.Tb0_s = self.mon_run.Tb_f[0]
            self.Tb_f_rate_init = 0.
            self.lut_dTb = None
            self.dTb = 0.
            if OPT.add_Tb_in is not None:
                self.add_Tb_in = np.array(OPT.add_Tb_in)
                self.Tb0 += OPT.add_Tb_in[1, 0]
                self.lut_dTb = myTables.TableInterp1D(np.array(OPT.add_Tb_in[0, :]), np.array(OPT.add_Tb_in[1, :]))
                self.dTb = self.lut_dTb.interp(self.mon_run.t[0])
            self.Tb_rap_init = self.mon_run.Tb_f[0] + self.dTb
            self.Tb_f_rap_init = self.mon_run.Tb_f[0] + self.dTb
            self.Tb_f_rate_rap_init = 0.
            self.Tb = self.mon_run.Tb_f[0]
            self.Tb_f = np.copy(self.mon_run.Tb_f)
            self.Tb_f_rate = np.copy(self.Tb_f) * 0.
            self.Tb_past = self.mon_run.Tb_f[0] + self.dTb
            self.Tb_f_past = self.mon_run.Tb_f[0] + self.dTb
            self.Tb_f_rate_past = np.copy(self.Tb_f) * 0.
            self.TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)

            self.LoopAmp = SensorLooparound(self.mon_run.ibmh_f, self.mon_run.ib_dyn_m, self.mon_run.e_wrap_m_trim,
                                            self.mon_run.e_wrap_m_filt)

            self.LoopNoa = SensorLooparound(self.mon_run.ibnh_f, self.mon_run.ib_dyn_n, self.mon_run.e_wrap_m_trim * 0.,
                                            self.mon_run.e_wrap_n_filt)
            self.ib_amp = self.mon_run.ibmh_f
            self.ib_noa = self.mon_run.ibnh_f
            self.ib_init = self.mon_run.ib_f[0]
            self.ib_dyn = ProArray(self.mon_run.ib_dyn)
            self.ib_charge_init = self.mon_run.ib_charge_f[0]
            self.vb_init = self.mon_run.vb_f[0]
            self.ibmm = self.mon_run.ibmh_f
            self.ibnm = self.mon_run.ibnh_f
            self.Tb_f_rap = self.mon_run.Tb_f

            self.z = self.mon_run.z
            self.z_init = self.z[0]

        self.i = 0
        self.sat_init = self.mon_run.sat[0]
        self.soc_s = self.mon_run.soc_s

        # q
        if not hasattr(self.mon_run, 'q_capacity'):
            self.q_cap = calculate_capacity(q_cap_rated_scaled=self.mon_run.qcrs, dqdt=self.mon_run.dqdt, tb_f=self.Tb_f,
                                            t_rated=self.mon_run.t_rated)
        else:
            self.q_cap = self.mon_run.q_capacity
        if not hasattr(self.mon_run, 'delta_q'):
            self.delta_q = -self.q_cap * (1. - self.mon_run.soc)
        else:
            self.delta_q = self.mon_run.delta_q
        if not hasattr(self.sim_run, 'qcap_s'):
            self.q_cap_s = calculate_capacity(q_cap_rated_scaled=self.mon_run.qcrs_s, dqdt=self.mon_run.dqdt, tb_f=self.Tb_f,
                                              t_rated=self.mon_run.t_rated)
        else:
            self.q_cap_s = self.sim_run.qcap_s
        if not hasattr(self.sim_run, 'dq_s'):
            self.delta_q_s = -self.q_cap_s * (1. - self.mon_run.soc_s)
        else:
            self.delta_q_s = self.sim_run.dq_s
        self.d_delta_q_s_init = 0.
        self.delta_q_s_init = self.delta_q_s[0]

        self.ib_in_s = self.sim_run.ib_in_s
        self.ib_in_s_init = self.ib_in_s[0]
        if not hasattr(self, 'ib_dyn_s'):
            self.ib_dyn_s = np.copy(self.ib_in_s)
        self.ib_dyn_s_init = self.ib_dyn_s[0]
        self.dv_dyn_s = self.sim_run.dv_dyn_s
        self.dv_dyn_s_init = self.dv_dyn_s[0]
        self.ib_s_init = self.ib_in_s_init
        self.ib_fut_s_init = self.ib_in_s_init
        self.ib_charge_s_init = self.ib_in_s_init
        self.ioc_s_init = self.ib_in_s_init
        self.voc_s_init = self.sim_run.voc_stat_s[0]
        self.soc_s_init = self.mon_run.soc_s[0]
        self.hx_init = self.voc_soc_init
        self.soc_init = self.mon_run.soc[0]
        self.x_init = self.soc_init
        self.x_prior_init = self.x_init
        self.soc_ekf_init = self.soc_init
        self.z_ekf_init = self.hx_init
        self.z_init = self.hx_init
        self.skip_e = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_t = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_sel = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_rap = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_s = np.bool(np.zeros(len(self.dv_dyn_s)))

    def __str__(self, prefix=''):
        s = prefix + "TFDelay:\n"
        s += "  Tb0 =  {:9.7f}  // deg C\n".format(self.Tb0)
        s += "  Tb0_s =  {:9.7f}  // deg C\n".format(self.Tb0_s)
        return s

    def assign_tb(self, mon_Tb, mon_Tb_f, mon_Tb_f_rate):
        self.Tb = mon_Tb + self.dTb
        self.Tb_f = mon_Tb_f + self.dTb
        self.Tb_f_rate = mon_Tb_f_rate

    def calc_dTb(self, i, SN, t):
        if self.dTb != 0.:
            dTb = SN.lut_dTb.interp(t[i])
        else:
            dTb = self.dTb
        return dTb

    def calc_temp_pass_1(self, OPT, mon_, sim_, i_temp):
        mon = mon_
        sim = sim_
        if hasattr(OPT.mon_run, 'Tb_hdwe'):
            mon.Tb_hdwe = OPT.mon_run.Tb_hdwe[i_temp]
        else:
            mon.Tb_hdwe = OPT.mon_run.Tb_f[i_temp]
        mon.reset_temp = (i_temp < 2) or mon.reset or OPT.run_type == 'HistSim'  # make sure temp init is longer than reset
        if hasattr(OPT.mon_run, 'Tt'):
            mon.dt_temp = OPT.mon_run.Tt[i_temp]
        else:
            mon.dt_temp = mon.dt
        if OPT.run_type == 'RunSim':
            if bool(self.mod_tb[i_temp]):
                mon.Tb = OPT.mon_run.Tb[i_temp]
            else:
                mon.Tb = mon.Tb_hdwe  # past value
            sim.Tb = mon.Tb
            mon.Tb_s =mon.Tb
        else:
            sim.Tb = OPT.mon_run.Tb_f[i_temp]
            mon.Tb = OPT.mon_run.Tb_f[i_temp]
            mon.Tb_s = OPT.mon_run.Tb_f[i_temp]
        if i_temp > 0:
            self.update_tb()
            mon.Tb_rap = self.Tb_past
            mon.Tb_f_rap = self.Tb_f_past
            mon.Tb_f_rate_rap = self.Tb_f_rate_past
        if hasattr(OPT.mon_run, 'Tb_mod'):
            sim.Tb_f = OPT.mon_run.Tb_mod[i_temp]
        else:
            sim.Tb_f = sim.Tb
        return mon, sim

    def calc_temp_pass_2(self, mon_run, mon, Battery_, i_temp):
        if hasattr(mon_run, 'Tb_hdwe_filt'):
            if self.mod_tb[i_temp]:
                mon.Tb_hdwe_filt = mon.Tb
            else:
                mon.Tb_hdwe_filt = \
                    self.TbSenseFilt.calculate_tau_seeded(mon.Tb_hdwe, mon_run.Tb_hdwe_filt[i_temp], mon.reset_temp,
                                                          mon.dt_temp, Battery_.TB_FILT, rmax=Battery_.T_RLIM,
                                                          rmin=-Battery_.T_RLIM)
        else:
            mon.Tb_hdwe_filt = \
                self.TbSenseFilt.calculate_tau_seeded(mon.Tb_hdwe, mon.Tb_hdwe,
                                                      mon.reset_temp,
                                                      mon.dt_temp, Battery_.TB_FILT, rmax=Battery_.T_RLIM,
                                                      rmin=-Battery_.T_RLIM)

        mon.Tb_hdwe_filt_rate = self.TbSenseFilt.rate
        mon.Tb_f_rate = mon.Tb_hdwe_filt_rate
        if not mon.reset_temp:
            mon.Tb_rap = self.Tb_past
        mon.Tb_f = mon.Tb_hdwe_filt
        self.Tb_f = mon.Tb_hdwe_filt
        self.assign_tb(mon.Tb, mon.Tb_f, mon.Tb_f_rate)
        mon.Tb_rstate = self.TbSenseFilt.rstate
        mon.Tb_state = self.TbSenseFilt.state
        return mon

    def update_ekf(self, i_ekf):
        self.z_init = self.z[i_ekf]

    # def ib_dyn(self, ind=None):
    #     if ind is None:
    #         return self.ib_dyn.el(self.i)
    #     else:
    #         return self.ib_dyn.el(max(ind, 0))
    #
    def update(self, i):
        self.i = min(max(i, 0), len(self.mon_run.time)-1)

    def update_ib_vb(self, i):
        self.i = min(max(i, 0), len(self.mon_run.time)-1)
        self.LoopAmp.update(i)
        self.LoopNoa.update(i)
        # self.ib_dyn_init = self.ib_dyn[i]
        # self.ib_dyn_init = self.ib_dyn[i]
        self.ib_in_s_init = self.ib_in_s[i]
        self.ib_dyn_s_init = self.ib_dyn_s[i]
        self.dv_dyn_s_init = self.dv_dyn_s[i]
        self.e_wrap_m_filt_init = self.mon_run.e_wrap_m_filt[i]
        self.e_wrap_m_trim_init = self.mon_run.e_wrap_m_trim[i]
        self.e_wrap_n_filt_init = self.mon_run.e_wrap_n_filt[i]

    def update_tb(self):
        self.Tb_past = self.Tb
        self.Tb_f_past = self.Tb_f
        self.Tb_f_rate_past = self.Tb_f_rate
