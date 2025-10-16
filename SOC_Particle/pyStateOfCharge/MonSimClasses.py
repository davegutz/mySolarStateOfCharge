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
from Battery import Battery
from myFilters import LagExp
from pyDAGx import myTables

class SensorLooparound:
    """Collect Looparound sense parameters to create proper delays in data feed and connections to model"""

    def __init__(self, ib, ib_dyn, e_wrap_trim, e_wrap_filt):
        self.ib = ib
        self.ib_init = self.ib[0]
        self.ib_dyn = ib_dyn
        self.ib_dyn_init = self.ib_dyn[0]
        self.e_wrap_trim = e_wrap_trim
        self.e_wrap_trim_init = self.e_wrap_trim[0]
        self.e_wrap_filt = e_wrap_filt
        self.e_wrap_filt_init = self.e_wrap_filt[0]

    def update(self, i):
        self.ib_init = self.ib[max(i - 1, 0)]
        self.ib_dyn_init = self.ib_dyn[i]
        self.e_wrap_trim_init = self.e_wrap_trim[i]
        self.e_wrap_filt_init = self.e_wrap_filt[i]


class Sensors:
    """Collect various sense parameters to create proper delays in data feed and connections to model"""

    def __init__(self, mon_ref=None, sim_ref=None, add_Tb_in=None, run_type=None):

        if run_type == 'RunSim':
            self.Tb0 = mon_ref.Tb_f[0]
            self.Tb0_s = mon_ref.Tb_mod[0]
            self.lut_dTb = None
            self.dTb = 0.
            if add_Tb_in is not None:
                self.add_Tb_in = np.array(add_Tb_in)
                self.Tb0 += add_Tb_in[1, 0]
                self.lut_dTb = myTables.TableInterp1D(np.array(add_Tb_in[0, :]), np.array(add_Tb_in[1, :]))
                self.dTb = lut_dTb.interp(t[0])
            self.Tb = mon_ref.Tb[0]
            self.Tb_f = mon_ref.Tb_f[0]
            self.Tb_f_rate = mon_ref.Tb_f_rate[0]
            self.Tb_past = mon_ref.Tb_rap[0] + self.dTb
            self.Tb_f_past = mon_ref.Tb_f_rap[0] + self.dTb
            self.Tb_f_rate_past = mon_ref.Tb_f_rate_rap[0]
            self.TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)
            self.LoopAmp = SensorLooparound(mon_ref.ibmh, mon_ref.ib_dyn_m, mon_ref.e_wrap_m_trim,
                                            mon_ref.e_wrap_m_filt)
            self.LoopNoa = SensorLooparound(mon_ref.ibnh, mon_ref.ib_dyn_n, mon_ref.e_wrap_m_trim * 0.,
                                            mon_ref.e_wrap_n_filt)
            self.ib_amp = mon_ref.ibmh
            self.ib_noa = mon_ref.ibnh
            self.ib_dyn = mon_ref.ib_dyn
            self.ib_dyn_init = mon_ref.ib_dyn[0]
            self.z = mon_ref.z
            self.z_init = self.z[0]
            self.ib_in_s = sim_ref.ib_in_s
            self.ib_in_s_init = self.ib_in_s[0]
            self.ib_dyn_s = sim_ref.ib_dyn_s
            self.soc_s_init = mon_ref.soc_s[0]
            self.ib_dyn_s_init = self.ib_dyn_s[0]
            self.dv_dyn_s = sim_ref.dv_dyn_s
            self.dt_s = sim_ref.dt_s
            self.dv_dyn_s_init = self.dv_dyn_s[0]
            self.d_delta_q_s_init = 0.
            self.ib_s_init = self.ib_in_s_init
            self.ib_fut_s_init = self.ib_in_s_init
            self.ib_charge_s_init = self.ib_in_s_init
            self.ioc_s_init = self.ib_in_s_init
            self.vb_s_init = mon_ref.vb[0]
            self.voc_stat_init = mon_ref.voc_stat[0]
            self.voc_s_init = sim_ref.voc_stat_s[0]  # is this right?
            self.Tb_hdwe_init = mon_ref.Tb_hdwe[0]
            self.Tb_hdwe_filt_init = mon_ref.Tb_hdwe_filt[0]
            self.Tb_hdwe_filt_rate_init = mon_ref.Tb_hdwe_filt_rate[0]
            self.e_wrap_init = mon_ref.e_wrap[0]
            self.e_wrap_filt_init = mon_ref.e_wrap_filt[0]
            self.e_wrap_m_init = mon_ref.e_wrap_m[0]
            self.e_wrap_m_filt_init = mon_ref.e_wrap_m_filt[0]
            self.e_wrap_m_trim_init = mon_ref.e_wrap_m_trim[0]
            self.e_wrap_n_init = mon_ref.e_wrap_n[0]
            self.e_wrap_n_filt_init = mon_ref.e_wrap_n_filt[0]
            self.e_wrap_n_trim_init = 0.
            self.voc_soc_init = mon_ref.voc_soc[0]
            self.vb_s_init = mon_ref.vb[0]
            if run_type == 'RunSim':
                self.Tb_init = mon_ref.Tb[0]
            else:
                self.Tb_init = mon_ref.Tb_f[0]
            self.Tb_f_init = mon_ref.Tb_f[0]
            self.Tb_f_rate_init = mon_ref.Tb_f_rate[0]
            self.lut_dTb = None
            self.dTb = 0.
            if add_Tb_in is not None:
                self.add_Tb_in = np.array(add_Tb_in)
                self.Tb0 += add_Tb_in[1, 0]
                self.lut_dTb = myTables.TableInterp1D(np.array(add_Tb_in[0, :]), np.array(add_Tb_in[1, :]))
                self.dTb = lut_dTb.interp(t[0])
            self.Tb_f_rap = mon_ref.Tb_f_rap
            self.Tb_rap_init = mon_ref.Tb_rap[0] + self.dTb
            self.Tb_f_rap_init = mon_ref.Tb_f_rap[0] + self.dTb
            self.Tb_f_rate_rap_init = mon_ref.Tb_f_rate_rap[0]
            self.ib_init = mon_ref.ib[0]
            self.ib_dyn_init = mon_ref.ib_dyn[0]
            self.ib_charge_init = mon_ref.ib_charge[0]
            self.vb_init = mon_ref.vb[0]
            self.soc_init = mon_ref.soc[0]
            self.reset_init = True
            self.sat_init = mon_ref.sat[0]
            self.reset_ekf_init = True
            self.voc_ekf_init = mon_ref.hx[0]
            self.voc_stat_init = mon_ref.voc_stat[0]
            self.x_init = mon_ref.x[0]
            self.x_prior_init = mon_ref.x_prior[0]
            self.hx_init = mon_ref.hx[0]
            self.soc_ekf_init = mon_ref.soc_ekf[0]
            self.z_ekf_init = mon_ref.z[0]
            self.z_init = mon_ref.z[0]
        else:
            self.Tb_hdwe_init = mon_ref.Tb_h_f[0]
            self.Tb_hdwe_filt_init = mon_ref.Tb_h_f[0]
            self.Tb_hdwe_filt_rate_init = 0.
            self.e_wrap_init = mon_ref.e_wrap[0]
            self.e_wrap_filt_init = mon_ref.e_wrap[0]
            self.e_wrap_m_init = mon_ref.e_wrap[0]
            self.e_wrap_m_filt_init = mon_ref.e_wrap[0]
            self.e_wrap_m_trim_init = 0.
            self.e_wrap_n_init = mon_ref.e_wrap[0]
            self.e_wrap_n_filt_init = mon_ref.e_wrap[0]
            self.e_wrap_n_trim_init = 0.
            self.voc_soc_init = mon_ref.voc_soc[0]
            self.voc_stat_init = mon_ref.voc_stat_f[0]
            self.vb_s_init = mon_ref.vb_f[0]
            self.Tb0 = mon_ref.Tb_f[0]
            self.Tb_f_init = mon_ref.Tb_f[0]
            self.Tb0_s = mon_ref.Tb_f[0]
            self.Tb_f_rate_init = 0.
            self.lut_dTb = None
            self.dTb = 0.
            if add_Tb_in is not None:
                self.add_Tb_in = np.array(add_Tb_in)
                self.Tb0 += add_Tb_in[1, 0]
                self.lut_dTb = myTables.TableInterp1D(np.array(add_Tb_in[0, :]), np.array(add_Tb_in[1, :]))
                self.dTb = lut_dTb.interp(t[0])
            self.Tb_rap_init = mon_ref.Tb_f[0] + self.dTb
            self.Tb_f_rap_init = mon_ref.Tb_f[0] + self.dTb
            self.Tb_f_rate_rap_init = 0.
            self.Tb = mon_ref.Tb_f[0]
            self.Tb_f = np.copy(mon_ref.Tb_f)
            self.Tb_f_rate = np.copy(self.Tb_f) * 0.
            self.Tb_past = mon_ref.Tb_f[0] + self.dTb
            self.Tb_f_past = mon_ref.Tb_f[0] + self.dTb
            self.Tb_f_rate_past = np.copy(self.Tb_f) * 0.
            self.TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)

            if not hasattr(mon_ref, 'ib_dyn_m'):
                mon_ref.ib_dyn_m = np.copy(mon_ref.ibmh_f)

            if not hasattr(mon_ref, 'e_wrap_m_trim'):
                mon_ref.e_wrap_m_trim = np.copy(mon_ref.ibmh_f) * 0.

            if not hasattr(mon_ref, 'e_wrap_m_filt'):
                mon_ref.e_wrap_m_filt = np.copy(mon_ref.e_wrap)

            self.LoopAmp = SensorLooparound(mon_ref.ibmh_f, mon_ref.ib_dyn_m, mon_ref.e_wrap_m_trim,
                                            mon_ref.e_wrap_m_filt)

            if not hasattr(mon_ref, 'ib_dyn_n'):
               mon_ref.ib_dyn_n = np.copy(mon_ref.ibnh_f)

            if not hasattr(mon_ref, 'e_wrap_n_trim'):
                mon_ref.e_wrap_n_trim = np.copy(mon_ref.ibnh_f) * 0.

            if not hasattr(mon_ref, 'e_wrap_n_filt'):
                mon_ref.e_wrap_n_filt = np.copy(mon_ref.e_wrap)

            self.LoopNoa = SensorLooparound(mon_ref.ibnh_f, mon_ref.ib_dyn_n, mon_ref.e_wrap_m_trim * 0.,
                                            mon_ref.e_wrap_n_filt)
            self.ib_amp = mon_ref.ibmh_f
            self.ib_noa = mon_ref.ibnh_f
            self.ib_init = mon_ref.ib_f[0]
            self.ib_dyn = np.copy(mon_ref.ib_f)
            self.ib_dyn_init = mon_ref.ib_f[0]
            self.ib_charge_init = mon_ref.ib_charge_f[0]
            self.vb_init = mon_ref.vb_f[0]
            self.ibmm = mon_ref.ibmh_f
            self.ibnm = mon_ref.ibnh_f
            self.Tb_f_rap = mon_ref.Tb_f

            self.z = mon_ref.z
            self.z_init = self.z[0]

        self.sat_init = mon_ref.sat[0]
        # dt_s
        self.dt_s = []
        for i in range(len(sim_ref.time)):
            if i == 0:
                self.dt_s.append(sim_ref.time[1] - sim_ref.time[0])
            else:
                self.dt_s.append(sim_ref.time[i] - sim_ref.time[i-1])
        self.soc_s = mon_ref.soc_s
        self.dq_s = -mon_ref.qcrs * (1. - mon_ref.soc_s)
        self.delta_q = -mon_ref.qcrs * (1. - mon_ref.soc)
        self.ib_in_s = sim_ref.ib_in_s
        self.ib_in_s_init = self.ib_in_s[0]
        self.ib_dyn_s = np.copy(self.ib_in_s)
        self.ib_dyn_s_init = self.ib_dyn_s[0]
        self.dv_dyn_s = sim_ref.dv_dyn_s
        self.dv_dyn_s_init = self.dv_dyn_s[0]
        self.d_delta_q_s_init = 0.
        self.ib_s_init = self.ib_in_s_init
        self.ib_fut_s_init = self.ib_in_s_init
        self.ib_charge_s_init = self.ib_in_s_init
        self.ioc_s_init = self.ib_in_s_init
        self.voc_s_init = sim_ref.voc_stat_s[0]  # is this right?
        self.ib_dyn_s_init = self.ib_in_s_init
        self.soc_s_init = mon_ref.soc_s[0]
        self.hx_init = self.voc_soc_init
        self.soc_init = mon_ref.soc[0]
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

    def calc_dTb(self, i):
        if self.dTb is not 0.:
            dTb = SN.lut_dTb.interp(t[i])
        else:
            dTb = self.dTb
        return dTb

    def temp_calc(self, mon_ref, mon, Battery, i_temp):
        if hasattr(mon_ref, 'Tb_hdwe_filt'):
            mon.Tb_hdwe_filt = \
                self.TbSenseFilt.calculate_tau_seeded(mon.Tb_hdwe, mon_ref.Tb_hdwe_filt[i_temp],
                                                      mon.reset_temp,
                                                      mon.dt_temp, Battery.TB_FILT, rmax=Battery.T_RLIM,
                                                      rmin=-Battery.T_RLIM)
        else:
            mon.Tb_hdwe_filt = \
                self.TbSenseFilt.calculate_tau_seeded(mon.Tb_hdwe, mon.Tb_hdwe,
                                                      mon.reset_temp,
                                                      mon.dt_temp, Battery.TB_FILT, rmax=Battery.T_RLIM,
                                                      rmin=-Battery.T_RLIM)

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

    def update_ib_vb(self, i):
        self.LoopAmp.update(i)
        self.LoopNoa.update(i)
        self.ib_dyn_init = self.ib_dyn[i]
        self.ib_in_s_init = self.ib_in_s[i]
        self.ib_dyn_s_init = self.ib_dyn_s[i]
        self.dv_dyn_s_init = self.dv_dyn_s[i]

    def update_tb(self):
        self.Tb_past = self.Tb
        self.Tb_f_past = self.Tb_f
        self.Tb_f_rate_past = self.Tb_f_rate

