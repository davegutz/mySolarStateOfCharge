# MonSimClasses:  Subclasses used to support replicate()
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

from Battery import calculate_capacity
from Battery import Battery as Battery
from filter.KF1x1 import KF1x1VarDtxx
import numpy.lib.recfunctions as rfn
from filter.myFilters import LagExp
from pyDAGx import myTables
import Globals as G
import numpy as np


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
        self.e_wrap_trim = e_wrap_trim
        self.e_wrap_filt = e_wrap_filt

    def update(self, i):
        self.ib_init = self.ib[max(i - 1, 0)]


class Sensors:
    """Collect various sense parameters to create proper delays in data feed and connections to model"""

    def __init__(self, OPT, run_type=None):
        self.run_type = run_type
        if self.run_type == 'HistSim':
            self.mon_run = OPT.mon_run.copy()
            self.sim_run = OPT.sim_run.copy()
        else:
            self.mon_run = OPT.mon_run
            self.sim_run = OPT.sim_run
        if self.run_type == 'RunSim':
            if hasattr(self.mon_run, 'mtb') and self.mon_run.mtb is not None:
                self.mod_tb = self.mon_run.mtb
            else:
                self.mod_tb = np.copy(self.mon_run.mod_data)
            self.Tb0 = self.mon_run.Tb_f_rap[0]
            self.Tb0_s = self.mon_run.Tb_rap[0]
            self.Tb = self.mon_run.Tb_rap[0]
            self.Tb_f = self.mon_run.Tb_f_rap[0]
            self.lut_dTb = None
            self.dTb = 0.
            if OPT.add_Tb_in is not None:
                self.add_Tb_in = np.array(OPT.add_Tb_in)
                self.Tb0 += OPT.add_Tb_in[1, 0]
                self.lut_dTb = myTables.TableInterp1D(np.array(OPT.add_Tb_in[0, :]), np.array(OPT.add_Tb_in[1, :]))
                self.dTb = self.lut_dTb.interp(self.mon_run.t[0])
            if self.mon_run.Tb_f_rate is not None:
                self.Tb_f_rate = self.mon_run.Tb_f_rate[0]
            else:
                self.Tb_f_rate = self.mon_run.Tb_f_rate_rap[0]
            self.Tb_past = self.mon_run.Tb_rap[0] + self.dTb
            self.Tb_f_past = self.mon_run.Tb_f_rap[0] + self.dTb
            self.Tb_f_rate_past = self.mon_run.Tb_f_rate_rap[0]
            self.TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)
            self.TbModelFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)
            self.LoopAmp = SensorLooparound(self.mon_run.ib_amp_hdwe, self.mon_run.ib_dyn_m, self.mon_run.e_wrap_m_trim,
                                            self.mon_run.e_wrap_m_filt)
            self.LoopNoa = SensorLooparound(self.mon_run.ib_noa_hdwe, self.mon_run.ib_dyn_n, self.mon_run.e_wrap_m_trim * 0.,
                                            self.mon_run.e_wrap_n_filt)
            self.Battery = Battery
            if hasattr(self.mon_run, 'vovcm'):
                self.KfShuntAmp = KF1x1VarDtxx(initial_position=self.mon_run.vovcm[0], initial_velocity=self.mon_run.x1m[0],
                                             dt=0.1, proc_noise_std=Battery.KF_Q_STD, meas_noise_std=Battery.KF_R_STD)
            if hasattr(self.mon_run, 'vovcn'):
                print(f"input:   KF_Q_STD {self.Battery.KF_Q_STD}  KF_R_STD {self.Battery.KF_R_STD}")
                self.Battery.KF_Q_STD /= 1.
                self.Battery.KF_R_STD /= 1.
                print(f"using:   KF_Q_STD {self.Battery.KF_Q_STD}  KF_R_STD {self.Battery.KF_R_STD}")
                self.KfShuntNoa = KF1x1VarDtxx(initial_position=self.mon_run.vovcn[0], initial_velocity=self.mon_run.x1n[0],
                                             dt=0.1, proc_noise_std=self.Battery.KF_Q_STD, meas_noise_std=self.Battery.KF_R_STD)

            self.ib_amp = 0.
            self.ib_noa = 0.
            self.ib_dyn = ProArray(self.mon_run.ib_dyn, mutable=True)
            self.z = self.mon_run.z
            self.ib_in_s = self.sim_run.ib_in_s
            self.ib_dyn_s = self.sim_run.ib_dyn_s
            self.dv_dyn_s = self.sim_run.dv_dyn_s
            self.dt_s = self.sim_run.dt_s
            self.d_delta_q_s_init = 0.
            self.Tb_hdwe_init = self.mon_run.Tb_hdwe[0]
            self.Tb_model_init = self.mon_run.Tb_model[0]
            self.Tb_hdwe_filt_init = self.mon_run.Tb_hdwe_filt[0]
            self.Tb_model_filt_init = self.mon_run.Tb_model_filt[0]
            self.Tb_model_filt_fut = self.mon_run.Tb_model_filt[0]
            self.Tb_model_filt_rate_fut = self.mon_run.Tb_model_filt_rate[0]
            self.Tb_hdwe_filt_rate_init = self.mon_run.Tb_hdwe_filt_rate[0]
            self.Tb_model_filt_rate_init = self.mon_run.Tb_model_filt_rate[0]
            self.e_wrap_init = self.mon_run.e_wrap[0]
            self.e_wrap_filt_init = self.mon_run.e_wrap_filt[0]
            self.e_wrap_m_init = self.mon_run.e_wrap_m[0]
            self.e_wrap_n_init = self.mon_run.e_wrap_n[0]
            self.vb_s_init = self.mon_run.vb[0]
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
            self.voc_stat_init = self.mon_run.voc_stat[0]
            self.dt_temp_fut = self.mon_run.Tt[1]
            self.dt_temp = self.mon_run.Tt[0]
            self.ib_amp_model = self.mon_run.ib_amp_model

        elif self.run_type == 'HistSim':
            if not hasattr(self.mon_run, 'ib_dyn_m'):
                self.mon_run.ib_dyn_m = np.copy(self.mon_run.ib_amp_hdwe_f)
            if not hasattr(self.mon_run, 'ib_dyn_n'):
               self.mon_run.ib_dyn_n = np.copy(self.mon_run.ib_noa_hdwe_f)

            self.dt_s = self.sim_run.dt_s
            if not hasattr(self.mon_run, 'ibmm'):
               self.mon_run.ibmm = np.copy(self.mon_run.ib_amp_hdwe_f)
            if not hasattr(self.mon_run, 'ib_noa_model'):
               self.mon_run.ib_noa_model = np.copy(self.mon_run.ib_noa_hdwe_f)
            if not hasattr(self.mon_run, 'ib_h'):
               self.mon_run.ib_h = np.copy(self.mon_run.ib_f)
            self.Battery = Battery

            self.Tb_hdwe_init = self.mon_run.Tb_h_f[0]
            self.Tb_hdwe_filt_init = self.mon_run.Tb_h_f[0]
            self.Tb_hdwe_filt_rate_init = 0.
            self.Tb_model_filt_rate_init = 0.
            self.Tb_model_init = self.mon_run.Tb_h_f[0]
            self.Tb_model_filt_init = self.mon_run.Tb_h_f[0]
            if hasattr(self.mon_run, 'e_wrap'):
                self.e_wrap_init = self.mon_run.e_wrap[0]
                self.e_wrap_m_init = self.mon_run.e_wrap_m[0]
                self.e_wrap_n_init = self.mon_run.e_wrap_n[0]
            else:
                self.e_wrap_init = self.mon_run.e_wrap_filt[0]
                self.e_wrap_m_init = self.mon_run.e_wrap_m_filt[0]
                self.e_wrap_n_init = self.mon_run.e_wrap_n_filt[0]
            self.e_wrap_filt_init = self.mon_run.e_wrap_filt[0]
            self.voc_stat_init = self.mon_run.voc_stat_f[0]
            self.vb_s_init = self.mon_run.vb_f[0]
            self.Tb0 = self.mon_run.Tb_f[0]
            self.Tb_f_rap = self.mon_run.Tb_f
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
            self.TbModelFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)
            self.TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)

            self.LoopAmp = SensorLooparound(self.mon_run.ib_amp_hdwe_f, self.mon_run.ib_dyn_m, self.mon_run.e_wrap_m_trim,
                                            self.mon_run.e_wrap_m_filt)

            self.LoopNoa = SensorLooparound(self.mon_run.ib_noa_hdwe_f, self.mon_run.ib_dyn_n, self.mon_run.e_wrap_m_trim * 0.,
                                            self.mon_run.e_wrap_n_filt)
            self.ib_amp = self.mon_run.ib_amp_hdwe_f
            self.ib_noa = self.mon_run.ib_noa_hdwe_f
            self.ib_init = self.mon_run.ib_f[0]
            self.ib_dyn = ProArray(self.mon_run.ib_dyn)
            self.ib_charge_init = self.mon_run.ib_charge_f[0]
            self.vb_init = self.mon_run.vb_f[0]
            self.ib_amp_model = self.mon_run.ib_amp_hdwe_f
            self.ib_noa_model = self.mon_run.ib_noa_hdwe_f

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
        if not hasattr(self.sim_run, 'delta_q_s'):
            self.delta_q_s = -self.q_cap_s * (1. - self.mon_run.soc_s)
        else:
            self.delta_q_s = self.sim_run.delta_q_s
        self.d_delta_q_s_init = 0.
        self.delta_q_s_init = self.delta_q_s[0]

        self.ib_in_s = self.sim_run.ib_in_s
        # self.ib_in_s_init = self.ib_in_s[0]
        if not hasattr(self, 'ib_dyn_s'):
            self.ib_dyn_s = np.copy(self.ib_in_s)
        self.dv_dyn_s = self.sim_run.dv_dyn_s
        self.ib_s_init = self.ib_in_s[0]
        self.ib_fut_s_init = self.ib_in_s[0]
        self.ib_charge_s_init = self.ib_in_s[0]
        self.ioc_s_init = self.ib_in_s[0]
        self.voc_s_init = self.sim_run.voc_stat_s[0]
        self.soc_s_init = self.mon_run.soc_s[0]
        self.hx_init = self.mon_run.voc_soc[0]
        self.soc_init = self.mon_run.soc[0]
        self.x_init = self.soc_init
        self.x_prior_init = self.x_init
        self.soc_ekf_init = self.soc_init
        self.z_init = self.hx_init
        self.skip_e = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_t = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_sel = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_rap = np.bool(np.zeros(len(self.dv_dyn_s)))
        self.skip_s = np.bool(np.zeros(len(self.dv_dyn_s)))

        self.VoVcm = 0.
        self.VoVcm_f = 0.

        self.VoVcn = 0.
        self.VoVcn_f = 0.
        self.iscn = 0.
        self.iscn_f = 0.

    def __str__(self, prefix=''):
        s = prefix + "TFDelay:\n"
        s += "  Tb0 =  {:9.7f}  // deg C\n".format(self.Tb0)
        s += "  Tb0_s =  {:9.7f}  // deg C\n".format(self.Tb0_s)
        return s

    def assign_tb(self, mon_Tb, mon_Tb_f, mon_Tb_f_rate):
        self.Tb = mon_Tb + self.dTb
        self.Tb_f = mon_Tb_f + self.dTb
        self.Tb_f_rate = mon_Tb_f_rate

    def calc_temp_pass_1(self, OPT, mon_, sim_, i_temp, rp):
        mon = mon_
        sim = sim_
        if hasattr(OPT.mon_run, 'Tb_hdwe'):
            mon.Tb_hdwe = OPT.mon_run.Tb_hdwe[i_temp]
        else:
            mon.Tb_hdwe = OPT.mon_run.Tb_f[i_temp]
        if hasattr(OPT.mon_run, 'Tb_model'):
            mon.Tb_model = OPT.mon_run.Tb_model[i_temp]
        else:
            mon.Tb_model = OPT.mon_run.Tb_f[i_temp]
        mon.reset_temp = (i_temp < 2) or mon.reset or OPT.run_type == 'HistSim'  # make sure temp init is longer than reset
        if hasattr(OPT.mon_run, 'Tt'):
            mon.dt_temp = OPT.mon_run.Tt[i_temp]
            index = min(i_temp+1, len(OPT.mon_run.Tt)-1)
            self.dt_temp = self.dt_temp_fut
            self.dt_temp_fut = OPT.mon_run.Tt[index]
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
        sim.Tb_f = self.Tb_f_past  # same  modeling and sensed

        return mon, sim

    def calc_temp_pass_2(self, mon_run, mon, Battery_, i_temp, rp):
        if hasattr(mon_run, 'Tb_hdwe_filt'):
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
        if hasattr(mon_run, 'Tb_model_filt'):
            mon.Tb_model_filt = self.Tb_model_filt_fut
            mon.Tb_model_filt_rate = self.Tb_model_filt_rate_fut
            index_temp = min(i_temp+1, len(mon_run.Tb_model_filt)-1)
            self.Tb_model_filt_fut = \
                self.TbModelFilt.calculate_tau_seeded(mon.Tb_model, mon_run.Tb_model_filt[index_temp], mon.reset_temp,
                                                      self.dt_temp_fut, Battery_.TB_FILT, rmax=Battery_.T_RLIM,
                                                      rmin=-Battery_.T_RLIM)
            self.Tb_model_filt_rate_fut = self.TbModelFilt.rate
        else:
            mon.Tb_model_filt = \
                self.TbModelFilt.calculate_tau_seeded(mon.Tb_model, mon.Tb_model,
                                                      mon.reset_temp,
                                                      mon.dt_temp, Battery_.TB_FILT, rmax=Battery_.T_RLIM,
                                                      rmin=-Battery_.T_RLIM)
            mon.Tb_model_filt_rate = self.TbModelFilt.rate


        mon.Tb_hdwe_filt_rate = self.TbSenseFilt.rate
        if not mon.reset_temp:
            mon.Tb_rap = self.Tb_past
            mon.Tb_f_rate = mon.Tb_model_filt_rate
        if rp.modeling_Tb:
            mon.Tb = mon.Tb_model
            mon.Tb_f = mon.Tb_model_filt
            mon.Tb_f_rate = mon.Tb_model_filt_rate
            mon.Tb_rstate = self.TbModelFilt.rstate
            mon.Tb_state = self.TbModelFilt.state
        else:
            mon.Tb = mon.Tb_hdwe
            mon.Tb_f = mon.Tb_hdwe_filt
            mon.Tb_f_rate = mon.Tb_hdwe_filt_rate
            mon.Tb_rstate = self.TbSenseFilt.rstate
            mon.Tb_state = self.TbSenseFilt.state
        self.assign_tb(mon.Tb, mon.Tb_f, mon.Tb_f_rate)
        if i_temp > 2:
            pass
        return mon

    def update_ekf(self, i_ekf):
        self.z_init = self.z[i_ekf]

    def update(self, i):
        self.i = min(max(i, 0), len(self.mon_run.time)-1)

    def update_ib_vb(self, i):
        self.i = min(max(i, 0), len(self.mon_run.time)-1)
        self.LoopAmp.update(i)
        self.LoopNoa.update(i)

        if hasattr(self.mon_run, 'kfres'):
            self.reset_kf = bool(self.mon_run.kfres[i])
            if hasattr(self.mon_run, 'vovcm'):
                self.VoVcm = self.mon_run.vovcm[i]
                self.KfShuntAmp.calculate(reset=self.reset_kf, dt=self.mon_run.ib_dyn_T_m[i], in_=self.VoVcm)
                self.VoVcm_f, _ = self.KfShuntAmp.get_state()
                self.VoVcm_f = float(self.VoVcm_f)
            self.VoVcn = self.mon_run.vovcn[i]
            self.KfShuntNoa.calculate(reset=self.reset_kf, dt=self.mon_run.ib_dyn_T_n[i], in_=self.VoVcn)
            self.VoVcn_f, _ = self.KfShuntNoa.get_state()
            self.VoVcn_f = float(self.VoVcn_f)
            self.iscn = float((self.VoVcn * Battery.SHUNT_NOA_GAIN + Battery.CURR_BIAS_NOA) / Battery.NP)
            self.iscn_f = float((self.VoVcn_f * Battery.SHUNT_NOA_GAIN + Battery.CURR_BIAS_NOA) / Battery.NP)
            # TODO:  implement iscn filter and scale with CURR_SCALE_DISCH (= 1. now everywhere so no worries at present)


    def update_tb(self):
        self.Tb_past = self.Tb
        self.Tb_f_past = self.Tb_f
        self.Tb_f_rate_past = self.Tb_f_rate
