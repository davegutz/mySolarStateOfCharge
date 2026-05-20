# Battery - general purpose battery class for modeling
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
# type: ignore
# noinspection PyAttributeOutsideInit,PyUnresolvedReferences,PyPep8Naming,PyShadowingNames,PyShadowingBuiltins,PyUnboundLocalVariable,PyUnfilledParameters
# pylint: disable=invalid-name, no-member, attribute-defined-outside-init, redefined-outer-name, redefined-builtin, used-before-assignment

"""Define a general purpose battery model including Randles' model and SoC-VOV model."""

import numpy as np
from filter.EKF1x1 import EKF1x1
from Coulombs import Coulombs, Chemistry
from hysteresis.Hysteresis import Hysteresis
import matplotlib.pyplot as plt
from filter.TFDelay import TFDelay
from filter.myFilters import LagExp, General2Pole, SlidingDeadband, TustinIntegrator
from filter.Scale import ScaleSelector
from plot.plq import plq
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})
# noinspection PyPep8Naming
import Globals as G
from battery_constants import BatteryConstants


class Retained:

    def __init__(self):
        self.cutback_gain_scalar = Battery.sp_cutback_gain_slr
        self.delta_q = 0.
        self.modeling = 0
        self.modeling_ib = False
        self.modeling_vb = False
        self.modeling_Tb = False
        self.tweak_test = False

    def add_modeling(self, modeling=None):
        self.modeling = modeling
        self.tweak_test = bool(0b1000 & int(self.modeling))
        self.modeling_ib = bool(0b0100 & int(self.modeling))
        self.modeling_vb = bool(0b0010 & int(self.modeling))
        self.modeling_Tb = bool(0b0001 & int(self.modeling))
        return self.modeling


def calculate_capacity(q_cap_rated_scaled=None, dqdt=None, tb_f=None, t_rated=None):
    q_cap = q_cap_rated_scaled * (1. + dqdt * (tb_f - t_rated))
    return q_cap


# noinspection PyPep8Naming
class Battery(BatteryConstants, Coulombs):

    # """Nominal battery bank capacity, Ah(100).Accounts for internal losses.This is
    #                         what gets delivered, e.g. Wshunt / NOMINAL_VB.  Also varies 0.2 - 0.4 C currents
    #                         or 20 - 40 A for a 100 Ah battery"""

    # Battery model:  Randles' dynamics, SOC-VOC model

    """Nominal battery bank capacity, Ah(100).Accounts for internal losses.This is
                            what gets delivered, e.g.Wshunt / NOMINAL_VB.  Also varies 0.2 - 0.4 C currents
                            or 20 - 40 A for a 100 Ah battery"""

    def __init__(self, OPT=None, q_cap_rated=BatteryConstants.NOM_UNIT_CAP*3600, temp_rlim=0.017,
                 t_rated=BatteryConstants.RATED_TEMP, Tb_f=BatteryConstants.NOMINAL_TB,
                 tweak_test=False, dvoc=0., mod_code=0, vsat_add=0., scale_cap=1., mon=None,
                 str_=None):
        """ Default values from Taborelli & Onori, 2013, State of Charge Estimation Using Extended Kalman Filters for
        Battery Management System.   Battery equations from LiFePO4 BattleBorn.xlsx and 'Generalized SOC-OCV Model Zhang
        etal.pdf.'  SOC-OCV curve fit './Battery State/BattleBorn Rev1.xls:Model Fit' using solver with min slope
        constraint >=0.02 V/soc.  m and n using Zhang values.   Had to scale soc because  actual capacity > NOM_BAT_CAP
        so equation error when soc<=0 to match data.    See Battery.h
        """
        # Parents
        Coulombs.__init__(self, OPT=OPT, q_cap_rated=q_cap_rated,  q_cap_rated_scaled=q_cap_rated, t_rated=t_rated,
                          tweak_test=tweak_test, dvoc=dvoc, Dw=Battery.sp_Dw)

        # Defaults
        self.time = -999.
        self.mod_data = 0.
        self.chem = mod_code
        self.nz = None
        self.q = 0  # Charge, C
        self.voc = Battery.NOMINAL_VB  # Model open circuit voltage, V
        self.voc_stat = self.voc  # Static model open circuit voltage from charge process, V
        self.voc_stat_past = self.voc_stat
        self.voc_stat_f = self.voc_stat
        self.dv_dyn = 0.  # Model current induced back emf, V
        self.ib_dyn = 0.  # Model current induced back emf before resistance multiply, A
        self.ib_dyn_T = 0.  # Randles update time, s
        self.ib_dyn_rstate = 0.  # Randles rstate, A
        self.ib_dyn_lstate = 0.  # Randles lstate, A
        self.vb = Battery.NOMINAL_VB  # Battery voltage at post, V
        self.ib = 0.  # Current into battery post, A
        self.ib_in = 0.  # Current into calculate, A
        self.ib_charge = 0.  # Current into count_coulombs, A
        self.ioc = 0  # Current into battery process accounting for hysteresis, A
        self.dv_dsoc = 0.  # Slope of soc-voc curve, V/%
        self.tcharge = 0.  # Charging time to 100%, hr
        self.sr = 1  # Resistance scalar
        self.vsat = self.chemistry.nom_vsat + vsat_add
        # range 0 - 50 C, V/deg C
        self.dt = 0.1  # Update time, s
        if OPT is not None:
            self.chemistry.r_0 *= OPT.slr_res_0
            self.chemistry.tau_ct *= OPT.stauct_mon
            self.chemistry.r_ct *= OPT.slr_res_ct
            self.chemistry.r_ss *= OPT.slr_r_ss
            if mon:
                self.s_hys = OPT.slr_hys_mon
                self.dvoc = OPT.add_voc_mon
            else:
                self.s_hys = OPT.slr_hys_sim
                self.dvoc = OPT.add_voc_sim
            self.chemistry.coul_eff *= OPT.slr_coul_eff
            if hasattr(OPT, 'unit'):
                self.unit = OPT.unit
        self.Tb = Tb_f
        self.Tb_f = Tb_f
        self.Tb_f_rate = None
        self.saved = Saved(str_)  # for plots and prints
        self.dv_hys = 0.  # Placeholder so BatterySim can be plotted
        self.tau_hys = 0.  # Placeholder so BatterySim can be plotted
        self.dv_dyn = 0.  # Placeholder so BatterySim can be plotted
        self.ib_dyn = 0.  # Placeholder so BatterySim can be plotted
        self.ib_dyn_T = 0.  # Placeholder so BatterySim can be plotted
        self.ib_dyn_rstate = 0.  # Placeholder so BatterySim can be plotted
        self.ib_dyn_lstate = 0.  # Placeholder so BatterySim can be plotted
        self.bms_off = False
        self.bms_off_past = self.bms_off
        self.mod = 7
        self.tweak_test = tweak_test
        self.ib_lag = 0.
        self.IbLag = LagExp(1., 1., -100., 100.)  # Lag to be run on saturation to produce ib_lag.  T and tau set at run time
        self.voc_soc = None
        self.voc_soc_new = 0.
        self.scale_cap = scale_cap
        self.Tb_rstate = None
        self.Tb_state = Tb_f
        self.Tb_hdwe_f = Tb_f
        self.Tb_hdwe_f_rate = 0.
        self.Tb_model_f_rate = 0.
        self.Tb_rstate = None
        self.Tb_state = Tb_f
        self.Tb_hdwe_f = Tb_f
        self.Tb_hdwe_f_rate = 0.
        self.Tb_model_f_rate = 0.
        self.reset = True
        self.voltage_low = False

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "Battery:\n"
        s += "  chem    = {:7.3f}  // Chemistry: 0=Battleborn, 1=CHINS\n".format(self.chem)
        s += "  tb  = {:7.3f}  // Battery temperature, deg C\n".format(self.Tb)
        s += "  dvoc_dt = {:9.6f}  // Change of VOC with operating temperature in range 0 - 50 C V/deg C\n"\
            .format(self.chemistry.dvoc_dt)
        s += "  r_0     = {:9.6f}  // Charge Transfer R0, ohms\n".format(self.chemistry.r_0)
        s += "  r_ct    = {:9.6f}  // Charge Transfer resistance, ohms\n".format(self.chemistry.r_ct)
        s += "  tau_ct = {:9.6f}  // Charge Transfer time constant, s (=1/Rdif/Cdif)\n".format(self.chemistry.tau_ct)
        s += "  r_ss    = {:9.6f}  // Steady state equivalent battery resistance, for solver, Ohms\n"\
            .format(self.chemistry.r_ss)
        s += "  r_sd    = {:9.6f}  // Equivalent model for EKF reference.	Parasitic discharge equivalent, ohms\n"\
            .format(self.chemistry.r_sd)
        s += "  tau_sd  = {:9.1f}  // Equivalent model for EKF reference.	Parasitic discharge time constant, sec\n"\
            .format(self.chemistry.tau_sd)
        s += "  bms_off  = {:7.1f}      // BMS off\n".format(self.bms_off)
        s += "  dv_dsoc = {:9.6f}  // Derivative scaled, V/fraction\n".format(self.dv_dsoc)
        s += "  ib =      {:7.3f}  // Battery terminal current, A\n".format(self.ib)
        s += "  ib_dyn =  {:7.3f}  // Current-induced back emf in current, A\n".format(self.ib_dyn)
        s += "  vb =      {:7.3f}  // Battery terminal voltage, V\n".format(self.vb)
        s += "  voc      ={:7.3f}  // Static model open circuit voltage, V\n".format(self.voc)
        s += "  voc_stat ={:7.3f}  // Estimated voc_soc (reference), V\n"\
            .format(self.voc_stat)
        s += "  voc_soc ={:7.3f}   // Static model open circuit voltage from table (reference), V\n"\
            .format(self.voc_soc)
        s += "  voc_stat_f={:7.3f} // Static filtered model open circuit voltage from table (reference), V\n"\
            .format(self.voc_stat_f)
        s += "  vsat =    {:7.3f}  // Saturation threshold at temperature, V\n".format(self.vsat)
        s += "  dv_dyn =  {:7.3f}  // Current-induced back emf, V\n".format(self.dv_dyn)
        s += "  q =       {:7.3f}  // Present charge available to use, except q_min_, C\n".format(self.q)
        s += "  sr =      {:7.3f}  // Resistance scalar\n".format(self.sr)
        s += "  dvoc_ =   {:7.3f}  // Delta voltage, V\n".format(self.dvoc)
        s += "  dt_ =     {:7.3f}  // Update time, s\n".format(self.dt)
        s += "  dv_hys  = {:7.3f}  // Hysteresis delta v, V\n".format(self.dv_hys)
        s += "  tau_hys = {:7.3f}  // Hysteresis time const, s\n".format(self.tau_hys)
        s += "  tweak_test={:d}     // Driving signal injection completely using software inj_soft_bias\n"\
            .format(self.tweak_test)
        s += "\n  "
        s += Coulombs.__str__(self, prefix + 'Battery:')
        return s

    def append_to(self, sv):
        """Append all scalar members of self to corresponding list members of sv (a Saved instance).
        If the attribute does not yet exist in sv, create it as a new list with the first value."""
        for key, val in vars(self).items():
            if isinstance(val, (bool, int, float, np.generic)) or val is None:
                if hasattr(sv, key):
                    getattr(sv, key).append(val)
                else:
                    setattr(sv, key, [val])

    def assign_tb(self, tb):
        self.Tb = tb

    def assign_tb_f(self, tb_f):
        self.Tb_f = tb_f

    def calc_h_jacobian(self, soc_lim, tb_f):
        if soc_lim > 0.5:
            dv_dsoc = (self.chemistry.lookup_voc(soc_lim, tb_f) -
                       self.chemistry.lookup_voc(soc_lim-0.01, tb_f)) / 0.01
        else:
            dv_dsoc = (self.chemistry.lookup_voc(soc_lim+0.01, tb_f) -
                       self.chemistry.lookup_voc(soc_lim, tb_f)) / 0.01
        return dv_dsoc

    def calc_soc_voc(self, soc, tb_f, printit=False):
        """SOC-OCV curve fit method per Zhang, etal """
        dv_dsoc = self.calc_h_jacobian(soc, tb_f)
        voc = self.chemistry.lookup_voc(soc, tb_f, printit=printit)
        if printit:
            print("soc=", soc, "tb_f=", tb_f, "dvoc=", self.dvoc, "voc=", voc)
        return voc, dv_dsoc

    def calculate(self, chem, Tb, Tb_f, vb, ib, dt, reset, calc_ekf, dt_ekf, SN, OPT,
                  q_capacity=None, rp=None, soc=None, saturated_init=None, reset_ekf=None, i=None,
                  i_ekf=None):
        # Battery
        raise NotImplementedError

    def look_hys(self, dv, soc):
        raise NotImplementedError


# noinspection PyPep8Naming
class BatteryMonitor(Battery, EKF1x1):
    """Extend Battery class to make a monitor"""
    def __init__(self, OPT=None, SN=None, q_cap_rated=Battery.NOM_UNIT_CAP*3600,
                 t_rated=Battery.RATED_TEMP, temp_rlim=0.017, scale=1.,
                 Tb_f=Battery.NOMINAL_TB, tweak_test=False, dvoc=0.,
                 mod_code=0, vsat_add=0.):
        ref = None
        if hasattr(OPT, 'slr_res_0'):
            ref = OPT.mon_run
        else:
            pass
        q_cap_rated_scaled = q_cap_rated * scale
        Battery.__init__(self, OPT=OPT, q_cap_rated=q_cap_rated_scaled, t_rated=t_rated, temp_rlim=temp_rlim, Tb_f=Tb_f,
                         tweak_test=tweak_test, dvoc=dvoc, mod_code=mod_code, scale_cap=scale, mon=True, str_='ver',
                         vsat_add=vsat_add)

        """ Default values from Taborelli & Onori, 2013, State of Charge Estimation Using Extended Kalman Filters for
        Battery Management System.   Battery equations from LiFePO4 BattleBorn.xlsx and 'Generalized SOC-OCV Model Zhang
        etal.pdf.'  SOC-OCV curve fit './Battery State/BattleBorn Rev1.xls:Model Fit' using solver with min slope
        constraint >=0.02 V/soc.  m and n using Zhang values.   Had to scale soc because  actual capacity > NOM_BAT_CAP
        so equations error when soc<=0 to match data.    See Battery.h
        """
        # Parents
        self.soc_ekf = 0.  # Filtered state of charge from ekf (0-1)
        EKF1x1.__init__(self)
        self.time_min = self.time / 60.
        self.time_day = self.time_min / 60. / 24.
        self.tcharge_ekf = 0.  # Charging time to 100% from ekf, hr
        self.voc = 0.  # Charging voltage, V
        self.q_ekf = 0  # Filtered charge calculated by ekf, C
        self.amp_hrs_remaining_ekf = 0  # Discharge amp*time left if drain to q_ekf=0, A-h
        self.amp_hrs_remaining_wt = 0  # Discharge amp*time left if drain soc_wt_ to 0, A-h
        self.e_soc_ekf = 0.  # analysis parameter
        self.e_voc_ekf = 0.  # analysis parameter
        self.Q = Battery.EKF_Q_SD_NORM * Battery.EKF_Q_SD_NORM  # EKF process uncertainty
        self.R = Battery.EKF_R_SD_NORM * Battery.EKF_R_SD_NORM  # EKF state uncertainty
        self.soc_s = 0.  # Model information
        self.EKF_converged = TFDelay(False, Battery.EKF_T_CONV, Battery.EKF_T_RES, Battery.EKF_NOM_DT)
        self.voc_stat_filt = LagExp(self.EKF_NOM_DT, self.VOC_STAT_FILT, self.VB_MIN, self.VB_MAX)  # Lag to be run on saturation to produce ib_lag.  T and tau set at run time
        self.y_ekf_filt_lag = LagExp(0.1, Battery.TAU_Y_FILT, Battery.MIN_Y_FILT, Battery.MAX_Y_FILT)
        self.WrapErrFilt = LagExp(0.1, Battery.WRAP_ERR_FILT, -Battery.MAX_WRAP_ERR_FILT, Battery.MAX_WRAP_ERR_FILT)
        self.y_filt = 0.
        self.ChargeTransfer = LagExp(dt=Battery.EKF_NOM_DT, max_=Battery.NOM_UNIT_CAP*scale,
                                     min_=-Battery.NOM_UNIT_CAP*scale, tau=self.chemistry.tau_ct)
        self.ib = 0.
        self.vb = 0.
        self.vb_model_rev = 0.
        self.voc_stat = 0.
        self.voc_stat_f = 0.
        self.voc = 0.
        self.voc_dead = 0.
        self.vsat = 0.
        self.dv_dyn = 0.
        self.ib_amp_hdwe = 0.
        self.ib_amp_model = 0.
        self.ib_noa_hdwe = 0.
        self.ib_noa_model = 0.
        self.ib_hdwe = 0.
        self.vb_model = 0.
        self.vb_hdwe = 0.
        self.vb_hdwe_f = 0.
        self.ib_dyn = 0.
        self.ib_dyn_in = 0.
        self.ib_dyn_r = False
        self.ib_dyn_T = 0.
        self.ib_dyn_rstate = 0.
        self.ib_dyn_lstate = 0.
        self.voc_stat_f_rstate = 0.
        self.voc_stat_f_lstate = 0.
        self.voc_stat_f_tau = 0.
        self.voc_stat_f_T = 0.
        self.voc_ekf = 0.
        self.Diff = Diff(self.dt)
        self.eframe = 0
        if OPT is not None:
            self.eframe_mult = OPT.eframe_mult
            self.dt_eframe = self.dt*self.ap_eframe_mult
        self.sdb_voc = SlidingDeadband(Battery.HDB_VB)
        self.e_wrap = 0.
        self.e_wrap_filt = 0.
        self.e_wrap_rate = 0.
        self.ib_past = 0.
        self.vb_past = 0.
        self.dt_past = 0.
        self.ib_amp = 0.
        self.ib_amp_pst = 0.
        self.ib_noa = 0.
        self.ib_noa_pst = 0.
        self.ib_noa_2pst = 0.
        self.e_wrap_m = None
        self.e_wrap_m_filt = None
        self.e_wrap_m_trim = None
        self.e_wrap_n = None
        self.e_wrap_n_filt = None
        self.e_wrap_n_trim = None
        self.e_wrap_n_rate = None
        self.e_wrap_m_rate = None
        self.disable_amp_fault = False
        self.LoopIbAmp = Looparound(Mon_=self, wrap_hi_volt=Battery.WRAP_HI_AMPV,
                                    wrap_lo_volt=Battery.WRAP_LO_AMPV,
                                    max_err=Battery.MAX_WRAP_ERR_FILT/(Battery.IB_ABS_MAX_NOA/Battery.IB_ABS_MAX_AMP),
                                    name="Amp")
        self.LoopIbNoa = Looparound(Mon_=self, wrap_hi_volt=Battery.WRAP_HI_NOAV,
                                    wrap_lo_volt=Battery.WRAP_LO_NOAV,
                                    max_err=Battery.MAX_WRAP_ERR_FILT, name="Noa")
        self.ewnhi_thr = 0.
        self.ewnlo_thr = 0.
        self.ewmhi_thr = 0.
        self.ewmlo_thr = 0.
        self.e_wrap_m_reset = True
        self.reset_ekf = None
        self.voc_stat_ekf = 0.
        self.dt_temp = None
        self.reset_temp = True
        self.Tb_rap = None
        self.Tb_model = None
        self.Tb_model = None
        self.Tb_f = None
        self.Tb_f_rate_rap = None
        self.dt_temp = 0.
        self.sel_brk_hdwe = ScaleSelector(Battery.HDWE_IB_HI_LO_NOA_LO, Battery.HDWE_IB_HI_LO_AMP_LO,
                                          Battery.HDWE_IB_HI_LO_AMP_HI, Battery.HDWE_IB_HI_LO_NOA_HI)
        self.reset_kf = True
        self.iscn_f = 0.
        self.frz = False
        self.wrap_hi_m_flt = False
        self.wrap_hi_m_fa = False
        self.wrap_lo_m_flt = False
        self.wrap_lo_m_fa = False
        self.wrap_hi_n_flt = False
        self.wrap_hi_n_fa = False
        self.wrap_lo_n_flt = False
        self.wrap_lo_n_fa = False

        self.ib_model = 0.
        self.ib_h  = 0.
        self.ib_s  = 0.
        self.mib  = False
        self.mvb = False
        self.mtb = False
        self.ib_diff  = 0.
        self.ib_dyn_m  = 0.
        self.ib_dyn_T_m  = 0.
        self.ib_dyn_rstate_m  = 0.
        self.ib_dyn_lstate_m  = 0.
        self.ib_dyn_tau_m  = 0.
        self.dv_dyn_m  = 0.
        self.voc_m = 0.
        self.voc_soc_m  = 0.
        self.ib_wrp_T_m  = 0.
        self.ib_wrp_tau_m  = 0.
        self.ib_wrp_state_m  = 0.
        self.ib_wrp_rate_m  = 0.
        self.ib_wrp_reset_m  = 0.
        self.e_wrap_m_trim  = 0.
        self.e_wrap_m_trimmed = 0.
        self.ib_dyn_n = 0.
        self.ib_dyn_T_n = 0.
        self.ib_dyn_rstate_n  = 0.
        self.ib_dyn_lstate_n = 0.
        self.ib_dyn_tau_n = 0.
        self.dv_dyn_n  = 0.
        self.ib_wrp_T_n  = 0.
        self.ib_wrp_tau_n  = 0.
        self.ib_wrp_state_n  = 0.
        self.ib_wrp_rate_n  = 0.
        self.e_wrap_n_trimmed  = 0.
        self.vb_h_f  = 0.
        self.y_ekf = 0.
        self.y_ekf_f = 0.
        self.qcap  = 0.
        self.qcrs  = 0.
        self.cc_dif  = 0.

        if SN is not None:
            self.Tb_hdwe = SN.mon_run.Tb_hdwe[0]
            self.Tb_hdwe_f = SN.mon_run.Tb_hdwe_f[0]
            self.Tb_hdwe_f_rate = SN.mon_run.Tb_hdwe_f_rate[0]
            self.Tb_model_f =SN.mon_run.Tb_model_f[0]
            self.Tb_model_f_rate = SN.mon_run.Tb_model_f_rate[0]
            self.Tb = SN.mon_run.Tb[0]
            self.Tb_f = SN.mon_run.Tb_f[0]
            self.Tb_hdwe = SN.mon_run.Tb_hdwe[0]
            self.Tb_hdwe_f =SN.mon_run.Tb_hdwe_f[0]
            self.Tb_hdwe_f_rate =SN.mon_run.Tb_hdwe_f_rate[0]
            self.Tb_model = SN.mon_run.Tb_model[0]
            self.Tb_model_f = SN.mon_run.Tb_model_f[0]
            self.Tb_model_f_rate = SN.mon_run.Tb_model_f_rate[0]
            self.e_wrap = SN.e_wrap_init
            self.e_wrap_filt = SN.mon_run.e_wrap_filt[0]
            self.ib_amp_lo = False
            self.ib_noa_lo = False
            self.ib_amp_hi = False
            self.ib_noa_hi = False
            self.e_wrap_m = SN.e_wrap_m_init
            self.e_wrap_m_filt = SN.mon_run.e_wrap_m_filt[0]
            self.e_wrap_m_trim = SN.mon_run.e_wrap_m_trim[0]
            self.e_wrap_n = SN.e_wrap_n_init
            self.e_wrap_n_filt = SN.mon_run.e_wrap_n_filt[0]
            self.e_wrap_n_trim = SN.mon_run.e_wrap_n_trim[0]
            self.e_wrap_n_trim = 0.
            self.voc_soc = SN.mon_run.voc_soc[0]
            self.voc_stat = self.voc_soc - self.e_wrap
            self.Tb = SN.mon_run.Tb_f[0]
            self.Tb_f = SN.Tb_f
            self.Tb_f_rate = SN.mon_run.Tb_f_rate[0]
            self.Tb_model = SN.mon_run.Tb_model[0]
            self.ib = SN.ib_init
            self.ib_dyn = SN.ib_dyn[0]
            self.ib_charge = SN.ib_charge_init
            self.ib_charge_ekf = self.ib_charge
            self.vb = SN.vb_init
            self.soc = SN.mon_run.soc[0]
            self.reset = True
            self.sat = SN.sat_init
            self.saturated = SN.mon_run.saturated[0]
            self.reset_ekf = True
            self.init_soc_ekf(ref,  0, 0)
            self.voc_ekf = SN.hx_init
            self.x = SN.x_init
            self.x_prior = SN.x_prior_init
            self.soc_ekf = SN.soc_ekf_init
            self.z = SN.z_init
        self.dt_s = 0.
        self.chm_s = 0.
        self.qcrs_s = 0.
        self.qcap_s = 0.
        self.bms_off_s = 0.
        # self.Tb_s = 0.
        # self.Tb_f_s = 0.
        self.vsat_s = 0.
        self.voc_s = 0.
        self.voc_stat_s = 0.
        self.dv_dyn_s = 0.
        self.d_delta_q_s = 0.
        self.delta_q_s = 0.
        self.dv_hys_s = 0.
        self.ib_charge_s = 0.
        self.ib_dyn_s = 0.
        self.ib_in_s = 0.
        self.ib_s = 0.
        self.ioc_s = 0.
        self.sat_s = 0.
        self.soc_s = 0.
        self.vb_s = 0.
        self.ib_dyn_T_s = 0.
        self.ib_dyn_lstate_s = 0.
        self.ib_dyn_rstate_s = 0.
        self.ib_dyn_tau_s = 0.
        self.tau_hys_s = 0.
        self.vb_s = 0.
        self.q_s = 0.
        self.ib_fut_s = 0.
        self.reset_s = 0.
        self.tau_s = 0.
        self.tau_hys_s = 0.
        self.kf_v_m = 0.
        self.kf_v_n = 0.
        self.y_ekf_f_T = 0.
        self.y_ekf_f_tau = 0.
        self.y_ekf_f_state = 0.
        self.Tb_fa = False
        self.ib_lo_limited_lo = False
        self.ib_lo_limited_hi = False
        self.ib_lo_active = True
        self.IbLoLimitedLo = TFDelay(in_=False, t_true=Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                                     t_false=Battery.IB_LO_ACTIVE_RES*Battery.cp_ts, dt=0.1)
        self.IbLoLimitedHi = TFDelay(in_=False, t_true=Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                                     t_false=Battery.IB_LO_ACTIVE_RES*Battery.cp_ts, dt=0.1)

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix
        s += Battery.__str__(self, prefix + 'BatteryMonitor:')
        s += "  amp_hrs_remaining_ekf_ =  {:7.3f}  // Discharge amp*time left if drain to q_ekf=0, A-h\n"\
            .format(self.amp_hrs_remaining_ekf)
        s += "  amp_hrs_remaining_wt_  =  {:7.3f}  // Discharge amp*time left if drain soc_wt_ to 0, A-h\n"\
            .format(self.amp_hrs_remaining_wt)
        s += "  q_ekf     {:7.3f}  // Filtered charge calculated by ekf, C\n".format(self.q_ekf)
        s += "  soc_ekf = {:7.3f}  // Solved state of charge, fraction\n".format(self.soc_ekf)
        s += "  tcharge = {:7.3f}  // Charging time to full, hr\n".format(self.tcharge)
        s += "  tcharge_ekf = {:7.3f}   // Charging time to full from ekf, hr\n".format(self.tcharge_ekf)
        s += "  mod     =               {:f}  // Modeling\n".format(self.mod)
        s += "\n  "
        s += EKF1x1.__str__(self, prefix + 'BatteryMonitor:')
        return s

    def assign_soc_s(self, soc_s):
        self.soc_s = soc_s

    # BatteryMonitor::calculate()
    # It is assumed that ekf always runs slower than subsampled input data stream
    # (EKF_EFRAME_MULT multi-frame always <= DP)
    def calculate(self, chem, Tb, Tb_f, vb, ib, dt, reset, calc_ekf, dt_ekf, SN, OPT,
                  q_capacity=None, rp=None, soc=None, saturated_init=None, reset_ekf=None, i=None,
                  i_ekf=None):
        self.reset = reset
        self.Tb = Tb
        self.Tb_f = Tb_f
        self.vb = vb
        self.ib_in = ib
        self.dt = dt
        self.ib_amp_hdwe = SN.mon_run.ib_amp_hdwe[G.i]
        self.ib_amp_model = SN.ib_amp_model[G.i]
        self.ib_noa_hdwe = SN.mon_run.ib_noa_hdwe[G.i]
        self.ib_noa_model = SN.mon_run.ib_noa_model[G.i]
        if  getattr(SN.mon_run, 'vb_model', None) is not None:
            self.vb_model = SN.mon_run.vb_model[G.i]
        if  getattr(SN.mon_run, 'vb_hdwe', None) is not None:
            self.vb_hdwe = SN.mon_run.vb_hdwe[G.i]
        if  getattr(SN.mon_run, 'vb_hdwe_f', None) is not None:
            self.vb_hdwe_f = SN.mon_run.vb_hdwe_f[G.i]
        if rp.modeling_ib:
            self.ib_amp = self.ib_amp_model
            self.ib_noa = self.ib_noa_model
            self.ib_amp_pst = SN.mon_run.ib_amp_model[max(G.i-1, 0)]
            self.ib_noa_pst = SN.mon_run.ib_noa_model[max(G.i-1, 0)]
        else:
            self.ib_amp = self.ib_amp_hdwe
            self.ib_noa = self.ib_noa_hdwe
            self.ib_amp_pst = SN.mon_run.ib_amp_hdwe[max(G.i-1, 0)]
            self.ib_noa_pst = SN.mon_run.ib_noa_hdwe[max(G.i-1, 0)]
        self.ib_hdwe = SN.mon_run.ib_h[G.i]
        if self.chm != chem:
            self.chemistry.assign_all_mod(chem, unit=self.unit)
            self.chm = chem

        self.vsat = self.chemistry.nom_vsat + (self.Tb_f - 25.) * self.chemistry.dvoc_dt + Battery.sp_vsat_add
        self.mod = rp.modeling
        # Overflow protection since ib past value used
        self.ib = max(min(self.ib_in, Battery.IMAX_NUM), -Battery.IMAX_NUM)

        # Ib diff logic
        self.ib_diff = self.Diff.calculate(reset=reset, dt=self.dt, ib_amp=self.ib_amp,  ib_noa=self.ib_noa)

        # Wrap logic
        self.wrap(reset=reset, ib_noa_hdwe=self.ib_noa_hdwe, SN=SN, ib_amp=self.ib_amp,
                  ib_noa=self.ib_noa, ib_amp_pst=self.ib_amp_pst, ib_noa_pst=self.ib_noa_pst, rp=rp)

        # Reversionary model
        self.vb_model_rev = self.voc_soc + self.dv_dyn + self.dv_hys

        # Table lookup
        self.voc_soc, self.dv_dsoc = self.calc_soc_voc(self.soc, self.Tb_f)

        # Battery management system model (uses past value bms_off and voc_stat)
        self.bms_off_past = self.bms_off
        if self.reset:
            self.bms_off_past = SN.mon_run.bms_off[G.i-1]
            self.voltage_low = self.bms_off = SN.mon_run.bms_off[G.i]
        else:
            if not self.bms_off:
                voltage_low_past = self.voltage_low
                self.voltage_low = self.voc_stat < self.chemistry.vb_down
                if (self.voltage_low != voltage_low_past) and not self.reset:
                    print(f"\nBMS OFF voc_stat {self.voc_stat} vb_down {self.chemistry.vb_down} vb_rising {self.chemistry.vb_rising} bms_off {self.bms_off} voltage_low {self.voltage_low} \n\n")
            else:
                voltage_low_past = self.voltage_low
                self.voltage_low = self.voc_stat < self.chemistry.vb_rising
                if self.voltage_low != voltage_low_past:
                    print(f"\nBMS ON voc_stat {self.voc_stat} vb_down {self.chemistry.vb_down} vb_rising {self.chemistry.vb_rising} bms_off {self.bms_off} voltage_low {self.voltage_low} \n\n")
        bms_charging = self.ib > Battery.IB_MIN_UP
        if not self.reset:
            self.bms_off = ( (self.Tb_f <= self.chemistry.low_t and not self.Tb_flt) or
                            (SN.mon_run.ib_really_quiet is not None and SN.mon_run.ib_really_quiet[G.i] and self.voltage_low and not rp.tweak_test) )  # KISS
        self.ib_charge = self.ib
        self.ib_charge_ekf = self.ib_charge
        if self.bms_off and not bms_charging:
            self.ib_charge = 0.
        if self.bms_off and self.voltage_low:
            self.ib = 0.
        self.ib_lag = self.IbLag.calculate_tau(self.ib, reset, self.dt, self.chemistry.ib_lag_tau)

        # Dynamic emf
        if rp.modeling_ib:
            dt_local = self.dt
            ib_dc = self.ib_past
        else:
            dt_local = self.dt
            ib_dc = self.ib_past
        self.ib_dyn = self.ChargeTransfer.calculate_tau_seeded(ib_dc, SN.ib_dyn[G.i], reset, dt_local,
                                                               self.chemistry.tau_ct)
        self.ib_dyn_in = self.ChargeTransfer.in_
        self.ib_dyn_r = self.ChargeTransfer.reset
        self.ib_dyn_T = self.ChargeTransfer.dt
        self.ib_dyn_rstate = self.ChargeTransfer.rstate
        self.ib_dyn_lstate = self.ChargeTransfer.state
        self.voc = self.vb - (self.ib_dyn*self.chemistry.r_ct + ib_dc*self.chemistry.r_0)
        self.voc_stat_past = self.voc_stat
        if self.bms_off and self.voltage_low:
            self.voc_stat = self.vb
            self.voc = self.vb
        self.dv_dyn = self.vb - self.voc

        # Hysteresis model
        self.dv_hys = 0.
        self.voc_stat = self.voc - self.dv_hys
        if reset:
            self.voc_stat = self.voc
        self.ioc = self.ib

        # EKF 1x1
        self.reset_ekf = reset_ekf
        if calc_ekf:
            self.voc_stat_ekf = self.voc_stat
            self.dt_eframe = dt_ekf
            ddq_dt = self.ib_charge_ekf
            if ddq_dt > 0. and not self.tweak_test:
                ddq_dt *= self.chemistry.coul_eff
            # ddq_dt -= self.chemistry.dqdt*self.q_capacity*temp_rate  7/29/2025 to agree with c++ (noisy)
            self.Q = Battery.EKF_Q_SD_NORM**2  # override
            self.R = Battery.EKF_R_SD_NORM**2  # override
            self.voc_stat_f =\
                self.voc_stat_filt.calculate_tau_seeded(self.voc_stat_ekf, SN.mon_run.voc_stat_f_lstate[i_ekf],
                                                        self.reset_ekf, self.dt_eframe, self.VOC_STAT_FILT)
            self.voc_stat_f_rstate = self.voc_stat_filt.rstate
            self.voc_stat_f_lstate = self.voc_stat_filt.state
            self.voc_stat_f_tau = self.voc_stat_filt.tau
            self.voc_stat_f_T = self.voc_stat_filt.dt
            self.frz = self.bms_off
            mr = SN.mon_run
            self.predict_ekf(u=ddq_dt, reset=self.reset_ekf, freeze=self.frz, OPT=OPT, i_ekf=i_ekf)  # u = d(q)/dt
            self.update_ekf(z=self.voc_stat_f, x_min=0., x_max=Battery.MXEPS, OPT=OPT, i_ekf=i_ekf)  # z = voc, voc_filtered = hx
            self.soc_ekf = self.x  # x = Vsoc (0-1 ideal capacitor voltage) proxy for soc
            self.y_ekf = self.y
            self.q_ekf = self.soc_ekf * self.q_capacity
            self.y_ekf_f = self.y_ekf_filt_lag.calculate_seeded(in_=self.y_ekf, _out_init=OPT.mon_run.y_ekf_f[G.i],
                                                                dt=self.dt_eframe, reset=self.reset_temp)
            self.y_ekf_f_T = self.y_ekf_filt_lag.dt
            self.y_ekf_f_tau = self.y_ekf_filt_lag.tau
            self.y_ekf_f_state = self.y_ekf_filt_lag.state
            # EKF convergence
            conv = abs(self.y_filt) < Battery.EKF_CONV
            self.EKF_converged.calculate(conv, Battery.EKF_T_CONV, Battery.EKF_T_RES, self.dt_eframe, self.reset_ekf)
            # print(f"{reset_ekf=} {self.soc_ekf} {self.x=} {self.voc_stat_ekf=}")
        self.eframe += 1
        if self.reset_ekf or self.eframe >= self.ap_eframe_mult:  # '>=' ensures reset with changes on the fly
            self.eframe = 0

        # Filtered voc
        self.voc_dead = self.sdb_voc.update_res(self.voc, reset)

        # Charge time
        if self.ib_charge > 0.1:
            self.tcharge_ekf = min(Battery.NOM_UNIT_CAP/self.ib_charge * (1. - self.soc_ekf), 24.)
        elif self.ib_charge < -0.1:
            self.tcharge_ekf = max(Battery.NOM_UNIT_CAP/self.ib_charge * self.soc_ekf, -24.)
        elif self.ib_charge >= 0.:
            self.tcharge_ekf = 24.*(1. - self.soc_ekf)
        else:
            self.tcharge_ekf = -24.*self.soc_ekf

        self.dv_dyn = self.dv_dyn
        self.voc_ekf = self.hx
        self.ib_past = self.ib
        self.vb_past = self.vb
        self.dt_past = self.dt

        return self.vb_model_rev

    def calc_charge_time(self, q, q_capacity, charge_curr, soc):
        delta_q = q - q_capacity
        if charge_curr > Battery.TCHARGE_DISPLAY_DEADBAND:
            self.tcharge = min(-delta_q / charge_curr / 3600., 24.)
        elif charge_curr < -Battery.TCHARGE_DISPLAY_DEADBAND:
            self.tcharge = max((q_capacity + delta_q - self.q_min) / charge_curr / 3600., -24.)
        elif charge_curr >= 0.:
            self.tcharge = 24.
        else:
            self.tcharge = -24.

        amp_hrs_remaining = (q_capacity - self.q_min + delta_q) / 3600.
        if soc > self.soc_min:
            self.amp_hrs_remaining_ekf = amp_hrs_remaining * (self.soc_ekf - self.soc_min) /\
                (soc - self.soc_min)
            self.amp_hrs_remaining_wt = amp_hrs_remaining * (self.soc - self.soc_min) /\
                (soc - self.soc_min)
        elif soc < self.soc_min:
            self.amp_hrs_remaining_ekf = amp_hrs_remaining * (self.soc_ekf - self.soc_min) / (soc - self.soc_min)
            self.amp_hrs_remaining_wt = amp_hrs_remaining * (self.soc - self.soc_min) / (soc - self.soc_min)
        else:
            self.amp_hrs_remaining_ekf = 0.
            self.amp_hrs_remaining_wt = 0.
        return self.tcharge

    # def count_coulombs(self, dt=0., ...):
    #     raise NotImplementedError

    def converged_ekf(self):
        return self.EKF_converged.state()

    def ekf_predict(self):
        """Process model"""
        self.Fx = 1. - self.dt_eframe / self.chemistry.tau_sd
        self.Bu = self.dt_eframe / self.chemistry.tau_sd * self.chemistry.r_sd
        return self.Fx, self.Bu

    def ekf_update(self):
        # Measurement function hx(x), x = soc ideal capacitor
        x_lim = max(min(self.x, Battery.MXEPS), 0.)
        self.x_for_hx = x_lim
        self.tb_f_for_hx = self.Tb_f
        self.hx, self.dv_dsoc = self.calc_soc_voc(x_lim, tb_f=self.tb_f_for_hx, printit=False)
        # Jacobian of measurement function
        self.H = self.dv_dsoc
        return self.hx, self.H, self.tb_f_for_hx, self.x_for_hx

    def init_soc_ekf(self, mr, i, i_ekf):
        if mr is None:
            return
        self.soc_ekf = mr.soc_ekf[i]
        if hasattr(mr, 'y'):
            self.y_ekf = mr.y[i_ekf]
        else:
            self.y_ekf = mr.y_ekf[i_ekf]
        self.init_ekf(mr.soc_ekf[i], 0.0)
        self.q_ekf = self.soc * self.q_capacity
        self.P = mr.P[i_ekf]

        if hasattr(mr, 'P_post'):
            self.P_post = mr.P_post[i_ekf]
        else:
            self.P_post = self.P

        if hasattr(mr, 'P_prior'):
            self.P_prior = mr.P_prior[i_ekf]
        else:
            self.P_prior = self.P

        if hasattr(mr, 'H'):
            self.H = mr.H[i_ekf]
        else:
            self.H = mr.z[i_ekf]

        if hasattr(mr, 'S'):
            self.S = mr.S[i_ekf]
        else:
            self.S = 0.

        if hasattr(mr, 'K'):
            self.K = mr.K[i_ekf]
        else:
            self.K = 0.

        if hasattr(mr, 'hx'):
            self.hx = mr.hx[i_ekf]
        else:
            self.hx = mr.voc_f[i]

        if hasattr(mr, 'dt_ekf'):
            self.dt_eframe = mr.dt_ekf[i_ekf]
        else:
            self.dt_eframe = mr.dt[i] * Battery.EKF_EFRAME_MULT

        self.x = mr.x[i_ekf]

        self.x_prior = mr.x_prior[i_ekf]

        if hasattr(mr, 'x_post'):
            self.x_post = mr.x_post[i_ekf]
        else:
            self.x_post = self.x

        if hasattr(mr, 'tb_f_for_hx'):
            try:
                self.tb_f_for_hx = mr.tb_f_for_hx[i_ekf]
            except IndexError:
                pass
        else:
            self.tb_f_for_hx = self.Tb_f

        if hasattr(mr, 'x_for_hx'):
            self.x_for_hx = mr.x_for_hx[i_ekf]
        else:
            self.x_for_hx = self.x

    def regauge(self, tb_f):
        if self.converged_ekf() and abs(self.soc_ekf - self.soc) > Battery.DF2:
            print("Resetting Coulomb Counter Monitor from ", self.soc, " to EKF=", self.soc_ekf, "...")
            self.apply_soc(self.soc_ekf, tb_f)
            print("confirmed ", self.soc)

    def save(self, time, dt, soc_run, voc_run, SN, rp, sim):  # BatteryMonitor
        self.time = time
        self.dt = dt
        self.time_min = self.time / 60.
        self.time_day = self.time_min / 60. / 24.
        if abs(soc_run) < 1e-6:
            soc_run = 1e-6
        self.e_soc_ekf = (self.soc_ekf - soc_run) / soc_run
        self.e_voc_ekf = (self.voc - voc_run) / voc_run
        self.iscn_f = SN.iscn_f
        self.mod_data = self.mod
        self.ib_model = SN.ib_in_s
        self.ib_h = self.ib_hdwe
        self.ib_s = SN.sim_run.ib_in_s
        self.mib = rp.modeling_ib
        self.mvb = rp.modeling_vb
        self.mtb = rp.modeling_Tb
        self.kf_v_m = SN.kf_v_m
        self.kf_v_n = SN.kf_v_n
        self.ib_dyn_m = self.LoopIbAmp.ib_dyn
        self.ib_dyn_rstate_m = self.LoopIbAmp.ChargeTransfer.rstate
        self.ib_dyn_lstate_m = self.LoopIbAmp.ChargeTransfer.state
        self.ib_dyn_tau_m = self.LoopIbAmp.ChargeTransfer.tau
        self.dv_dyn_m = self.LoopIbAmp.dv_dyn
        self.voc_m = self.LoopIbAmp.voc
        self.voc_soc_m = self.LoopIbAmp.voc_soc
        self.ib_wrp_T_m = self.LoopIbAmp.WrapErrFilt.dt
        self.ib_wrp_tau_m = self.LoopIbAmp.WrapErrFilt.tau
        self.ib_wrp_state_m = self.LoopIbAmp.WrapErrFilt.state
        self.ib_wrp_rate_m = self.LoopIbAmp.e_wrap_rate
        self.ib_wrp_reset_m = self.LoopIbAmp.reset
        self.e_wrap_m_trimmed = self.LoopIbAmp.e_wrap_trimmed
        self.ib_dyn_n = self.LoopIbNoa.ib_dyn
        self.ib_dyn_rstate_n = self.LoopIbNoa.ChargeTransfer.rstate
        self.ib_dyn_lstate_n = self.LoopIbNoa.ChargeTransfer.state
        self.ib_dyn_tau_n = self.LoopIbNoa.ChargeTransfer.tau
        self.dv_dyn_n = self.LoopIbNoa.dv_dyn
        self.ib_wrp_T_n = self.LoopIbNoa.WrapErrFilt.dt
        self.ib_wrp_tau_n = self.LoopIbNoa.WrapErrFilt.tau
        self.ib_wrp_state_n = self.LoopIbNoa.WrapErrFilt.state
        self.ib_wrp_rate_n = self.LoopIbNoa.e_wrap_rate
        self.e_wrap_n_trimmed = self.LoopIbNoa.e_wrap_trimmed
        self.y_ekf = self.y
        self.qcap = self.q_capacity
        self.qcrs = self.q_cap_rated_scaled
        self.cc_dif = self.soc_ekf - self.soc
        self.dt_s = sim.dt
        self.chm_s  = sim.chm
        self.qcrs_s  = sim.q_cap_rated_scaled
        self.qcap_s  = sim.q_capacity
        self.bms_off_s  = sim.bms_off
        self.Tb_s  = sim.Tb
        self.Tb_f_s  = sim.Tb_f
        self.vsat_s  = sim.vsat
        self.voc_s  = sim.voc
        self.voc_stat_s  = sim.voc_stat
        self.dv_dyn_s  = sim.dv_dyn_s
        self.d_delta_q_s  = sim.d_delta_q
        self.delta_q_s  = sim.delta_q
        self.dv_hys_s  = sim.dv_hys
        self.ib_charge_s  = sim.ib_charge
        self.ib_dyn_s  = sim.ib_dyn
        self.ib_in_s  = sim.ib_in
        self.ib_s  = sim.ib
        self.ioc_s  = sim.ioc
        self.sat_s  = sim.sat
        self.soc_s  = sim.soc
        self.vb_s  = sim.vb
        self.Tb_s = sim.Tb
        self.ib_dyn_T_s  = sim.ib_dyn_T
        self.ib_dyn_lstate_s  = sim.ib_dyn_lstate
        self.ib_dyn_rstate_s  = sim.ib_dyn_rstate
        self.ib_dyn_tau_s  = sim.chemistry.tau_ct
        self.tau_hys_s  = sim.tau_hys
        self.q_s  = sim.q
        self.ib_fut_s  = sim.ib_fut
        self.reset_s  = sim.reset
        self.tau_s  = sim.tau_hys
        self.tau_hys_s  = sim.tau_hys
        # Append all parameters to
        self.append_to(self.saved)
        pass

    def wrap(self, reset=True, ib_noa_hdwe=0., SN=None, ib_amp=0., ib_noa=0.,
             ib_amp_pst=None, ib_noa_pst=None, rp=None):
        """Wrap logic"""
        dt_local = self.dt

        # e_wrap scalars normally calculated in Sensors
        if self.soc >= Battery.WRAP_SOC_HI_OFF:
            ewsat_slr = Battery.WRAP_SOC_HI_SLR
            ewmin_slr = 1.
        elif self.soc <= max(self.soc_min+Battery.WRAP_SOC_LO_OFF_REL, Battery.WRAP_SOC_LO_OFF_ABS):
            ewsat_slr = 1.
            ewmin_slr = Battery.WRAP_SOC_LO_SLR
            #  else if ( Mon->voc_soc()>(Mon->vsat()-WRAP_HI_SETAT_MARG) ||
#          ( Mon->voc_stat()>(Mon->vsat()-WRAP_HI_SETAT_MARG) && Mon->C_rate()>WRAP_MOD_C_RATE && Mon->soc()>WRAP_SOC_MOD_OFF) ) // Use voc_stat to get some anticipation

        elif (
                self.voc_soc > (self.vsat - Battery.WRAP_HI_SETAT_MARG) or
                ((self.voc_stat > (self.vsat-Battery.WRAP_HI_SETAT_MARG)) and (self.ib / Battery.NOM_UNIT_CAP > Battery.WRAP_MOD_C_RATE) and
                    (self.soc > Battery.WRAP_SOC_MOD_OFF))
            ):
            ewsat_slr = Battery.WRAP_HI_SETAT_SLR
            ewmin_slr = 1.
        else:
            ewsat_slr = 1.
            ewmin_slr = 1.
        print(f"{self.voc_soc=} {self.vsat=} {self.voc_stat=} {self.ib / Battery.NOM_UNIT_CAP=} {self.soc=} {ewmin_slr=} {ewsat_slr=}")

        # Individual wrap logic
        if ib_noa is not None:
            if rp.modeling_ib or SN.run_type == 'HistSim':
                self.ib_noa = ib_noa
                self.ib_noa_pst = ib_noa_pst
                dt_local = self.dt
                ibnoa = self.ib_noa
            else:
                self.ib_noa = ib_noa
                self.ib_noa_pst = ib_noa_pst
                dt_local = self.dt_past
                ibnoa = self.ib_noa_pst
            self.LoopIbNoa.calculate(reset=reset, rp=rp, ib=ibnoa, loop_gain=Battery.NOA_WRAP_TRIM_GAIN,
                                     dt=dt_local, ewmin_slr=ewmin_slr, ewsat_slr=ewsat_slr,
                                     ib_dyn_init=SN.WrapLoopNoa.ib_dyn[G.i],
                                     e_wrap_filt_init=SN.mon_run.e_wrap_n_filt[G.i],
                                     e_wrap_trim_init=SN.mon_run.e_wrap_n_trim[G.i], freeze=False)
            self.ib_dyn_T_n = self.LoopIbNoa.ChargeTransfer.dt
            self.e_wrap_n = self.LoopIbNoa.e_wrap
            self.e_wrap_n_filt = self.LoopIbNoa.e_wrap_filt
            self.e_wrap_n_rate = self.LoopIbNoa.e_wrap_rate
            self.e_wrap_n_trim = self.LoopIbNoa.e_wrap_trim
            self.ewnhi_thr = self.LoopIbNoa.ewhi_thr
            self.ewnlo_thr = self.LoopIbNoa.ewlo_thr
            self.wrap_hi_n_flt = self.LoopIbNoa.hi_fault
            self.wrap_hi_n_fa = self.LoopIbNoa.hi_fail
            self.wrap_lo_n_flt = self.LoopIbNoa.lo_fault
            self.wrap_lo_n_fa = self.LoopIbNoa.lo_fail
        if ib_amp is not None:
            if rp.modeling_ib or SN.run_type == 'HistSim':
                self.ib_amp = ib_amp
                self.ib_amp_pst = ib_amp_pst
                ibamp = self.ib_amp
            else:
                self.ib_amp = ib_amp
                self.ib_amp_pst = ib_amp_pst
                ibamp = self.ib_amp_pst
            self.ib_amp_hi = self.ib_amp >= Battery.HDWE_IB_HI_LO_AMP_HI
            self.ib_amp_lo = self.ib_amp <= Battery.HDWE_IB_HI_LO_AMP_LO
            self.ib_noa_hi = self.ib_noa >= Battery.HDWE_IB_HI_LO_NOA_HI
            self.ib_noa_lo = self.ib_noa <= Battery.HDWE_IB_HI_LO_NOA_LO
            self.ib_lo_limited_lo = self.IbLoLimitedLo.calculate(self.ib_amp_lo, Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                                                                 Battery.IB_LO_ACTIVE_RES*Battery.cp_ts, dt=dt_local,
                                                                 reset=self.e_wrap_m_reset)
            self.ib_lo_limited_hi = self.IbLoLimitedHi.calculate(self.ib_amp_hi, Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                                                                 Battery.IB_LO_ACTIVE_RES*Battery.cp_ts, dt=dt_local,
                                                                 reset=self.e_wrap_m_reset)
            self.ib_lo_active = not self.ib_lo_limited_hi and not self.ib_lo_limited_lo
            self.disable_amp_fault = (self.ib_amp_hi and self.ib_noa_hi) or (self.ib_amp_lo and self.ib_noa_lo)
            self.e_wrap_m_reset = reset
            self.LoopIbAmp.calculate(reset=self.e_wrap_m_reset, rp=rp, ib=ibamp, loop_gain=Battery.AMP_WRAP_TRIM_GAIN,
                                     dt=dt_local, ewmin_slr=ewmin_slr, ewsat_slr=ewsat_slr,
                                     ib_dyn_init=SN.WrapLoopAmp.ib_dyn[G.i],
                                     e_wrap_filt_init=SN.mon_run.e_wrap_m_filt[G.i],
                                     e_wrap_trim_init=SN.mon_run.e_wrap_m_trim[G.i], freeze=not self.ib_lo_active)
            self.ib_dyn_T_m = self.LoopIbAmp.ChargeTransfer.dt
            self.ewmhi_thr = self.LoopIbAmp.ewhi_thr
            self.ewmlo_thr = self.LoopIbAmp.ewlo_thr
            self.e_wrap_m = self.LoopIbAmp.e_wrap
            self.e_wrap_m_filt = self.LoopIbAmp.e_wrap_filt
            self.e_wrap_m_rate = self.LoopIbAmp.e_wrap_rate
            self.e_wrap_m_trim = self.LoopIbAmp.e_wrap_trim
            self.wrap_hi_m_flt = self.LoopIbAmp.hi_fault
            self.wrap_hi_m_fa = self.LoopIbAmp.hi_fail
            self.wrap_lo_m_flt = self.LoopIbAmp.lo_fault
            self.wrap_lo_m_fa = self.LoopIbAmp.lo_fail

        # Scale for final selection
        self.e_wrap = self.sel_brk_hdwe.scale_select(ib_noa_hdwe, self.e_wrap_m, self.e_wrap_n)
        self.e_wrap_filt = self.sel_brk_hdwe.scale_select(ib_noa_hdwe, self.e_wrap_m_filt, self.e_wrap_n_filt)
        self.e_wrap_rate = self.sel_brk_hdwe.scale_select(ib_noa_hdwe, self.e_wrap_m_rate, self.e_wrap_n_rate)


# noinspection PyPep8Naming
class BatterySim(Battery):
    """Extend Battery class to make a model"""

    def __init__(self, OPT=None, SN=None, q_cap_rated=Battery.NOM_UNIT_CAP*3600,
                 t_rated=Battery.RATED_TEMP, temp_rlim=0.017, scale=1.,
                 Tb_f=Battery.NOMINAL_TB, tweak_test=False, mod_code=0, vsat_add=0.):
        Battery.__init__(self, OPT=OPT, q_cap_rated=q_cap_rated, t_rated=t_rated, temp_rlim=temp_rlim, Tb_f=Tb_f,
                         tweak_test=tweak_test, dvoc=OPT.add_voc_sim, mod_code=mod_code, scale_cap=scale, mon=False,
                         str_='ver_s', vsat_add=vsat_add)
        self.chemistry = Chemistry(mod_code=mod_code, dvoc=OPT.add_voc_sim, unit=OPT.unit)
        self.chemistry.assign_all_mod(mod_code, unit=OPT.unit)
        self.lut_voc = None
        self.sat_ib_max = 0.  # Current cutback to be applied to modeled ib output, A
        # self.sat_ib_null = 0.1*Battery.NOM_UNIT_CAP  # Current cutback value for voc=vsat, A
        self.sat_ib_null = 0.  # Current cutback value for soc=1, A
        # self.sat_cutback_gain = 4.8  # Gain to retard ib when voc exceeds vsat, dimensionless
        self.sat_cutback_gain = 1000.*Battery.sp_cutback_gain_slr  # Gain to retard ib when soc approaches 1, dimensionless
        self.model_cutback = False  # Indicate current being limited on saturation cutback, T = cutback limited
        self.model_saturated = False  # Indicator of maximal cutback, T = cutback saturated
        self.ib_sat = 0.5  # Threshold to declare saturation.  This regeneratively slows down charging so if too
        # small takes too long, A
        self.s_cap = scale  # Rated capacity scalar
        if scale is not None:
            self.apply_cap_scale(scale)
        self.hys = Hysteresis(scale=OPT.slr_hys_sim*Battery.ap_hys_scale, dv_hys=OPT.sim_run.dv_hys_s[0], scale_cap=OPT.slr_hys_cap_sim, slr_cap_chg=OPT.slr_cap_chg,
                              slr_cap_dis=OPT.slr_cap_dis, slr_hys_chg=OPT.slr_hys_chg, slr_hys_dis=OPT.slr_hys_dis, chem=self.chem,
                              chemistry=self.chemistry)  # Battery hysteresis model - drift of voc
        self.tweak_test = tweak_test
        self.voc = 0.  # Charging voltage, V
        self.ChargeTransfer = LagExp(dt=Battery.EKF_NOM_DT, tau=self.chemistry.tau_ct,
                                     max_=Battery.NOM_UNIT_CAP*scale, min_=-Battery.NOM_UNIT_CAP*scale)
        self.d_delta_q = 0.  # Charging rate, Coulombs/sec
        self.ib_charge = 0.  # Charge current, A
        self.saved_s = SavedS('ver_s')  # for plots and prints
        self.ib_fut = 0.  # Future value of limited current, A
        self.reset_temp_past = self.model_saturated
        self.dt_past = 0.
        self.dt_s = 0.
        self.chm_s = 0.
        self.qcrs_s = 0.
        self.qcap_s = 0.
        self.bms_off_s = 0
        # self.Tb_s = 0.
        # self.Tb_f_s = 0.
        self.vsat_s = 0.
        self.voc_s = 0.
        self.voc_stat_s = 0.
        self.dv_dyn_s = 0.
        self.d_delta_q_s = 0.
        self.delta_q_s = 0.
        self.ib_dyn_r = True
        self.dv_hys_s = 0.
        self.ib_charge_s = 0.
        self.ib_dyn_s = 0.
        self.ib_in_s = 0.
        self.ib_s = 0.
        self.ioc_s = 0.
        self.sat_s = 0.
        self.soc_s = 0.
        self.vb_s = 0.
        self.ib_dyn_T_s = 0.
        self.ib_dyn_lstate_s = 0.
        self.ib_dyn_rstate_s = 0.
        self.ib_dyn_tau_s = 0.
        self.tau_hys_s = 0.
        self.vb_s = 0.
        self.q_s = 0.
        self.ib_fut_s = 0.
        self.reset_s = 0.
        self.tau_s = 0.
        self.tau_hys_s = 0.
        if SN is not None:
            self.Tb = SN.mon_run.Tb_f[0]
            self.dv_dyn = SN.dv_dyn_s[0]
            self.ib_in = SN.sim_run.ib_in_s[0]
            self.d_delta_q = SN.d_delta_q_s_init
            self.delta_q = SN.delta_q_s_init
            self.ib = SN.ib_s_init
            self.ib_fut = SN.ib_fut_s_init
            self.ib_charge = SN.ib_charge_s_init
            self.ioc = SN.ioc_s_init
            if SN.run_type == 'HistSim':
                self.vb = SN.mon_run.vb_f[0]
            else:
                self.vb = SN.mon_run.vb[0]
            self.voc = SN.voc_s_init
            self.ib_dyn = SN.ib_dyn_s[0]
            self.soc = SN.mon_run.soc_s[0]

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "BatterySim:\n"
        s += "  sat_ib_max =      {:7.3f}  // Current cutback to be applied to modeled ib output, A\n".\
            format(self.sat_ib_max)
        s += "  ib_null    =      {:7.3f}  // Current cutback value for voc=vsat, A\n".\
            format(self.sat_ib_null)
        s += "  sat_cutback_gain = {:6.2f}  // Gain to retard ib when voc exceeds vsat, dimensionless\n".\
            format(self.sat_cutback_gain)
        s += "  model_cutback =         {:d}  // Indicate that modeled current being limited on" \
             " saturation cutback, T = cutback limited\n".format(self.model_cutback)
        s += "  model_saturated =       {:f}  // Indicator of maximal cutback, T = cutback saturated\n".\
            format(self.model_saturated)
        s += "  ib_sat =          {:7.3f}  // Threshold to declare saturation.  This regeneratively slows" \
             " down charging so if too\n".format(self.ib_sat)
        s += "  ib_in  =          {:7.3f}  // Saved value of current input, A\n".format(self.ib_in)
        s += "  ib     =          {:7.3f}  // Open circuit current into posts, A\n".format(self.ib)
        s += "  ib_fut =          {:7.3f}  // Future value of limited current, A\n".format(self.ib_fut)
        s += "  voc     =         {:7.3f}  // Open circuit voltage, V\n".format(self.voc)
        s += "  voc_stat=         {:7.3f}  // Static, table lookup value of voc before applying hysteresis, V\n".\
            format(self.voc_stat)
        s += "  mod     =               {:f}  // Modeling\n".format(self.mod)
        s += "  \n  "
        s += self.hys.__str__(prefix + 'BatterySim:')
        s += "  \n  "
        s += Battery.__str__(self, prefix + 'BatterySim:')
        return s

    # BatterySim::calculate()
    def calculate(self, chem, Tb, Tb_f, vb, ib, dt, reset, calc_ekf, dt_ekf, SN, OPT,
                  q_capacity=None, rp=None, soc=None, saturated_init=None, reset_ekf=None, i=None,
                  i_ekf=None):
        self.reset = reset
        if self.chm != chem:
            self.chemistry.assign_all_mod(chem, self.unit)
            self.chm = chem

        self.Tb = Tb
        self.dt_past = self.dt
        self.dt = dt
        self.ib_in = ib
        if self.reset and SN.sim_run.bms_off_s[0]:
            self.ib_fut = 0.
        self.ib = max(min(self.ib_fut, Battery.IMAX_NUM), -Battery.IMAX_NUM)
        self.mod = rp.modeling
        soc_lim = max(min(soc, 1.), -0.2)  # dag 9/3/2022

        # VOC-OCV model
        self.voc_stat, self.dv_dsoc = self.calc_soc_voc(soc + Battery.D_SOC_S, self.Tb_f)
        self.voc_stat += Battery.ap_dv_voc_soc
        # slightly beyond but don't windup
        self.voc_stat = min(self.voc_stat + (soc - soc_lim) * self.dv_dsoc, self.vsat * 1.2)

        # Hysteresis model
        self.hys.calculate_hys(ib, self.soc, self.chm)
        init_low = self.bms_off or (self.soc < (self.soc_min + Battery.HYS_SOC_MIN_MARG) and
                                    self.ib > Battery.HYS_IB_THR)
        self.dv_hys, self.tau_hys = self.hys.update(self.dt, init_high=self.model_saturated, init_low=init_low, e_wrap=0.,
                                                    chem=self.chm)
        self.voc = self.voc_stat + self.dv_hys
        self.voc_soc = self.voc_stat
        self.ioc = self.hys.ioc

        # Battery management system (bms)   I believe bms can see only vb but using this for a model causes
        # lots of chatter as it shuts off, restores vb due to loss of dynamic current, then repeats shutoff.
        # Using voc_ is not better because change in dv_hys_ causes the same effect.   So using nice quiet
        # voc_stat_ for ease of simulation, not accuracy.
        if not self.bms_off:
            self.voltage_low = self.voc_stat < self.chemistry.vb_down_sim
        else:
            self.voltage_low = self.voc_stat < self.chemistry.vb_rising_sim
        bms_charging = self.ib_in > Battery.IB_MIN_UP
        self.bms_off = (self.Tb_f < self.chemistry.low_t) or (self.voltage_low and not rp.tweak_test)
        ib_charge_fut = self.ib_in
        if self.bms_off and self.mod and not bms_charging:
            ib_charge_fut = 0.
        if self.bms_off and self.voltage_low:
            self.ib = 0.
        self.ib_lag = self.IbLag.calculate_tau(self.ib, self.reset, self.dt, self.chemistry.ib_lag_tau)
        # Charge transfer dynamics
        self.ib_dyn = self.ChargeTransfer.calculate_tau_seeded(self.ib, SN.ib_dyn_s[G.i], self.reset, self.dt,
                                                               self.chemistry.tau_ct)
        self.ib_dyn_r = self.ChargeTransfer.reset
        self.ib_dyn_T = self.ChargeTransfer.dt
        self.ib_dyn_rstate = self.ChargeTransfer.rstate
        self.ib_dyn_lstate = self.ChargeTransfer.state
        self.vb = self.voc + self.ib_dyn*self.chemistry.r_ct + self.ib*self.chemistry.r_0
        if self.bms_off:
            if Battery.ap_dc_dc_on:
                self.vb = Battery.VB_DC_DC
            else:
                self.vb = 0.
        self.dv_dyn = self.vb - self.voc

        # Saturation logic, both full and empty
        self.vsat = sat_voc(self.Tb_f, self.chemistry.rated_temp, self.chemistry.nom_vsat, self.chemistry.dvoc_dt,
                            vsat_add=Battery.sp_vsat_add)
        self.sat_ib_max = (self.sat_ib_null + (1 - self.soc - Battery.ap_ds_voc_soc) * self.sat_cutback_gain
                           * Battery.sp_cutback_gain_slr)
        if rp.tweak_test or (not rp.modeling_ib):
            self.sat_ib_max = ib_charge_fut
        self.ib_fut = min(ib_charge_fut, self.sat_ib_max)  # the feedback of self.ib
        # self.ib_charge = ib_charge_fut# same time plane as volt calcs.  (This prevents sat logic from working)
        self.ib_charge = self.ib_fut  # same time plane as volt calcs
        # empty  **** don't know why this was here.  cannot bms_off_ empty because that causes weird interaction with bms logic and also doesn't make sense to have a different empty cutoff when modeling.  If there is a need for an empty cutoff, should be based on voltage not current.  So removing for now.  Can revisit if needed.
        # if self.mod > 0.:
        #     if (self.q <= 0.) & (self.ib_charge < 0.):
        #         # print("q", self.q, "empty")
        #         self.ib_charge = 0.  # empty
        self.model_cutback = (self.voc_stat > self.vsat) & (self.ib_fut == self.sat_ib_max)
        self.model_saturated = self.model_cutback & (self.ib_fut < self.ib_sat)
        if self.reset and SN.mon_run.saturated[0] is not None:
            self.model_saturated = SN.mon_run.saturated[0]
        self.sat = self.model_saturated

        return self.vb

    def count_coulombs(self, OPT, SN, chem, reset_temp, tb_f, charge_curr, sat,
                       saturated, mon_sat=None, rp=None):
        # BatterySim
        """Coulomb counter based on true=actual capacity
        Internal resistance of battery is a loss
        Inputs:
            dt              Integration step, s
            tb_f            Battery temperature, deg C  (filtered usually to reduce electrical noise artifacts)
            charge_curr     Charge, A
            sat             Indicator that battery is saturated (VOC>threshold(temp)), T/F
            use_mon_soc     Command to drive integrator with input mon_soc
            SN.soc_s        Auxiliary integrator setting, fraction soc
        Outputs:
            soc     State of charge, fraction (0-1.5)
        """
        if self.chm != chem:
            self.chemistry.assign_all_mod(chem, self.unit)
            self.chm = chem
        self.ib_charge = charge_curr
        self.Tb_f = tb_f
        self.d_delta_q = self.ib_charge * self.dt
        if self.ib_charge > 0.:
            self.d_delta_q *= self.chemistry.coul_eff

        # Rate limit temperature.  When modeling, initialize to no change
        self.Tb_f_rate = SN.Tb_f_rate_past

        # Saturation and re - init.Goal is to set q_capacity and hold it so remember last saturation status
        if OPT.use_mon_soc or not bool(SN.mon_run.mvb[G.i]):
            if mon_sat or self.reset_temp_past:
                self.apply_delta_q_brief(SN.delta_q_s[G.i])
        elif self.model_saturated and reset_temp:
            self.delta_q = 0.

        # one pass flag
        self.resetting = False

        # Integration can go to - 20 %
        self.q_capacity = self.calculate_capacity(self.Tb_f)
        if (rp.modeling_ib and not self.reset_temp_past) or (not rp.modeling_ib and not reset_temp):
            self.delta_q += self.d_delta_q
            self.delta_q = max(min( self.delta_q, 0.), -self.q_capacity * 1.2)
        self.q = self.q_capacity + self.delta_q

        # Normalize
        if self.q_capacity != 0. and self.q_capacity is not None:
            self.soc = self.q / self.q_capacity
        self.soc_min = self.chemistry.lut_min_soc.interp(self.Tb_f)
        self.q_min = self.soc_min * self.q_capacity

        self.reset_temp_past = reset_temp
        return self.soc

    def save(self, time, dt):  # BatterySim
        self.time = time
        self.dt = dt
        # Append all parameters to
        self.append_to(self.saved)

    def save_s(self, time):
        self.time = time
        self.dt_s = self.dt
        self.chm_s = self.chm
        self.qcrs_s = self.q_cap_rated_scaled
        self.qcap_s = self.q_capacity
        self.bms_off_s = self.bms_off
        self.Tb_s = self.Tb
        self.Tb_f_s = self.Tb_f
        self.vsat_s = self.vsat
        self.voc_s = self.voc
        self.voc_stat_s = self.voc_stat
        self.dv_dyn_s = self.dv_dyn
        self.d_delta_q_s = self.d_delta_q
        self.delta_q_s = self.delta_q
        self.dv_hys_s = self.dv_hys
        self.ib_charge_s = self.ib_charge
        self.ib_dyn_s = self.ib_dyn
        self.ib_in_s = self.ib_in
        self.ib_s = self.ib
        self.ioc_s = self.ioc
        self.sat_s = self.sat
        self.soc_s = self.soc
        self.vb_s = self.vb
        self.ib_dyn_T_s = self.ib_dyn_T
        self.ib_dyn_lstate_s = self.ib_dyn_lstate
        self.ib_dyn_rstate_s = self.ib_dyn_rstate
        self.ib_dyn_tau_s = self.chemistry.tau_ct
        self.tau_hys_s = self.tau_hys
        self.vb_s = self.vb
        self.q_s = self.q
        self.ib_fut_s = self.ib_fut
        self.reset_s = self.reset
        self.tau_s = self.tau_hys
        self.tau_hys_s = self.tau_hys
        # Append all parameters to
        self.append_to(self.saved_s)


# Other functions
def is_sat(tb_f, rated_temp, voc, soc, nom_vsat, dvoc_dt, low_t, vsat_add=0.):
    vsat = sat_voc(tb_f, rated_temp, nom_vsat, dvoc_dt, vsat_add=vsat_add)
    return tb_f > low_t and (voc >= vsat or soc >= Battery.MXEPS)


def sat_voc(tb_f, rated_temp, vsat, dvoc_dt, vsat_add=0.):
    return vsat + (tb_f-rated_temp)*dvoc_dt + vsat_add


# noinspection PyPep8Naming
class Diff:
    """Compare predicted voltage to actual and track toward zero to eliminate biases """

    def __init__(self, dt=0.1):
        self.reset = True
        self.dt = dt
        self.ib_lo_limited_hi = False
        self.ib_lo_limited_lo = False
        self.ib_diff = 0.
        self.LoHi = TFDelay(dt=self.dt, in_=False, t_true=Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                            t_false=Battery.IB_LO_ACTIVE_RES*Battery.cp_ts)
        self.LoLo = TFDelay(dt=self.dt, in_=False, t_true=Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                            t_false=Battery.IB_LO_ACTIVE_RES*Battery.cp_ts)

    # Update the loop
    # needs to be called twice with reset=True to initialize properly
    def calculate(self, reset=True, dt=None, ib_amp=None, ib_noa=None):
        self.reset = reset
        self.dt = dt

        ib_amp_hi = ib_amp >= Battery.HDWE_IB_HI_LO_AMP_HI
        self.ib_lo_limited_hi = self.LoHi.calculate(in_=ib_amp_hi, t_true=Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                                                    t_false=Battery.IB_LO_ACTIVE_RES*Battery.cp_ts,
                                                    dt=self.dt, reset=self.reset)  # non-latching
        ib_amp_lo = ib_amp <= Battery.HDWE_IB_HI_LO_AMP_LO
        self.ib_lo_limited_lo = self.LoLo.calculate(in_=ib_amp_lo, t_true=Battery.IB_LO_ACTIVE_SET*Battery.cp_ts,
                                                    t_false=Battery.IB_LO_ACTIVE_RES*Battery.cp_ts,
                                                    dt=self.dt, reset=self.reset)  # non-latching

        # Match C++ Fault::ib_logic(): disable when both sensors simultaneously at same limit
        ib_noa_hi = ib_noa >= Battery.HDWE_IB_HI_LO_NOA_HI
        ib_noa_lo = ib_noa <= Battery.HDWE_IB_HI_LO_NOA_LO
        disable_amp_fault = (ib_amp_hi and ib_noa_hi) or (ib_amp_lo and ib_noa_lo)

        self.ib_diff = ib_amp - ib_noa
        if disable_amp_fault:
            pass  # both sensors pegged together: keep raw diff (lgv)
        elif self.ib_lo_limited_hi:
            self.ib_diff = max(0., self.ib_diff)
        elif self.ib_lo_limited_lo:
            self.ib_diff = min(0., self.ib_diff)

        return self.ib_diff


# noinspection PyPep8Naming
class Looparound:
    """Compare predicted voltage to actual and track toward zero to eliminate biases """

    def __init__(self, Mon_, wrap_hi_volt=0., wrap_lo_volt=0., max_err=None, name=''):
        self.Mon = Mon_
        self.reset = True
        self.dt = 0.
        self.dt_past = 0.
        self.dv_dyn = 0.
        self.e_wrap = 0.
        self.e_wrap_filt = 0.
        self.e_wrap_rate = 0.
        self.ib_dyn = 0.
        self.wrap_hi_volt = wrap_hi_volt
        self.wrap_lo_volt = wrap_lo_volt
        self.e_wrap_trim = 0.
        self.e_wrap_trimmed = 0.
        self.hi_fail = False
        self.hi_fault = False
        self.lo_fail = False
        self.lo_fault = False
        self.chem = Mon_.chemistry
        self.ChargeTransfer = LagExp(dt=Battery.EKF_NOM_DT, max_=Battery.NOM_UNIT_CAP*self.Mon.scale_cap,
                                     min_=-Battery.NOM_UNIT_CAP*self.Mon.scale_cap, tau=self.chem.tau_ct)
        self.ewhi_thr = 0.
        self.ewhi_thr_base = 0.
        self.ewlo_thr = 0.
        self.ewlo_thr_base = 0.
        self.ib = 0.
        self.ib_past = 0.
        self.ib_past2 = 0.
        self.Trim = TustinIntegrator(dt=2., min_=-max_err*10., max_=max_err*10.)
        self.vb = 0.
        self.voc = 0.
        self.voc_soc = 0.
        self.WrapErrFilt = LagExp(dt=2., min_=-max_err, max_=max_err, tau=Battery.WRAP_ERR_FILT)
        self.WrapHi = TFDelay(dt=2., in_=False, t_true=Battery.WRAP_HI_SET, t_false=Battery.WRAP_HI_RES)
        self.WrapLo = TFDelay(dt=2., in_=False, t_true=Battery.WRAP_LO_SET, t_false=Battery.WRAP_LO_RES)
        self.name = name

    # Update the loop
    # needs to be called twice with reset=True to initialize properly
    def calculate(self, reset=True, rp=None, ib=0., loop_gain=0., dt=None, ewsat_slr=1., ewmin_slr=1.,
                  ib_dyn_init=0., e_wrap_filt_init=0., e_wrap_trim_init=0., freeze=False):
        frozen = 1. - float(freeze)
        self.reset = reset
        self.dt = dt
        self.ib = ib
        if rp.modeling_vb:
            self.vb = self.Mon.vb_past
        else:
            self.vb = self.Mon.vb
        self.voc_soc = self.Mon.voc_soc
        if rp.modeling_ib:
            dt_into_ct = self.dt_past
            dt_into_wrap = self.dt_past
            ib_into_ct = self.ib_past2
        else:
            dt_into_ct = self.dt
            dt_into_wrap = self.dt
            ib_into_ct = self.ib_past

        self.ib_dyn = self.ChargeTransfer.calculate_tau_seeded(ib_into_ct, ib_dyn_init, self.reset, dt_into_ct,
                                                               self.chem.tau_ct, text=self.name)
        # print(f"{reset=} {ib=} {self.ib=} {self.ib_past=} {self.ChargeTransfer.rstate=}")
        self.dv_dyn = (self.ib_dyn * self.chem.r_ct + ib_into_ct * self.chem.r_0)
        self.voc = self.vb - self.dv_dyn
        self.e_wrap = self.voc_soc - self.voc

        # Trimmer using past values
        trim_rate_lim = max(min(self.e_wrap_filt * loop_gain, Battery.MAX_TRIM_RATE),
                            -Battery.MAX_TRIM_RATE)
        self.e_wrap_trim = -self.Trim.calculate_lim(in_=trim_rate_lim*frozen, dt=min(dt_into_wrap,
                                                    Battery.F_MAX_T_WRAP),
                                                    reset=self.reset, init_value = -e_wrap_trim_init,
                                                    max_=-self.ewlo_thr_base * Battery.EWLO_TRM_SLR,
                                                    min_=-self.ewhi_thr_base * Battery.EWHI_TRM_SLR)
        self.e_wrap_trimmed = self.e_wrap + self.e_wrap_trim
        e_wrap_filt_rate = 1e300
        if freeze:
            e_wrap_filt_rate = 0.
        self.e_wrap_filt = self.WrapErrFilt.calculate_seeded(in_=self.e_wrap_trimmed, _out_init=e_wrap_filt_init,
                                                             reset=self.reset, dt=dt_into_wrap, text=self.name,
                                                             rmin=-e_wrap_filt_rate, rmax=e_wrap_filt_rate)
        self.e_wrap_rate = self.WrapErrFilt.rate

        # Thresholds. Scalars are calculated by Flt->wrap_scalars()
        self.ewhi_thr_base = self.wrap_hi_volt * Battery.ap_ewhi_slr
        self.ewhi_thr = self.ewhi_thr_base * ewsat_slr * ewmin_slr
        print(f'{self.wrap_hi_volt=} {Battery.ap_ewhi_slr=} {self.ewhi_thr_base=} {ewsat_slr=} {ewmin_slr=}')
        self.ewlo_thr_base = self.wrap_lo_volt * Battery.ap_ewlo_slr
        self.ewlo_thr = self.ewlo_thr_base * ewsat_slr * ewmin_slr

        # sat logic screens out voc jump when ib>0 when saturated
        # wrap_hi and wrap_lo don't latch because need them available to check next ib sensor selection for dual ib sensor
        # wrap_vb latches because vb is single sensor  faultAssign( (e_wrap_filt_ >= ewhi_thr_ && !Mon->sat()), WRAP_HI_FLT);

        self.hi_fault = self.e_wrap_filt >= self.ewhi_thr
        self.hi_fail = self.WrapHi.calculate(in_=self.hi_fault, t_true=Battery.WRAP_HI_SET,
                                             t_false=Battery.WRAP_HI_RES,
                                             dt=self.dt_past, reset=self.reset)  # non-latching
        self.lo_fault = self.e_wrap_filt <= self.ewlo_thr
        self.lo_fail = self.WrapLo.calculate(in_=self.lo_fault, t_true=Battery.WRAP_LO_SET,
                                             t_false=Battery.WRAP_LO_RES,
                                             dt=self.dt_past, reset=self.reset)  # non-latching
        self.ib_past2 = self.ib_past
        self.ib_past = self.ib
        self.dt_past = self.dt


class Saved:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self, str_=None):
        self.str = str_
        self.time_run_start = None
        self.time = []
        self.time_min = []
        self.time_day = []
        self.time_t = []
        self.reset_temp = []
        self.dt = []
        self.dt_temp = []
        self.chm = []
        self.qcrs = []
        self.bmso = []
        self.ib = []
        self.ib_in = []
        self.ib_charge = []
        self.ioc = []
        self.vb = []
        self.voc = []
        self.voc_soc = []
        self.voc_stat = []
        self.voc_stat_f = []
        self.dv_hys = []
        self.tau_hys = []
        self.dv_dyn = []
        self.ib_dyn = []
        self.ib_dyn_T = []
        self.ib_dyn_rstate = []
        self.ib_dyn_lstate = []
        self.voc_stat_f_rstate = []
        self.voc_stat_f_lstate = []
        self.voc_stat_f_a = []
        self.voc_stat_f_b = []
        self.voc_stat_f_c = []
        self.voc_stat_f_tau = []
        self.voc_stat_f_T = []
        self.soc = []
        self.soc_ekf = []
        self.voc = []
        self.Fx = []
        self.Bu = []
        self.P = []
        self.Q = []
        self.dt_eframe = []
        self.voc_stat_ekf = []
        self.R = []
        self.H = []
        self.S = []
        self.K = []
        self.hx = []
        self.u_ekf = []
        self.x_ekf = []
        self.y_ekf = []
        self.y_ekf_f = []
        self.z = []
        self.x_prior = []
        self.P_prior = []
        self.x_post = []
        self.P_post = []
        self.e_soc_ekf = []
        self.e_voc_ekf = []
        self.tb_f_for_hx = []
        self.x_for_hx = []
        self.ib = []  # Bank current, A
        self.vb = []  # Bank voltage, V
        self.Tb = []  # Bank temp, C
        self.Tb_f = []  # Filtered bank temp, C
        self.sat = []  # Indication that battery is saturated, T=saturated
        self.saturated = []  # Confirmation that battery is saturated, T=saturation confirmed
        self.sel = []  # Current source selection, 0=amp, 1=no amp
        self.mod_data = []  # Configuration control code, 0=all hardware, 7=all simulated, +8 tweak test
        self.Tb = []  # Battery bank temperature, deg C
        self.Tb_f = []  # Battery bank filtered temperature, deg C
        self.Tb_f_rate = []  # Temp rate, deg C / s
        self.Tb_rap = []  # Battery bank temperature, deg C
        self.Tb_f = []  # Battery bank filtered temperature, deg C
        self.Tb_f_rate_rap = []  # Temp rate, deg C / s
        self.vsat = []  # Monitor Bank saturation threshold at temperature, deg C
        self.dv_dyn = []  # Monitor Bank current induced back emf, V
        self.ib_dyn = []  # Monitor Bank current induced back emf before resistance multiply, A
        self.ib_dyn_T = []  # Monitor Bank Randles update time, A
        self.ib_dyn_rstate = []  # Monitor Randles current state, A
        self.ib_dyn_lstate = []  # Monitor Randles current state, A
        self.voc_stat = []  # Monitor Static bank open circuit voltage, V
        self.voc = []  # Monitor Static bank open circuit voltage, V
        self.voc_ekf = []  # Monitor bank solved static open circuit voltage, V
        self.y_ekf = []  # Monitor single battery solver error, V
        self.y_ekf_f = []  # Filtered EKF y residual value, V
        self.soc_s = []  # Simulated state of charge, fraction
        self.soc_ekf = []  # Solved state of charge, fraction
        # self.soc = []  # Coulomb Counter fraction of saturation charge (q_capacity_) available (0-1)
        self.d_delta_q = []  # Charging rate, Coulombs/sec
        self.ib_charge = []  # Charging current, A
        self.q = []  # Present charge available to use, except q_min_, C
        self.delta_q = []  # Charge change since saturated, C
        self.d_delta_q = []  # Charge change since saturated, C
        self.q_capacity = []  # Saturation charge at temperature, C
        self.bms_off = []  # Voltage low without faults, battery management system has shut off battery
        self.reset = []  # Reset flag used for initialization
        self.reset_ekf = []  # Reset flag used for initialization
        self.e_wrap = []  # Verification of wrap calculation, V
        self.e_wrap_filt = []  # Verification of filtered wrap calculation, V
        self.ib_dyn_m = []  # Verification of wrap calculation, A
        self.dv_dyn_m = []  # Verification of wrap calculation, V
        self.e_wrap_m = []  # Verification of wrap calculation, V
        self.e_wrap_m_filt = []  # Verification of filtered wrap calculation, V
        self.e_wrap_m_trim = []  # Verification of filtered wrap calculation, V
        self.ib_dyn_n = []  # Verification of wrap calculation, A
        self.dv_dyn_n = []  # Verification of wrap calculation, V
        self.e_wrap_n = []  # Verification of wrap calculation, V
        self.e_wrap_n_filt = []  # Verification of filtered wrap calculation, V
        self.e_wrap_n_trim = []  # Verification of filtered wrap calculation, V
        self.e_wrap_rate = []  # Verification of filtered wrap rate calculation, V/s
        self.ib_lag = []  # Lagged ib, A
        self.voc_soc_new = []  # New schedule values
        self.ib_amp = []
        self.ib_amp_model = []
        self.ib_amp_hdwe = []
        self.ib_noa = []
        self.ib_noa_model = []
        self.ib_noa_hdwe = []
        self.ewmhi_thr = []
        self.ewmlo_thr = []
        self.ewnhi_thr = []
        self.ewnlo_thr = []
        self.Tb_rstate = []
        self.Tb_lstate = []
        self.Tb_hdwe = []
        self.Tb_hdwe_f = []
        self.Tb_model_f = []
        self.Tb_hdwe_f_rate = []
        self.Tb_model_f_rate = []
        self.e_wrap_m_reset = []
        self.reset_kf = []
        self.iscn_f = []
        self.Tb_model = []
        self.vb_hdwe = []
        self.vb_hdwe_f = []
        self.vsat = []


class SavedS:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self, str_=None):
        self.str_ = str_
        self.time_run_start = None
        self.time = []
        self.unit = []  # text title
        self.c_time = []  # Control time, s
        self.dt = []
        self.chm_s = []
        self.qcrs_s = []
        self.qcap_s = []
        self.bms_off_s = []
        self.Tb_s = []
        self.Tb_f_s = []
        self.vsat_s = []
        self.voc_s = []
        self.voc_stat_s = []
        self.dv_dyn_s = []
        self.dv_hys_s = []
        self.tau_hys_s = []
        self.tau_s = []
        self.vb_s = []
        self.ib_s = []
        self.ib_dyn_s = []
        self.ib_dyn_T_s = []
        self.ib_dyn_rstate_s = []
        self.ib_dyn_lstate_s = []
        self.ib_in_s = []
        self.d_delta_q_s = []
        self.ib_charge_s = []
        self.ib_fut_s = []
        self.sat_s = []
        self.ddq_s = []
        self.delta_q_s = []
        self.q_s = []
        self.qcap_s = []
        self.soc_s = []
        self.reset_s = []
        self.ioc_s = []

    def __str__(self):
        s = "unit_m,c_time,Tb_s,vsat_s,voc_stat_s,dv_dyn_s,vb_s,ib_s,ib_dyn_s,sat_s,ddq_s,dq_s,q_s,qcap_s,soc_s,\
        reset_s,tau_s,\n"
        for i in range(len(self.time)):
            s += 'sim,'
            s += "{:13.3f},".format(self.time[i])
            s += "{:7.3f},".format(self.dt[i])
            s += "{:5.2f},".format(self.Tb_s[i])
            s += "{:8.3f},".format(self.vsat_s[i])
            s += "{:5.2f},".format(self.voc_stat_s[i])
            s += "{:5.2f},".format(self.dv_dyn_s[i])
            s += "{:5.2f},".format(self.vb_s[i])
            s += "{:8.3f},".format(self.ib_s[i])
            s += "{:8.3f},".format(self.ib_dyn_s[i])
            s += "{:8.3f},".format(self.ib_in_s[i])
            s += "{:8.3f},".format(self.ib_fut_s[i])
            s += "{:1.0f},".format(self.sat_s[i])
            s += "{:5.3f},".format(self.ddq_s[i])
            s += "{:5.3f},".format(self.delta_q_s[i])
            s += "{:5.3f},".format(self.qcap_s[i])
            s += "{:7.3f},".format(self.soc_s[i])
            s += "{:d},".format(self.reset_s[i])
            s += "{:7.3f},".format(self.tau_s[i])
            s += "\n"
        return s
