# Battery - general purpose battery class for modeling
# Copyright (C) 2021 Dave Gutz
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

"""Define a general purpose battery model including Randles' model and SoC-VOV model."""

import numpy as np
from EKF1x1 import EKF1x1
from Coulombs import Coulombs, Chemistry
from Hysteresis import Hysteresis
import matplotlib.pyplot as plt
from TFDelay import TFDelay
from myFilters import LagTustin, LagExp, General2Pole, RateLimit, SlidingDeadband, TustinIntegrator, RateLagExp
from Scale import ScaleSelector
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})
import Globals as G


class Retained:

    def __init__(self):
        self.cutback_gain_scalar = 1.
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


class Battery(Coulombs):
    import Globals as G
    # Battery constants
    NOM_UNIT_CAP = 108.4  # Nominal battery unit capacity.  (* 'Sc' or '*BS'/'*BP'), Ah
    NOM_SYS_VOLT = 12.  # Nominal system output, V, at which the reported amps are used (12)
    mxeps_bb = 1.05  # Numerical maximum of coefficient model with scaled soc
    TCHARGE_DISPLAY_DEADBAND = 0.1  # Inside this +/- deadband, charge time is displayed '---', A
    DF2 = 0.70  # Threshold to reset Coulomb Counter if different from ekf, fraction (0.05)
    EKF_CONV = 2e-3  # EKF tracking error indicating convergence, V (1e-3)
    EKF_T_CONV = 30.  # EKF set convergence test time, sec (30.)
    EKF_T_RESET = (EKF_T_CONV / 2.)  # EKF reset test time, sec ('up 1, down 2')
    EKF_NOM_DT = 0.1  # EKF nominal update time, s (initialization; actual value varies)
    TAU_Y_FILT = 5.  # EKF y-filter time constant, sec (5.)
    MIN_Y_FILT = -0.5  # EKF y-filter minimum, V (-0.5)
    MAX_Y_FILT = 0.5  # EKF y-filter maximum, V (0.5)
    WN_Y_FILT = 0.1  # EKF y-filter-2 natural frequency, r/s (0.1)
    ZETA_Y_FILT = 0.9  # EKF y-filter-2 damping factor (0.9)
    TMAX_FILT = 3.  # Maximum y-filter-2 sample time, s (3.)
    EKF_Q_SD_NORM = 0.0015  # Standard deviation of normal EKF process uncertainty, V (0.0015)
    EKF_R_SD_NORM = 0.5  # Standard deviation of normal EKF state uncertainty, fraction (0-1) (0.5)
    IMAX_NUM = 100000.  # Overflow protection since ib past value used
    HYS_SOC_MIN_MARG = 0.15  # Add to soc_min to set thr for detecting low endpoint condition for reset of hysteresis
    HYS_IB_THR = 1.  # Ignore reset if opposite situation exists
    HYS_SCALE = 1.  # Used to disable hysteresis from sim on the app
    IB_MIN_UP = 0.2  # Min up charge current for come alive, BMS logic, and fault
    cp_eframe_mult = 20  # Run EKF 20 times slower than Coulomb Counter
    VB_DC_DC = 13.5  # Estimated dc-dc charger, V
    HDB_VBATT = 0.05  # Half deadband to filter vb, V (0.05)
    WRAP_ERR_FILT = 4.  # Wrap error filter time constant, s (4)
    MAX_WRAP_ERR_FILT = 10.  # Anti-windup wrap error filter, V (10)
    IB_ABS_MAX_AMP = 12.  # Hard range limit of bank sensor electrically impossible (=1.65 * SHUNT_GAIN * SHUNT_AMP_R1 / SHUNT_AMP_R2 *1.05) but saw -11.48 A (12)
    IB_ABS_MAX_NOA = 78.5  # Hard range limit of sensor electrically impossible (=1.65 * SHUNT_GAIN * SHUNT_NOA_R1 / SHUNT_NOA_R2 *1.05) A (78.5)
    MAX_TRIM_RATE = 0.005  # Max allowable amp e_wraptrim rate, V/s (0.005)
    F_MAX_T_WRAP = 2.8  # Maximum update time of Wrap filter for stability at WRAP_ERR_FILT, s (2.8)
    D_SOC_S = 0.  # Bias on soc to voc-soc lookup to simulate error in estimation, esp cold battery near 0 C
    VB_OFF_BB = 10.  # BMS shutoff level, Battleborn, v (10)
    VB_OFF_CH = 11.  # BMS shutoff level, CHINS, v (11)
    AMP_WRAP_TRIM_GAIN = 0.015  # Amp looparound trim gain r/s (0.015)
    NOA_WRAP_TRIM_GAIN = 0.0  # Noa looparound trim gain r/s (0.0)
    WRAP_LO_S = 9.  # Wrap low failure set time, sec (9) // 9 is legacy must be quicker than SAT test
    WRAP_LO_R = (WRAP_LO_S/2.)  # Wrap low failure reset time, sec ('up 1, down 2')
    WRAP_HI_S = WRAP_LO_S  # Wrap high failure set time, sec (WRAP_LO_S)
    WRAP_HI_R = (WRAP_HI_S/2.)  # Wrap high failure reset time, sec ('up 1, down 2')
    WRAP_HI_AMP = 3.2  # Wrap high voltage threshold amplified, A(3.2)
    WRAP_LO_AMP = -4.  # Wrap high voltage threshold amplified, A (-4)
    WRAP_HI_NOA = 32  # Wrap high voltage threshold non-amplified, A(32)
    WRAP_LO_NOA = -40.  # Wrap high voltage threshold non-amplified, A (-40)
    HDWE_IB_HI_LO = 1.  # Type of selection logic philosophy. Only True is implemented and debugged now
    HDWE_IB_HI_LO_NOA_LO = -11. # Fully NOA unit discharge transition, A (-11, soc4p2)
    HDWE_IB_HI_LO_AMP_LO = -10. # Fully NOA unit discharge transition, A (-10, soc4p2)
    HDWE_IB_HI_LO_AMP_HI = 10.  # Fully NOA unit charge transition, A (10, soc4p2)
    HDWE_IB_HI_LO_NOA_HI = 11.  # Fully NOA unit charge transition, A (11, soc4p2)
    WRAP_SOC_HI_OFF = 0.97  # Disable e_wrap_hi when saturated (0.97)
    WRAP_SOC_LO_OFF_REL = 0.2  # Disable e_wrap when near empty (soc lo for high Tb where soc_min=.2, voltage cutback, 0.2)
    WRAP_SOC_LO_OFF_ABS = 0.35  # Disable e_wrap when near empty (soc lo any Tb, 0.35)
    WRAP_HI_SAT_MARG = 0.2  # Wrap voltage margin to saturation, V (0.2)
    WRAP_MOD_C_RATE = 0.02  # Moderate charge rate threshold to engage wrap threshold (0.02 to prevent trip near saturation .05 too large)
    WRAP_SOC_MOD_OFF = 0.85  # Disable e_wrap_lo when nearing saturated and moderate C_rate(0.85)
    WRAP_SOC_HI_SLR = 1000.  # Huge to disable e_wrap (1000)
    WRAP_SOC_LO_SLR = 60.  # Large to disable e_wrap (60. for startup)
    VOC_STAT_FILT = 120.  # Clean up noise (120)
    VB_MIN = 2.  # Signal selection hard fault threshold, V (0.  < 2. < 10 bms shutoff, reads ~3 without power when off)
    VB_MAX = 17.  # Signal selection hard fault threshold, V (17. < VB_CONV_GAIN*4095)
    TB_MAX = 60.  # Signal selection hard fault threshold 2wire only, C (60.)
    TB_MIN = -40.  # Signal selection hard fault threshold 2wire only, C (-40.)
    TB_FILT = 120.  # Temperature filter lag, s (120)
    T_RLIM = 0.00085  # Temperature sensor rate limit to minimize jumps in Coulomb counting, deg C/s (0.00085 allows 0.05 deg for 1 minute)
    DISAB_LO_SET = 0.4  # Disable lo=amp wrap fault set persistence, s (0.4)
    DISAB_LO_RESET = 0.8  # Disable lo=amp wrap fault reset persistence, s (0.8)
    SHUNT_AMP_GAIN = 1.  # hdwe gain, A/V
    CURR_BIAS_AMP = 0.  # hdwe bias, A
    SHUNT_NOA_GAIN = 1.  # hdwe gain, A/V
    CURR_BIAS_NOA = 0.  # hdwe bias, A
    NS = 1  # Number serial batteries in bank, for converting raw Ib,Vb to ib, vb per battery unit
    NP = 1  # Number parallel batteries in bank, for converting raw Ib,Vb to ib, vb per battery unit
    # KF_Q_STD = 0.015  # Shunt KF process uncertainty
    # KF_R_STD = 0.001  # Shunt KF state uncertainty
    KF_Q_STD = 0.0003  # Shunt KF process uncertainty
    KF_R_STD = 0.1000  # Shunt KF state uncertainty
    dc_dc_on = 0.  # Truck charging

    # """Nominal battery bank capacity, Ah(100).Accounts for internal losses.This is
    #                         what gets delivered, e.g. Wshunt / NOM_SYS_VOLT.  Also varies 0.2 - 0.4 C currents
    #                         or 20 - 40 A for a 100 Ah battery"""

    # Battery model:  Randles' dynamics, SOC-VOC model

    """Nominal battery bank capacity, Ah(100).Accounts for internal losses.This is
                            what gets delivered, e.g.Wshunt / NOM_SYS_VOLT.  Also varies 0.2 - 0.4 C currents
                            or 20 - 40 A for a 100 Ah battery"""

    def __init__(self, OPT=None, q_cap_rated=NOM_UNIT_CAP*3600, temp_rlim=0.017, t_rated=25., tb_f=25., tweak_test=False,
                 dvoc=0., mod_code=0,
                 scale_cap=1., mon=None):
        """ Default values from Taborelli & Onori, 2013, State of Charge Estimation Using Extended Kalman Filters for
        Battery Management System.   Battery equations from LiFePO4 BattleBorn.xlsx and 'Generalized SOC-OCV Model Zhang
        etal.pdf.'  SOC-OCV curve fit './Battery State/BattleBorn Rev1.xls:Model Fit' using solver with min slope
        constraint >=0.02 V/soc.  m and n using Zhang values.   Had to scale soc because  actual capacity > NOM_BAT_CAP
        so equation error when soc<=0 to match data.    See Battery.h
        """
        # Parents
        Coulombs.__init__(self, OPT, q_cap_rated,  q_cap_rated, t_rated, temp_rlim, tweak_test, dvoc=dvoc)

        # Defaults
        self.chem = mod_code
        self.nz = None
        self.q = 0  # Charge, C
        self.voc = Battery.NOM_SYS_VOLT  # Model open circuit voltage, V
        self.voc_stat = self.voc  # Static model open circuit voltage from charge process, V
        self.voc_stat_f = self.voc_stat
        self.dv_dyn = 0.  # Model current induced back emf, V
        self.ib_dyn = 0.  # Model current induced back emf before resistance multiply, A
        self.ib_dyn_rstate = 0.  # Model current rate, A
        self.ib_dyn_lstate = 0.  # Model current rate, A
        self.vb = Battery.NOM_SYS_VOLT  # Battery voltage at post, V
        self.ib = 0.  # Current into battery post, A
        self.ib_in = 0.  # Current into calculate, A
        self.ib_charge = 0.  # Current into count_coulombs, A
        self.ioc = 0  # Current into battery process accounting for hysteresis, A
        self.dv_dsoc = 0.  # Slope of soc-voc curve, V/%
        self.tcharge = 0.  # Charging time to 100%, hr
        self.sr = 1  # Resistance scalar
        self.vsat = self.chemistry.nom_vsat
        # range 0 - 50 C, V/deg C
        self.dt = 0  # Update time, s
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
        self.Tb = tb_f
        self.Tb_f = tb_f
        self.Tb_f_rate = None
        self.saved = Saved()  # for plots and prints
        self.dv_hys = 0.  # Placeholder so BatterySim can be plotted
        self.tau_hys = 0.  # Placeholder so BatterySim can be plotted
        self.dv_dyn = 0.  # Placeholder so BatterySim can be plotted
        self.ib_dyn = 0.  # Placeholder so BatterySim can be plotted
        self.ib_dyn_rstate = 0.  # Placeholder so BatterySim can be plotted
        self.ib_dyn_lstate = 0.  # Placeholder so BatterySim can be plotted
        self.bms_off = False
        self.mod = 7
        self.sel = 0
        self.tweak_test = tweak_test
        self.ib_lag = 0.
        self.IbLag = LagExp(1., 1., -100., 100.)  # Lag to be run on sat to produce ib_lag.  T and tau set at run time
        self.voc_soc = None
        self.voc_soc_new = 0.
        self.scale_cap = scale_cap
        self.Tb_rstate = None
        self.Tb_state = None
        self.Tb_hdwe_filt = None
        self.Tb_hdwe_filt_rate = None
        self.Tb_model_filt_rate = None
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

    def calc_soc_voc(self, soc, tb_f):
        """SOC-OCV curve fit method per Zhang, etal """
        dv_dsoc = self.calc_h_jacobian(soc, tb_f)
        voc = self.chemistry.lookup_voc(soc, tb_f)
        # print("soc=", soc, "tb_f=", tb_f, "dvoc=", self.dvoc, "voc=", voc)
        return voc, dv_dsoc

    def calculate(self, chem, vb, ib, dt, reset, calc_ekf, dt_ekf, SN, OPT,
                  q_capacity=None, rp=None, reset_ekf=None, soc=None, sat_init=None):
        # Battery
        raise NotImplementedError

    def look_hys(self, dv, soc):
        raise NotImplementedError


class BatteryMonitor(Battery, EKF1x1):
    """Extend Battery class to make a monitor"""
    def __init__(self, OPT=None, SN=None, q_cap_rated=Battery.NOM_UNIT_CAP*3600, t_rated=25., temp_rlim=0.017, scale=1.,
                 tb_f=25., tweak_test=False, dvoc=0., mod_code=0):
        if hasattr(OPT, 'slr_res_0'):
            ref = OPT.mon_run
        else:
            pass
        q_cap_rated_scaled = q_cap_rated * scale
        Battery.__init__(self, OPT=OPT, q_cap_rated=q_cap_rated_scaled, t_rated=t_rated, temp_rlim=temp_rlim, tb_f=tb_f,
                         tweak_test=tweak_test, dvoc=dvoc, mod_code=mod_code, scale_cap=scale, mon=True)

        """ Default values from Taborelli & Onori, 2013, State of Charge Estimation Using Extended Kalman Filters for
        Battery Management System.   Battery equations from LiFePO4 BattleBorn.xlsx and 'Generalized SOC-OCV Model Zhang
        etal.pdf.'  SOC-OCV curve fit './Battery State/BattleBorn Rev1.xls:Model Fit' using solver with min slope
        constraint >=0.02 V/soc.  m and n using Zhang values.   Had to scale soc because  actual capacity > NOM_BAT_CAP
        so equations error when soc<=0 to match data.    See Battery.h
        """
        # Parents
        EKF1x1.__init__(self)
        self.tcharge_ekf = 0.  # Charging time to 100% from ekf, hr
        self.voc = 0.  # Charging voltage, V
        self.soc_ekf = 0.  # Filtered state of charge from ekf (0-1)
        self.q_ekf = 0  # Filtered charge calculated by ekf, C
        self.amp_hrs_remaining_ekf = 0  # Discharge amp*time left if drain to q_ekf=0, A-h
        self.amp_hrs_remaining_wt = 0  # Discharge amp*time left if drain soc_wt_ to 0, A-h
        self.e_soc_ekf = 0.  # analysis parameter
        self.e_voc_ekf = 0.  # analysis parameter
        self.Q = Battery.EKF_Q_SD_NORM * Battery.EKF_Q_SD_NORM  # EKF process uncertainty
        self.R = Battery.EKF_R_SD_NORM * Battery.EKF_R_SD_NORM  # EKF state uncertainty
        self.soc_s = 0.  # Model information
        self.EKF_converged = TFDelay(False, Battery.EKF_T_CONV, Battery.EKF_T_RESET, Battery.EKF_NOM_DT)
        self.voc_stat_filt = LagExp(self.EKF_NOM_DT, self.VOC_STAT_FILT, self.VB_MIN, self.VB_MAX)  # Lag to be run on sat to produce ib_lag.  T and tau set at run time
        self.y_filt_lag = LagTustin(0.1, Battery.TAU_Y_FILT, Battery.MIN_Y_FILT, Battery.MAX_Y_FILT)
        self.WrapErrFilt = LagTustin(0.1, Battery.WRAP_ERR_FILT, -Battery.MAX_WRAP_ERR_FILT, Battery.MAX_WRAP_ERR_FILT)
        self.y_filt = 0.
        self.y_filt_2Ord = General2Pole(0.1, Battery.WN_Y_FILT, Battery.ZETA_Y_FILT, Battery.MIN_Y_FILT,
                                        Battery.MAX_Y_FILT)
        self.y_filt2 = 0.
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
        self.ib_dyn_rstate = 0.
        self.ib_dyn_lstate = 0.
        self.voc_stat_f_rstate = 0.
        self.voc_stat_f_lstate = 0.
        self.voc_stat_f_tau = 0.
        self.voc_stat_f_T = 0.
        self.voc_ekf = 0.
        self.eframe = 0
        if OPT is not None:
            self.eframe_mult = OPT.eframe_mult
            self.dt_eframe = self.dt*self.eframe_mult
        self.sdb_voc = SlidingDeadband(Battery.HDB_VBATT)
        self.e_wrap = 0.
        self.e_wrap_filt = 0.
        self.e_wrap_rate = 0.
        self.ib_past = 0.
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
        self.disable_amp_fault_per = False
        self.DisabAmpFltPer = TFDelay(False, Battery.DISAB_LO_SET, Battery.DISAB_LO_RESET, 0.1)
        self.LoopIbAmp = Looparound(Mon_=self, wrap_hi_amp=Battery.WRAP_HI_AMP, wrap_lo_amp=Battery.WRAP_LO_AMP,
                                    max_err=Battery.MAX_WRAP_ERR_FILT/(Battery.IB_ABS_MAX_NOA/Battery.IB_ABS_MAX_AMP),
                                    name="Amp")
        self.LoopIbNoa = Looparound(Mon_=self, wrap_hi_amp=Battery.WRAP_HI_NOA, wrap_lo_amp=Battery.WRAP_LO_NOA,
                                    max_err=Battery.MAX_WRAP_ERR_FILT, name="Noa")
        self.ewnhi_thr = None
        self.ewnlo_thr = None
        self.ewmhi_thr = None
        self.e_wrap_m_reset = True
        self.ewmlo_thr = None
        self.reset_ekf = None
        self.voc_stat_ekf = 0.
        self.dt_temp = None
        self.reset_temp = True
        self.Tb_rap = None
        self.Tb_model = None
        self.Tb_f_rap = None
        self.Tb_f_rate_rap = None
        self.dt_temp = 0.
        self.sel_brk_hdwe = ScaleSelector(Battery.HDWE_IB_HI_LO_NOA_LO, Battery.HDWE_IB_HI_LO_AMP_LO,
                                          Battery.HDWE_IB_HI_LO_AMP_HI, Battery.HDWE_IB_HI_LO_NOA_HI)
        self.reset_kf = False
        self.iscn_f = 0.
        if SN is not None:
            self.Tb_hdwe = SN.Tb_hdwe_init
            self.Tb_hdwe_filt =SN.Tb_hdwe_filt_init
            self.Tb_hdwe_filt_rate = SN.Tb_hdwe_filt_rate_init
            self.Tb_model_filt =SN.Tb_model_filt_init
            self.Tb_model_filt_rate = SN.Tb_model_filt_rate_init
            self.e_wrap = SN.e_wrap_init
            self.e_wrap_filt = SN.e_wrap_filt_init
            self.ib_amp_lo = False
            self.ib_noa_lo = False
            self.ib_amp_hi = False
            self.ib_noa_hi = False
            self.e_wrap_m = SN.e_wrap_m_init
            self.e_wrap_m_filt = SN.e_wrap_m_filt_init
            self.e_wrap_m_trim = SN.e_wrap_m_trim_init
            self.e_wrap_n = SN.e_wrap_n_init
            self.e_wrap_n_filt = SN.e_wrap_n_filt_init
            self.e_wrap_n_trim = SN.e_wrap_n_trim_init
            self.voc_soc = SN.voc_soc_init
            self.voc_stat = self.voc_soc - self.e_wrap
            self.Tb = SN.Tb0
            self.Tb_f = SN.Tb_f_init
            self.Tb_f_rate = SN.Tb_f_rate_init
            self.Tb_rap = SN.Tb_rap_init
            self.Tb_model = SN.Tb_model_init
            self.Tb_f_rap = SN.Tb_f_rap_init
            self.Tb_f_rate_rap = SN.Tb_f_rate_rap_init
            self.ib = SN.ib_init
            self.ib_dyn = SN.ib_dyn[0]
            self.ib_charge = SN.ib_charge_init
            self.ib_charge_ekf = self.ib_charge
            self.vb = SN.vb_init
            self.soc = SN.soc_init
            self.reset = True
            self.sat = SN.sat_init
            self.reset_ekf = True
            self.init_soc_ekf(ref,  0, 0)
            self.voc_ekf = SN.hx_init
            self.x = SN.x_init
            self.x_prior = SN.x_prior_init
            self.soc_ekf = SN.soc_ekf_init
            self.z_ekf = SN.z_init
            self.z = SN.z_init
            self.disable_amp_fault_per = SN.mon_run.disable_amp_fault_per[0]

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
    def calculate(self, chem, vb, ib, dt, reset, calc_ekf, dt_ekf, SN, OPT,
                  q_capacity=None, rp=None, soc=None, sat_init=None, reset_ekf=None, i=None):
        self.ib_amp_hdwe = SN.mon_run.ibmh[G.i]
        self.ib_amp_model = SN.mon_run.ibmm[G.i]
        self.ib_noa_hdwe = SN.mon_run.ibnh[G.i]
        self.ib_noa_model = SN.mon_run.ibnm[G.i]
        if hasattr(SN.mon_run, 'vb_model'):
            self.vb_model = SN.mon_run.vb_model[G.i]
        if hasattr(SN.mon_run, 'vb_hdwe'):
            self.vb_hdwe = SN.mon_run.vb_hdwe[G.i]
        if hasattr(SN.mon_run, 'vb_hdwe_f'):
            self.vb_hdwe_f = SN.mon_run.vb_hdwe_f[G.i]
        if rp.modeling_ib:
            self.ib_amp = self.ib_amp_model
            self.ib_noa = self.ib_noa_model
            self.ib_amp_pst = SN.mon_run.ibmm[max(G.i-1, 0)]
            self.ib_noa_pst = SN.mon_run.ibnm[max(G.i-1, 0)]
        else:
            self.ib_amp = self.ib_amp_hdwe
            self.ib_noa = self.ib_noa_hdwe
            self.ib_amp_pst = SN.mon_run.ibmh[max(G.i - 1, 0)]
            self.ib_noa_pst = SN.mon_run.ibnh[max(G.i - 1, 0)]
        # self.ib_hdwe = self.ib_noa_hdwe
        self.ib_hdwe = SN.mon_run.ib_h[G.i]
        if self.chm != chem:
            self.chemistry.assign_all_mod(chem, unit=self.unit)
            self.chm = chem

        self.vsat = self.chemistry.nom_vsat + (self.Tb_f - 25.) * self.chemistry.dvoc_dt
        self.dt = dt
        self.ib_in = ib
        if OPT.IB_CHARGE_NOA:
            self.ib_in = self.ib_noa
        self.mod = rp.modeling
        # Overflow protection since ib past value used
        self.ib = max(min(self.ib_in, Battery.IMAX_NUM), -Battery.IMAX_NUM)

        # Wrap logic
        self.wrap(reset=reset, modeling_ib=rp.modeling_ib, ib_noa_hdwe=self.ib_noa_hdwe, SN=SN, ib_amp=self.ib_amp,
                  ib_noa=self.ib_noa, ib_amp_pst=self.ib_amp_pst, ib_noa_pst=self.ib_noa_pst, rp=rp)

        # Reversionary model
        self.vb_model_rev = self.voc_soc + self.dv_dyn + self.dv_hys

        # Table lookup
        self.voc_soc, self.dv_dsoc = self.calc_soc_voc(self.soc, self.Tb_f_rap)

        # Battery management system model (uses past value bms_off and voc_stat)
        if not self.bms_off:
            self.voltage_low = self.voc_stat < self.chemistry.vb_down
        else:
            self.voltage_low = self.voc_stat < self.chemistry.vb_rising
        bms_charging = self.ib > Battery.IB_MIN_UP
        if reset and SN.mon_run.bms_off[0] is not None:
            self.bms_off = SN.mon_run.bms_off[0]
        else:
            self.bms_off = (self.Tb_f <= self.chemistry.low_t) or (self.voltage_low and not rp.tweak_test)  # KISS
        self.ib_charge = self.ib
        self.ib_charge_ekf = self.ib_charge
        if self.bms_off and not bms_charging:
            self.ib_charge = 0.
        if self.bms_off and self.voltage_low:
            self.ib = 0.
        self.ib_lag = self.IbLag.calculate_tau(self.ib, reset, self.dt, self.chemistry.ib_lag_tau)
        if reset:
            self.ib_past = self.ib

        # Dynamic emf
        if rp.modeling_ib:
            ib_dc = self.ib_past
        else:
            # ib_dc = self.ib
            ib_dc = self.ib_past
        self.vb = vb
        self.ib_dyn = self.ChargeTransfer.calculate_tau_seeded(ib_dc, SN.ib_dyn[G.i], reset, self.dt,
                                                               self.chemistry.tau_ct)
        self.ib_dyn_rstate = self.ChargeTransfer.rstate
        self.ib_dyn_lstate = self.ChargeTransfer.state
        self.voc = self.vb - (self.ib_dyn*self.chemistry.r_ct + ib_dc*self.chemistry.r_0)
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
            if not self.reset_ekf:
                pass
            # print(f"{reset_ekf=} {self.soc_ekf} {self.x_ekf=} {self.voc_stat_ekf=}")
            self.voc_stat_ekf = self.voc_stat
            self.dt_eframe = dt_ekf
            ddq_dt = self.ib_charge_ekf
            if ddq_dt > 0. and not self.tweak_test:
                ddq_dt *= self.chemistry.coul_eff
            # ddq_dt -= self.chemistry.dqdt*self.q_capacity*temp_rate  7/29/2025 to agree with c++ (noisy)
            self.Q = Battery.EKF_Q_SD_NORM**2  # override
            self.R = Battery.EKF_R_SD_NORM**2  # override
            self.voc_stat_f =\
                self.voc_stat_filt.calculate_tau_seeded(self.voc_stat_ekf, SN.z_init, self.reset_ekf, self.dt_eframe,
                                                        self.VOC_STAT_FILT)
            self.voc_stat_f_rstate = self.voc_stat_filt.rstate
            self.voc_stat_f_lstate = self.voc_stat_filt.state
            self.voc_stat_f_tau = self.voc_stat_filt.tau
            self.voc_stat_f_T = self.voc_stat_filt.dt
            self.predict_ekf(u=ddq_dt, reset=self.reset_ekf, freeze=self.bms_off)  # u = d(q)/dt
            self.update_ekf(z=self.voc_stat_f, x_min=0., x_max=1.)  # z = voc, voc_filtered = hx
            self.soc_ekf = self.x  # x = Vsoc (0-1 ideal capacitor voltage) proxy for soc
            self.q_ekf = self.soc_ekf * self.q_capacity
            self.y_filt = self.y_filt_lag.calculate(in_=self.y_ekf, dt=min(self.dt_eframe, Battery.EKF_T_RESET),
                                                    reset=self.reset_ekf)
            self.y_filt2 = self.y_filt_2Ord.calculate(in_=self.y_ekf, dt=min(self.dt_eframe, Battery.TMAX_FILT),
                                                      reset=self.reset_ekf)
            # EKF convergence
            conv = abs(self.y_filt) < Battery.EKF_CONV
            self.EKF_converged.calculate(conv, Battery.EKF_T_CONV, Battery.EKF_T_RESET,
                                         min(self.dt_eframe, Battery.EKF_T_RESET), self.reset_ekf)
            # print(f"{reset_ekf=} {self.soc_ekf} {self.x=} {self.voc_stat_ekf=}")
        self.eframe += 1
        if self.reset_ekf or self.eframe >= self.eframe_mult:  # '>=' ensures reset with changes on the fly
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

    # def count_coulombs(self, dt=0., reset=False, tb_f=25., charge_curr=0., sat=True):
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
        x_lim = max(min(self.x, 1.), 0.)
        self.x_for_hx = x_lim
        self.tb_f_for_hx = self.Tb_f_rap
        self.hx, self.dv_dsoc = self.calc_soc_voc(x_lim, tb_f=self.Tb_f_rap)
        # Jacobian of measurement function
        self.H = self.dv_dsoc
        return self.hx, self.H, self.tb_f_for_hx, self.x_for_hx

    def init_soc_ekf(self, mr, i, i_ekf):
        self.soc_ekf = mr.soc_ekf[i]
        self.y_ekf = mr.y_ekf[i]
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

        self.x = mr.soc_ekf[i]

        if hasattr(mr, 'x_prior'):
            self.x_prior = mr.x_prior[i_ekf]
        else:
            self.x_prior = self.x

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

    def save(self, time, dt, soc_run, voc_run, iscn_f):  # BatteryMonitor
        self.saved.time.append(time)
        self.saved.time_min.append(time / 60.)
        self.saved.time_day.append(time / 3600. / 24.)
        self.saved.dt_temp.append(self.dt_temp)
        self.saved.reset_temp.append(self.reset_temp)
        self.saved.chm.append(self.chm)
        self.saved.qcrs.append(self.q_cap_rated_scaled)
        self.saved.delta_q.append(self.delta_q)
        self.saved.d_delta_q.append(self.delta_q)
        self.saved.dt.append(dt)
        self.saved.ib.append(self.ib)
        self.saved.ib_in.append(self.ib_in)
        self.saved.ib_charge.append(self.ib_charge)
        self.saved.ioc.append(self.ioc)
        self.saved.vb.append(self.vb)
        self.saved.dv_hys.append(self.dv_hys)
        self.saved.tau_hys.append(self.tau_hys)
        self.saved.dv_dyn.append(self.dv_dyn)
        self.saved.ib_dyn.append(self.ib_dyn)
        self.saved.ib_dyn_rstate.append(self.ib_dyn_rstate)
        self.saved.ib_dyn_lstate.append(self.ib_dyn_lstate)
        self.saved.voc_stat_f_rstate.append(self.voc_stat_f_rstate)
        self.saved.voc_stat_f_lstate.append(self.voc_stat_f_lstate)
        self.saved.voc_stat_f_tau.append(self.voc_stat_f_tau)
        self.saved.voc_stat_f_T.append(self.voc_stat_f_T)
        self.saved.voc.append(self.voc)
        self.saved.voc_soc.append(self.voc_soc)
        self.saved.voc_stat.append(self.voc_stat)
        self.saved.voc_stat_f.append(self.voc_stat_f)
        self.saved.soc.append(self.soc)
        self.saved.soc_ekf.append(self.soc_ekf)
        self.saved.Fx.append(self.Fx)
        self.saved.Bu.append(self.Bu)
        self.saved.P.append(self.P)
        self.saved.Q.append(self.Q)
        self.saved.dt_eframe.append(self.dt_eframe)
        self.saved.voc_stat_ekf.append(self.voc_stat_ekf)
        self.saved.R.append(self.R)
        self.saved.H.append(self.H)
        self.saved.S.append(self.S)
        self.saved.K.append(self.K)
        self.saved.hx.append(self.hx)
        self.saved.u_ekf.append(self.u_ekf)
        self.saved.x_ekf.append(self.x)
        self.saved.y_ekf.append(self.y_ekf)
        self.saved.y_filt.append(self.y_filt)
        self.saved.y_filt2.append(self.y_filt2)
        self.saved.z_ekf.append(self.z_ekf)
        self.saved.x_prior.append(self.x_prior)
        self.saved.P_prior.append(self.P_prior)
        self.saved.x_post.append(self.x_post)
        self.saved.P_post.append(self.P_post)
        if abs(soc_run) < 1e-6:
            soc_run = 1e-6
        self.e_soc_ekf = (self.soc_ekf - soc_run) / soc_run
        self.e_voc_ekf = (self.voc - voc_run) / voc_run
        self.saved.e_soc_ekf.append(self.e_soc_ekf)
        self.saved.e_voc_ekf.append(self.e_voc_ekf)
        self.saved.tb_f_for_hx.append(self.tb_f_for_hx)
        self.saved.x_for_hx.append(self.x_for_hx)
        self.saved.Tb.append(self.Tb)
        self.saved.Tb_f.append(self.Tb_f)
        self.saved.Tb_model.append(self.Tb_model)
        self.saved.Tb_f_rate.append(self.Tb_f_rate)
        self.saved.Tb_rap.append(self.Tb_rap)
        self.saved.Tb_f_rap.append(self.Tb_f_rap)
        self.saved.Tb_f_rate_rap.append(self.Tb_f_rate_rap)
        self.saved.vsat.append(self.vsat)
        self.saved.voc_ekf.append(self.voc_ekf)
        self.saved.sat.append(int(self.sat))
        self.saved.sel.append(self.sel)
        self.saved.mod_data.append(self.mod)
        self.saved.soc_s.append(self.soc_s)
        self.saved.bms_off.append(self.bms_off)
        self.saved.reset.append(self.reset)
        self.saved.reset_ekf.append(self.reset_ekf)
        self.saved.e_wrap.append(self.e_wrap)
        self.saved.e_wrap_filt.append(self.e_wrap_filt)
        self.saved.ib_dyn_m.append(self.LoopIbAmp.ib_dyn)
        self.saved.dv_dyn_m.append(self.LoopIbAmp.dv_dyn)
        self.saved.e_wrap_m.append(self.e_wrap_m)
        self.saved.e_wrap_m_filt.append(self.e_wrap_m_filt)
        self.saved.e_wrap_m_trim.append(self.e_wrap_m_trim)
        self.saved.ib_dyn_n.append(self.LoopIbNoa.ib_dyn)
        self.saved.dv_dyn_n.append(self.LoopIbNoa.dv_dyn)
        self.saved.e_wrap_n.append(self.e_wrap_n)
        self.saved.e_wrap_n_filt.append(self.e_wrap_n_filt)
        self.saved.e_wrap_n_trim.append(self.e_wrap_n_trim)
        self.saved.e_wrap_rate.append(self.e_wrap_rate)
        self.saved.ib_amp.append(self.ib_amp)
        self.saved.ib_amp_model.append(self.ib_amp_model)
        self.saved.ib_noa.append(self.ib_noa)
        self.saved.ib_noa_model.append(self.ib_noa_model)
        self.saved.ib_lag.append(self.ib_lag)
        self.saved.voc_soc_new.append(self.voc_soc_new)
        self.saved.ewmhi_thr.append(self.ewmhi_thr)
        self.saved.e_wrap_m_reset.append(self.e_wrap_m_reset)
        self.saved.ewmlo_thr.append(self.ewmlo_thr)
        self.saved.ewnhi_thr.append(self.ewnhi_thr)
        self.saved.ewnlo_thr.append(self.ewnlo_thr)
        self.saved.q.append(self.q)
        self.saved.q_capacity.append(self.q_capacity)
        self.saved.Tb_rstate.append(self.Tb_rstate)
        self.saved.Tb_lstate.append(self.Tb_state)
        self.saved.Tb_hdwe.append(self.Tb_hdwe)
        self.saved.Tb_hdwe_filt.append(self.Tb_hdwe_filt)
        self.saved.Tb_model_filt.append(self.Tb_model_filt)
        self.saved.Tb_hdwe_filt_rate.append(self.Tb_hdwe_filt_rate)
        self.saved.reset_kf.append(self.reset_kf)
        self.saved.iscn_f.append(iscn_f)
        self.saved.vb_hdwe.append(self.vb_hdwe)
        self.saved.vb_hdwe_f.append(self.vb_hdwe_f)

    def wrap(self, reset=True, modeling_ib=None, ib_noa_hdwe=0., SN=None, ib_amp=0., ib_noa=0.,
             ib_amp_pst=None, ib_noa_pst=None, rp=None):
        """Wrap logic"""

        # e_wrap scalars normally calculated in Sensors
        if self.soc >= Battery.WRAP_SOC_HI_OFF:
            ewsat_slr = Battery.WRAP_SOC_HI_SLR
            ewmin_slr = 1.
        elif self.soc <= max(self.soc_min+Battery.WRAP_SOC_LO_OFF_REL, Battery.WRAP_SOC_LO_OFF_ABS):
            ewsat_slr = 1.
            ewmin_slr = Battery.WRAP_SOC_LO_SLR
        elif (self.voc_soc > (self.vsat - Battery.WRAP_HI_SAT_MARG) or
            (self.voc_stat > (self.vsat-Battery.WRAP_HI_SAT_MARG) and
             self.ib / Battery.NOM_UNIT_CAP > Battery.WRAP_MOD_C_RATE and
             self.soc > Battery.WRAP_SOC_MOD_OFF)):
            ewsat_slr = Battery.WRAP_SOC_HI_SLR
            ewmin_slr = 1.
        else:
            ewsat_slr = 1.
            ewmin_slr = 1.

        # Individual wrap logic
        if ib_noa is not None:
            if rp.modeling_ib:
                self.ib_noa = ib_noa
                self.ib_noa_pst = ib_noa_pst
            else:
                self.ib_noa = ib_noa
                self.ib_noa_pst = ib_noa_pst
            # print(f"{self.ib_noa=}", end='')
            self.LoopIbNoa.calculate(reset=reset, rp=rp, ib=self.ib_noa, loop_gain=Battery.NOA_WRAP_TRIM_GAIN,
                                     dt=min(self.dt, Battery.F_MAX_T_WRAP), ewmin_slr=ewmin_slr, ewsat_slr=ewsat_slr,
                                     ib_init = SN.LoopNoa.ib_init, ib_dyn_init=SN.LoopNoa.ib_dyn[G.i],
                                     e_wrap_filt_init = SN.e_wrap_n_filt_init, e_wrap_trim_init = SN.e_wrap_n_trim_init)
            self.e_wrap_n = self.LoopIbNoa.e_wrap
            self.e_wrap_n_filt = self.LoopIbNoa.e_wrap_filt
            self.e_wrap_n_rate = self.LoopIbNoa.e_wrap_rate
            self.e_wrap_n_trim = self.LoopIbNoa.e_wrap_trim
            self.ewnhi_thr = self.LoopIbNoa.ewhi_thr
            self.ewnlo_thr = self.LoopIbNoa.ewlo_thr
        if ib_amp is not None:
            if rp.modeling_ib:
                self.ib_amp = ib_amp
                self.ib_amp_pst = ib_amp_pst
                ib_m_init = SN.LoopAmp.ib[max(G.i-2, 0)]
                ib_dyn_m_init = SN.LoopAmp.ib_dyn[G.i]
            else:
                self.ib_amp = ib_amp
                self.ib_amp_pst = ib_amp_pst
                ib_m_init = SN.LoopAmp.ib[G.i]
                ib_dyn_m_init = SN.LoopAmp.ib_dyn[G.i]
            self.ib_amp_hi = self.ib_amp >= Battery.HDWE_IB_HI_LO_AMP_HI
            self.ib_amp_lo = self.ib_amp <= Battery.HDWE_IB_HI_LO_AMP_LO
            self.ib_noa_hi = self.ib_noa >= Battery.HDWE_IB_HI_LO_NOA_HI
            self.ib_noa_lo = self.ib_noa <= Battery.HDWE_IB_HI_LO_NOA_LO
            if self.ib_noa_lo:
                pass
            self.disable_amp_fault = (self.ib_amp_hi and self.ib_noa_hi) or (self.ib_amp_lo and self.ib_noa_lo)
            self.disable_amp_fault_per = self.DisabAmpFltPer.calculate(self.disable_amp_fault, Battery.DISAB_LO_SET,
                                                                       Battery.DISAB_LO_RESET, self.dt, reset)
            # print(f"ib_amp_hi/lo, ib_noa_hi/lo = {self.ib_amp_hi} {self.ib_amp_lo} {self.ib_noa_hi} {self.ib_noa_lo}")
            self.e_wrap_m_reset = reset or self.disable_amp_fault
            if rp.modeling_ib:
                dt_local = self.dt
            else:
                dt_local = self.dt_past
            self.LoopIbAmp.calculate(reset=self.e_wrap_m_reset, rp=rp, ib=self.ib_amp,
                                     loop_gain=Battery.AMP_WRAP_TRIM_GAIN, dt=min(dt_local, Battery.F_MAX_T_WRAP),
                                     ewmin_slr=ewmin_slr, ewsat_slr=ewsat_slr, ib_init=ib_m_init,
                                     ib_dyn_init=ib_dyn_m_init, e_wrap_filt_init=SN.e_wrap_m_filt_init,
                                     e_wrap_trim_init=SN.e_wrap_m_trim_init)
            self.ewmhi_thr = self.LoopIbAmp.ewhi_thr
            self.ewmlo_thr = self.LoopIbAmp.ewlo_thr
            self.e_wrap_m = self.LoopIbAmp.e_wrap
            self.e_wrap_m_filt = self.LoopIbAmp.e_wrap_filt
            self.e_wrap_m_rate = self.LoopIbAmp.e_wrap_rate
            self.e_wrap_m_trim = self.LoopIbAmp.e_wrap_trim

        # Scale for final selection
        self.e_wrap = self.sel_brk_hdwe.scale_select(ib_noa_hdwe, self.e_wrap_m, self.e_wrap_n)
        self.e_wrap_filt = self.sel_brk_hdwe.scale_select(ib_noa_hdwe, self.e_wrap_m_filt, self.e_wrap_n_filt)
        self.e_wrap_rate = self.sel_brk_hdwe.scale_select(ib_noa_hdwe, self.e_wrap_m_rate, self.e_wrap_n_rate)


class BatterySim(Battery):
    """Extend Battery class to make a model"""

    def __init__(self, OPT=None, SN=None, q_cap_rated=Battery.NOM_UNIT_CAP*3600, t_rated=25., temp_rlim=0.017,
                 scale=1., tb_f=25., tweak_test=False, mod_code=0):
        Battery.__init__(self, OPT=OPT, q_cap_rated=q_cap_rated, t_rated=t_rated, temp_rlim=temp_rlim, tb_f=tb_f,
                         tweak_test=tweak_test, dvoc=OPT.add_voc_sim, mod_code=mod_code, scale_cap=scale, mon=False)
        self.chemistry = Chemistry(mod_code=mod_code, dvoc=OPT.add_voc_sim, unit=OPT.unit)
        self.chemistry.assign_all_mod(mod_code, unit=OPT.unit)
        self.lut_voc = None
        self.sat_ib_max = 0.  # Current cutback to be applied to modeled ib output, A
        # self.sat_ib_null = 0.1*Battery.NOM_UNIT_CAP  # Current cutback value for voc=vsat, A
        self.sat_ib_null = 0.  # Current cutback value for soc=1, A
        # self.sat_cutback_gain = 4.8  # Gain to retard ib when voc exceeds vsat, dimensionless
        self.sat_cutback_gain = 1000.*OPT.slr_cutback_gain  # Gain to retard ib when soc approaches 1, dimensionless
        self.add_s_voc_soc = OPT.add_s_voc_soc
        self.model_cutback = False  # Indicate current being limited on saturation cutback, T = cutback limited
        self.model_saturated = False  # Indicator of maximal cutback, T = cutback saturated
        self.ib_sat = 0.5  # Threshold to declare saturation.  This regeneratively slows down charging so if too
        # small takes too long, A
        self.s_cap = scale  # Rated capacity scalar
        if scale is not None:
            self.apply_cap_scale(scale)
        self.hys = Hysteresis(scale=OPT.slr_hys_sim*Battery.HYS_SCALE, dv_hys=OPT.mon_run.dv_hys[0], scale_cap=OPT.slr_hys_cap_sim, slr_cap_chg=OPT.slr_cap_chg,
                              slr_cap_dis=OPT.slr_cap_dis, slr_hys_chg=OPT.slr_hys_chg, slr_hys_dis=OPT.slr_hys_dis, chem=self.chem,
                              chemistry=self.chemistry)  # Battery hysteresis model - drift of voc
        self.tweak_test = tweak_test
        self.voc = 0.  # Charging voltage, V
        self.ChargeTransfer = LagExp(dt=Battery.EKF_NOM_DT, tau=self.chemistry.tau_ct,
                                     max_=Battery.NOM_UNIT_CAP*scale, min_=-Battery.NOM_UNIT_CAP*scale)
        self.d_delta_q = 0.  # Charging rate, Coulombs/sec
        self.ib_charge = 0.  # Charge current, A
        self.saved_s = SavedS()  # for plots and prints
        self.ib_fut = 0.  # Future value of limited current, A
        self.reset_temp_past = self.sat
        self.dt_past = 0.
        # self.q_eps = 0.  # tiny adjustment to charge to book-keep soc_s and delta_q_s to be the same as data stream
        if SN is not None:
            self.Tb = SN.Tb0
            self.dv_dyn = SN.dv_dyn_s_init
            self.ib_in = SN.ib_in_s_init
            self.d_delta_q = SN.d_delta_q_s_init
            self.delta_q = SN.delta_q_s_init
            self.ib = SN.ib_s_init
            self.ib_fut = SN.ib_fut_s_init
            self.ib_charge = SN.ib_charge_s_init
            self.ioc = SN.ioc_s_init
            self.vb = SN.vb_s_init
            self.voc = SN.voc_s_init
            self.ib_dyn = SN.ib_dyn_s_init
            self.soc = SN.soc_s_init

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
    def calculate(self, chem, vb, ib, dt, reset, calc_ekf, dt_ekf, SN, OPT,
                  q_capacity=None, rp=None, reset_ekf=None, soc=None, sat_init=None):
        self.reset = reset
        if self.chm != chem:
            self.chemistry.assign_all_mod(chem, self.unit)
            self.chm = chem

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
        # slightly beyond but don't windup
        self.voc_stat = min(self.voc_stat + (soc - soc_lim) * self.dv_dsoc, self.vsat * 1.2)

        # Hysteresis model
        self.hys.calculate_hys(ib, self.soc, self.chm)
        init_low = self.bms_off or (self.soc < (self.soc_min + Battery.HYS_SOC_MIN_MARG) and
                                    self.ib > Battery.HYS_IB_THR)
        self.dv_hys, self.tau_hys = self.hys.update(self.dt, init_high=self.sat, init_low=init_low, e_wrap=0.,
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
        self.ib_dyn = self.ChargeTransfer.calculate_tau_seeded(self.ib, SN.ib_dyn_s_init, self.reset, self.dt,
                                                               self.chemistry.tau_ct)
        self.ib_dyn_rstate = self.ChargeTransfer.rstate
        self.ib_dyn_lstate = self.ChargeTransfer.state
        self.vb = self.voc + self.ib_dyn*self.chemistry.r_ct + self.ib*self.chemistry.r_0
        if self.bms_off:
            if Battery.dc_dc_on:
                self.vb = Battery.VB_DC_DC
            else:
                self.vb = 0.
        self.dv_dyn = self.vb - self.voc

        # Saturation logic, both full and empty
        self.vsat = sat_voc(self.Tb_f, self.chemistry.rated_temp, self.chemistry.nom_vsat, self.chemistry.dvoc_dt)
        self.sat_ib_max = (self.sat_ib_null + (1 - self.soc - self.add_s_voc_soc) * self.sat_cutback_gain *
                           rp.cutback_gain_scalar)
        if rp.tweak_test or (not rp.modeling_ib):
            self.sat_ib_max = ib_charge_fut
        self.ib_fut = min(ib_charge_fut, self.sat_ib_max)  # the feedback of self.ib
        # self.ib_charge = ib_charge_fut# same time plane as volt calcs.  (This prevents sat logic from working)
        self.ib_charge = self.ib_fut  # same time plane as volt calcs
        if self.mod > 0.:
            if (self.q <= 0.) & (self.ib_charge < 0.):
                # print("q", self.q, "empty")
                self.ib_charge = 0.  # empty
        self.model_cutback = (self.voc_stat > self.vsat) & (self.ib_fut == self.sat_ib_max)
        self.model_saturated = self.model_cutback & (self.ib_fut < self.ib_sat)
        if self.reset and sat_init is not None:
            self.model_saturated = sat_init
            self.sat = sat_init
        self.sat = self.model_saturated

        return self.vb

    def count_coulombs(self, OPT, SN, chem, reset_temp, tb_f, charge_curr, sat, mon_sat=None):
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
        if not self.reset_temp_past:
            self.delta_q += self.d_delta_q
            self.delta_q = max(min( self.delta_q, 0.), -self.q_capacity * 1.2)
        self.q = self.q_capacity + self.delta_q

        # Normalize
        self.soc = self.q / self.q_capacity
        self.soc_min = self.chemistry.lut_min_soc.interp(self.Tb_f)
        self.q_min = self.soc_min * self.q_capacity

        self.reset_temp_past = reset_temp
        return self.soc

    def save(self, time, dt):  # BatterySim
        self.saved.time.append(time)
        self.saved.dt.append(dt)
        self.saved.ib.append(self.ib)
        self.saved.ib_in.append(self.ib_in)
        self.saved.ib_charge.append(self.ib_charge)
        self.saved.chm.append(self.chm)
        self.saved.bmso.append(self.bms_off)
        self.saved.ioc.append(self.ioc)
        self.saved.vb.append(self.vb)
        self.saved.dv_hys.append(self.dv_hys)
        self.saved.tau_hys.append(self.tau_hys)
        self.saved.dv_dyn.append(self.dv_dyn)
        self.saved.ib_dyn.append(self.ib_dyn)
        self.saved.ib_dyn_rstate.append(self.ib_dyn_rstate)
        self.saved.ib_dyn_lstate.append(self.ib_dyn_lstate)
        self.saved.voc.append(self.voc)
        self.saved.voc_stat.append(self.voc_stat)
        self.saved.soc.append(self.soc)
        self.saved.d_delta_q.append(self.d_delta_q)
        self.saved.Tb.append(self.Tb)
        self.saved.vsat.append(self.vsat)
        self.saved.sat.append(int(self.model_saturated))
        self.saved.delta_q.append(self.delta_q)
        self.saved.q.append(self.q)
        self.saved.q_capacity.append(self.q_capacity)
        self.saved.bms_off.append(self.bms_off)

    def save_s(self, time):
        self.saved_s.time.append(time)
        self.saved_s.chm_s.append(self.chm)
        self.saved_s.qcrs_s.append(self.q_cap_rated_scaled)
        self.saved_s.qcap_s.append(self.q_capacity)
        self.saved_s.bms_off_s.append(self.bms_off)
        self.saved_s.Tb_s.append(self.Tb)
        self.saved_s.Tb_f_s.append(self.Tb_f)
        self.saved_s.vsat_s.append(self.vsat)
        self.saved_s.voc_s.append(self.voc)
        self.saved_s.voc_stat_s.append(self.voc_stat)
        self.saved_s.dv_dyn_s.append(self.dv_dyn)
        self.saved_s.dv_hys_s.append(self.dv_hys)
        self.saved_s.ib_dyn_s.append(self.ib_dyn)
        self.saved_s.ib_dyn_rstate_s.append(self.ib_dyn_rstate)
        self.saved_s.ib_dyn_lstate_s.append(self.ib_dyn_lstate)
        self.saved_s.tau_hys_s.append(self.tau_hys)
        self.saved_s.vb_s.append(self.vb)
        self.saved_s.ib_s.append(self.ib)
        self.saved_s.ib_in_s.append(self.ib_in)
        self.saved_s.d_delta_q_s.append(self.d_delta_q)
        self.saved_s.ib_charge_s.append(self.ib_charge)
        self.saved_s.ib_fut_s.append(self.ib_fut)
        self.saved_s.sat_s.append(int(self.sat))
        self.saved_s.delta_q_s.append(self.delta_q)
        self.saved_s.q_s.append(self.q)
        self.saved_s.soc_s.append(self.soc)
        self.saved_s.reset_s.append(self.reset)
        self.saved_s.tau_s.append(self.tau_hys)


# Other functions
def is_sat(tb_f, rated_temp, voc, soc, nom_vsat, dvoc_dt, low_t):
    vsat = sat_voc(tb_f, rated_temp, nom_vsat, dvoc_dt)
    return tb_f > low_t and (voc >= vsat or soc >= Battery.mxeps_bb)


def sat_voc(tb_f, rated_temp, vsat, dvoc_dt):
    return vsat + (tb_f-rated_temp)*dvoc_dt


class Looparound:
    """Compare predicted voltage to actual and track toward zero to eliminate biases """

    def __init__(self, Mon_, wrap_hi_amp=0., wrap_lo_amp=0., max_err=None, name=''):
        self.Mon = Mon_
        self.reset = True
        self.dt = 0.
        self.dt_past = 0.
        self.dv_dyn = 0.
        self.e_wrap = 0.
        self.e_wrap_filt = 0.
        self.e_wrap_rate = 0.
        self.ib_dyn = 0.
        self.wrap_hi_amp = wrap_hi_amp
        self.wrap_lo_amp = wrap_lo_amp
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
        self.ewlo_thr = 0.
        self.ib = 0.
        self.ib_past = 0.
        self.ib_past2 = 0.
        self.Trim = TustinIntegrator(dt=2., min_=-max_err*10., max_=max_err*10.)
        self.vb = 0.
        self.voc = 0.
        self.voc_soc = 0.
        self.WrapErrFilt = LagTustin(dt=2., min_=-max_err, max_=max_err, tau=Battery.WRAP_ERR_FILT)
        self.WrapHi = TFDelay(dt=2., in_=False, t_true=Battery.WRAP_HI_S, t_false=Battery.WRAP_HI_R)
        self.WrapLo = TFDelay(dt=2., in_=False, t_true=Battery.WRAP_LO_S, t_false=Battery.WRAP_LO_R)
        self.name = name

    # Update the loop
    # needs to be called twice with reset=True to initialize properly
    def calculate(self, reset=True, rp=None, ib=0., loop_gain=0., dt=None, ewsat_slr=1., ewmin_slr=1.,
                  ib_init=0., ib_dyn_init=0., e_wrap_filt_init=0., e_wrap_trim_init=0.):
        self.reset = reset
        self.dt = dt
        self.ib = ib
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
        self.dv_dyn = (self.ib_dyn* self.chem.r_ct + ib_into_ct * self.chem.r_0)
        self.voc = self.vb - self.dv_dyn
        self.e_wrap = self.voc_soc - self.voc

        # Trimmer using past values
        trim_rate_lim = max(min(self.e_wrap_filt * loop_gain, Battery.MAX_TRIM_RATE), -Battery.MAX_TRIM_RATE)
        # e_wrap_trim_ = -Trim_->calculate(trim_rate_lim, min(Sen_->T, F_MAX_T_WRAP), reset_, trim_init);
        self.e_wrap_trim = -self.Trim.calculate(in_=trim_rate_lim, dt=min(dt_into_wrap, Battery.F_MAX_T_WRAP), reset=self.reset,
                                                init_value = -e_wrap_trim_init)
        self.e_wrap_trimmed = self.e_wrap + self.e_wrap_trim
        self.e_wrap_filt = self.WrapErrFilt.calculate_seeded(in_=self.e_wrap_trimmed, _out_init=e_wrap_filt_init,
                                                             reset=self.reset,
                                                             dt=min(dt_into_wrap, Battery.F_MAX_T_WRAP),
                                                             text=self.name)
        self.e_wrap_rate = self.WrapErrFilt.rate

        # Thresholds. Scalars are calculated by Flt->wrap_scalars()
        self.ewhi_thr = self.Mon.chemistry.r_ss * self.wrap_hi_amp * ewsat_slr * ewmin_slr
        self.ewlo_thr = self.Mon.chemistry.r_ss * self.wrap_lo_amp * ewsat_slr * ewmin_slr

        # sat logic screens out voc jump when ib>0 when saturated
        # wrap_hi and wrap_lo don't latch because need them available to check next ib sensor selection for dual ib sensor
        # wrap_vb latches because vb is single sensor  faultAssign( (e_wrap_filt_ >= ewhi_thr_ && !Mon->sat()), WRAP_HI_FLT);

        self.hi_fault = self.e_wrap_filt >= self.ewhi_thr
        self.hi_fail = self.WrapHi.calculate(in_=self.hi_fault, t_true=Battery.WRAP_HI_S, t_false=Battery.WRAP_HI_R,
                                             dt=self.dt_past, reset=self.reset)  # non-latching
        self.lo_fault = self.e_wrap_filt <= self.ewlo_thr
        self.lo_fail = self.WrapLo.calculate(in_=self.lo_fault, t_true=Battery.WRAP_LO_S, t_false=Battery.WRAP_LO_R,
                                             dt=self.dt_past, reset=self.reset)  # non-latching
        self.ib_past2 = self.ib_past
        self.ib_past = self.ib
        self.dt_past = self.dt


class Saved:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self):
        self.time_run = None
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
        self.y_filt = []
        self.y_filt2 = []
        self.z_ekf = []
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
        self.sat = []  # Indication that battery is saturated, T=saturated
        self.sel = []  # Current source selection, 0=amp, 1=no amp
        self.mod_data = []  # Configuration control code, 0=all hardware, 7=all simulated, +8 tweak test
        self.Tb = []  # Battery bank temperature, deg C
        self.Tb_f = []  # Battery bank filtered temperature, deg C
        self.Tb_f_rate = []  # Temp rate, deg C / s
        self.Tb_rap = []  # Battery bank temperature, deg C
        self.Tb_f_rap = []  # Battery bank filtered temperature, deg C
        self.Tb_f_rate_rap = []  # Temp rate, deg C / s
        self.vsat = []  # Monitor Bank saturation threshold at temperature, deg C
        self.dv_dyn = []  # Monitor Bank current induced back emf, V
        self.ib_dyn = []  # Monitor Bank current induced back emf before resistance multiply, A
        self.ib_dyn_rstate = []  # Monitor Bank current, A
        self.ib_dyn_lstate = []  # Monitor Bank current, A
        self.voc_stat = []  # Monitor Static bank open circuit voltage, V
        self.voc = []  # Monitor Static bank open circuit voltage, V
        self.voc_ekf = []  # Monitor bank solved static open circuit voltage, V
        self.y_ekf = []  # Monitor single battery solver error, V
        self.y_filt = []  # Filtered EKF y residual value, V
        self.y_filt2 = []  # Filtered EKF y residual value, V
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
        # self.e_wrap_trim = []  # Verification of filtered wrap calculation, V
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
        self.ib_noa = []
        self.ib_noa_model = []
        self.ewmhi_thr = []
        self.ewmlo_thr = []
        self.ewnhi_thr = []
        self.ewnlo_thr = []
        self.Tb_rstate = []
        self.Tb_lstate = []
        self.Tb_hdwe = []
        self.Tb_hdwe_filt = []
        self.Tb_model_filt = []
        self.Tb_hdwe_filt_rate = []
        self.Tb_model_filt_rate = []
        self.e_wrap_m_reset = []
        self.reset_kf = []
        self.iscn_f = []
        self.Tb_model = []
        self.vb_hdwe = []
        self.vb_hdwe_f = []


def overall_batt(mv, sv, filename,
                 mv1=None, sv1=None, suffix1=None, fig_files=None, plot_title=None, fig_list=None, suffix='',
                 use_time_day=False):
    if fig_files is None:
        fig_files = []

    if mv1 is None:
        if use_time_day:
            mv.time = mv.time_day - mv.time_day[0]
            sv.time = sv.time_day - sv.time_day[0]

        plt.figure()  # Batt 1
        fig_list += 1
        plt.subplot(321)
        plt.title(plot_title + ' B 1')
        print('B 1', end=':  ')
        plt.plot(mv.time, mv.ib, color='green',   linestyle='-', label='ib'+suffix)
        plt.plot(mv.time, mv.ioc, color='magenta', linestyle='--', label='ioc'+suffix)
        plt.legend(loc=1)
        plt.subplot(323)
        plt.plot(mv.time, mv.vb, color='green', linestyle='-', label='vb'+suffix)
        plt.plot(sv.time, sv.vb, color='black', linestyle='--', label='vb_s'+suffix)
        plt.plot(mv.time, mv.voc_stat, color='orange', linestyle='-.', label='voc_stat'+suffix)
        plt.plot(sv.time, sv.voc_stat, color='cyan', linestyle=':', label='voc_stat_s,'+suffix)
        plt.plot(mv.time, mv.voc, color='magenta', label='voc'+suffix)
        plt.plot(sv.time, sv.voc, color='black', linestyle='--', label='voc_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(324)
        # plt.legend(loc=1)
        plt.subplot(322)
        plt.plot(mv.time, mv.soc, color='red', linestyle='-', label='soc'+suffix)
        plt.plot(sv.time, sv.soc, color='black', linestyle='dotted', label='soc_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(325)
        plt.plot(mv.time, mv.chm, color='cyan', linestyle='--', label='chm'+suffix)
        plt.plot(sv.time, sv.chm, color='black', linestyle=':', label='chm_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(326)
        plt.plot(sv.soc, sv.voc, color='red', linestyle='-', label='SIM voc_stat vs soc'+suffix)
        plt.plot(mv.soc, mv.voc_soc, color='black', linestyle='--', label='MON voc_soc'+suffix+' vs soc')
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 2
        fig_list += 1
        plt.subplot(111)
        plt.title(plot_title + ' B 2')
        print('B 2', end=':  ')
        plt.plot(mv.time, mv.vb, color='green', linestyle='-', label='vb'+suffix)
        plt.plot(sv.time, sv.vb, color='black', linestyle='--', label='vb_s'+suffix)
        plt.plot(mv.time, mv.voc_stat, color='orange', linestyle='-.', label='voc_stat'+suffix)
        plt.plot(sv.time, sv.voc_stat, color='cyan', linestyle=':', label='voc_stat'+suffix)
        plt.plot(mv.time, mv.voc, color='magenta', linestyle='-', label='voc'+suffix)
        plt.plot(sv.time, sv.voc, color='black', linestyle='--', label='voc_s'+suffix)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 4
        fig_list += 1
        plt.subplot(321)
        plt.title(plot_title+' B 4 MON vs SIM')
        print('B 4 MON vs SIM', end=':  ')
        plt.plot(mv.time, mv.ib, color='green', linestyle='-', label='ib'+suffix)
        plt.plot(sv.time, sv.ib, color='black', linestyle='--', label='ib_s'+suffix)
        plt.plot(sv.time, sv.ib_in, color='red', linestyle='-.', label='ib_in_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(322)
        plt.plot(mv.time, mv.vb, color='green', linestyle='-', label='vb'+suffix)
        plt.plot(sv.time, sv.vb, color='black', linestyle='--', label='vb_s'+suffix)
        plt.plot(mv.time, mv.voc, color='cyan', linestyle='-', label='voc'+suffix)
        plt.plot(sv.time, sv.voc, color='red', linestyle='--', label='voc_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(323)
        plt.plot(mv.time, mv.vb, color='green', linestyle='-', label='vb'+suffix)
        plt.plot(sv.time, sv.vb, color='orange', linestyle='--', label='vb_s'+suffix)
        plt.plot(mv.time, mv.voc, color='cyan', linestyle='-.', label='voc'+suffix)
        plt.plot(sv.time, sv.voc, color='red', linestyle=':', label='voc_s'+suffix)
        plt.plot(mv.time, mv.voc_stat, color='magenta', linestyle='--', label='voc_stat'+suffix)
        plt.plot(sv.time, sv.voc_stat, color='black', linestyle=':', label='voc_stat_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(324)
        plt.plot(mv.time, mv.dv_dyn, color='green', linestyle='-', label='dv_dyn'+suffix)
        plt.plot(sv.time, sv.dv_dyn, color='black', linestyle='--', label='dv_dyn_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(325)
        plt.plot(mv.time, mv.dv_hys, color='green', linestyle='-', label='dv_hys'+suffix)
        plt.plot(sv.time, sv.dv_hys, color='black', linestyle='--', label='dv_hys_s'+suffix)
        plt.plot(mv.time, mv.tau_hys, color='cyan', linestyle='-', label='tau_hys'+suffix)
        plt.plot(sv.time, sv.tau_hys, color='red', linestyle='--', label='tau_hys_s'+suffix)
        plt.legend(loc=1)
        plt.subplot(326)
        plt.plot(mv.time, mv.vb, color='green', linestyle='-', label='vb'+suffix)
        plt.plot(sv.time, sv.vb, color='black', linestyle='--', label='vb_s'+suffix)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 5
        fig_list += 1
        plt.subplot(331)
        plt.title(plot_title+' B 5 **EKF')
        print('B 5 **EKF', end=':  ')
        plt.plot(mv.time, mv.x_ekf, color='red', linestyle='-', label='x ekf'+suffix)
        plt.legend(loc=4)
        plt.subplot(332)
        plt.plot(mv.time, mv.hx, color='cyan', linestyle='-', label='hx ekf'+suffix)
        plt.plot(mv.time, mv.z_ekf, color='black', linestyle='--', label='z ekf'+suffix)
        plt.legend(loc=4)
        plt.subplot(333)
        plt.plot(mv.time, mv.y_ekf, color='green', linestyle='-', label='y ekf'+suffix)
        plt.plot(mv.time, mv.y_filt, color='black', linestyle='--', label='y filt'+suffix)
        plt.plot(mv.time, mv.y_filt2, color='cyan', linestyle='-.', label='y filt2'+suffix)
        plt.legend(loc=4)
        plt.subplot(334)
        plt.plot(mv.time, mv.H, color='magenta', linestyle='-', label='H ekf'+suffix)
        plt.ylim(0, 150)
        plt.legend(loc=3)
        plt.subplot(335)
        plt.plot(mv.time, mv.P, color='orange', linestyle='-', label='P ekf'+suffix)
        plt.legend(loc=3)
        plt.subplot(336)
        plt.plot(mv.time, mv.Fx, color='red', linestyle='-', label='Fx ekf'+suffix)
        plt.legend(loc=2)
        plt.subplot(337)
        plt.plot(mv.time, mv.Bu, color='blue', linestyle='-', label='Bu ekf'+suffix)
        plt.legend(loc=2)
        plt.subplot(338)
        plt.plot(mv.time, mv.K, color='red', linestyle='-', label='K ekf'+suffix)
        plt.legend(loc=4)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 6
        fig_list += 1
        plt.title(plot_title + ' B 6')
        print('B 6', end=':  ')
        plt.plot(mv.time, mv.e_voc_ekf, color='blue', linestyle='-.', label='e_voc'+suffix)
        plt.plot(mv.time, mv.e_soc_ekf, color='red', linestyle='dotted', label='e_soc_ekf'+suffix)
        plt.ylim(-0.01, 0.01)
        plt.legend(loc=2)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 7
        fig_list += 1
        plt.title(plot_title + ' B 7')
        print('B 7', end=':  ')
        plt.plot(mv.time, mv.voc, color='red', linestyle='-', label='voc'+suffix)
        plt.plot(mv.time, mv.voc_ekf, color='blue', linestyle='-.', label='voc_ekf'+suffix)
        plt.plot(sv.time, sv.voc, color='green', linestyle=':', label='voc_s'+suffix)
        plt.legend(loc=4)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 8
        fig_list += 1
        plt.title(plot_title + ' B 8')
        print('B 8', end=':  ')
        plt.plot(mv.time, mv.soc_ekf, color='blue', linestyle='-', label='soc_ekf'+suffix)
        plt.plot(sv.time, sv.soc, color='green', linestyle='-.', label='soc_s'+suffix)
        plt.plot(mv.time, mv.soc, color='red', linestyle=':', label='soc'+suffix)
        plt.legend(loc=4)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 9
        fig_list += 1
        plt.title(plot_title + ' B 9')
        print('B 9', end=':  ')
        plt.plot(mv.time, mv.e_voc_ekf, color='blue', linestyle='-.', label='e_voc'+suffix)
        plt.plot(mv.time, mv.e_soc_ekf, color='red', linestyle='dotted', label='e_soc_ekf'+suffix)
        plt.legend(loc=2)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 10
        fig_list += 1
        plt.subplot(221)
        plt.title(plot_title + ' B 10')
        print('B 10', end=':  ')
        plt.plot(sv.time, sv.soc, color='red', linestyle='-', label='soc'+suffix)
        plt.legend(loc=1)
        plt.subplot(223)
        plt.plot(sv.time, sv.ib, color='blue', linestyle='-', label='ib, A'+suffix)
        plt.plot(sv.time, sv.ioc, color='green', linestyle='-', label='ioc hys indicator, A'+suffix)
        plt.legend(loc=1)
        plt.subplot(224)
        plt.plot(sv.time, sv.dv_hys, color='red', linestyle='-', label='dv_hys, V'+suffix)
        plt.plot(sv.time, sv.tau_hys, color='blue', linestyle='--', label='tau_hys, V'+suffix)
        plt.legend(loc=2)
        fig_file_name = filename + "_" + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()  # Batt 11
        fig_list += 1
        plt.subplot(111)
        plt.title(plot_title + ' B 11')
        print('B 11', end=':  ')
        plt.plot(sv.soc, sv.voc_stat, color='black', linestyle='dotted', label='SIM voc_stat vs soc'+suffix)
        plt.legend(loc=2)
        fig_file_name = filename + "_" + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

    else:
        if use_time_day:
            mv.time = mv.time_day - mv.time_day[0]
            try:
                sv.time = sv.time_day - sv.time_day[0]
            except IOError:
                pass
            mv1.time = mv1.time_day - mv1.time_day[0]
            try:
                sv1.time = sv1.time_day - sv1.time_day[0]
            except IOError:
                pass
        reset_max = max(abs(min(mv.vbc_dot)), max(mv.vbc_dot), abs(min(mv1.vbc_dot)), max(mv1.vbc_dot))
        # noinspection PyTypeChecker
        reset_index_max = max(np.where(np.array(mv1.reset) > 0))
        t_init = mv1.time[reset_index_max[-1]]
        mv.time -= t_init
        mv1.time -= t_init
        sv.time -= t_init
        sv1.time -= t_init

        plt.figure()
        fig_list += 1
        plt.subplot(331)
        plt.title(plot_title + ' Battover 1')
        print('Battover 1', end=':  ')
        plt.plot(mv.time, mv.ib, color='green',   linestyle='-', label='ib'+suffix)
        plt.plot(mv1.time, mv1.ib, color='black', linestyle='--', label='ib' + suffix1)
        plt.plot(mv.time, mv.ioc, color='magenta', linestyle='-.', label='ioc'+suffix)
        plt.plot(mv1.time, mv1.ioc, color='blue', linestyle=':', label='ioc' + suffix1)
        plt.legend(loc=1)
        plt.subplot(332)
        # plt.legend(loc=1)
        plt.subplot(333)
        # plt.legend(loc=1)
        plt.subplot(334)
        plt.plot(mv.time, mv.vb, color='green', linestyle='-', label='vb' + suffix)
        plt.plot(mv1.time, mv1.vb, color='black', linestyle='--', label='vb' + suffix1)
        plt.legend(loc=1)
        plt.subplot(335)
        plt.plot(mv.time, mv.voc_stat, color='magenta', linestyle='-.', label='voc_stat' + suffix)
        plt.plot(mv1.time, mv1.voc_stat, color='blue', linestyle=':', label='voc_stat' + suffix1)
        plt.legend(loc=1)
        plt.subplot(336)
        plt.plot(mv.time, mv.voc, color='green', linestyle='-', label='voc' + suffix)
        plt.plot(mv1.time, mv1.voc, color='black', linestyle='--', label='voc' + suffix1)
        plt.legend(loc=1)
        plt.subplot(337)
        plt.plot(mv.time, mv.vbc_dot, color='green', linestyle='-', label='vbc_dot' + suffix)
        plt.plot(mv1.time, mv1.vbc_dot, color='black', linestyle='--', label='vbc_dot' + suffix1)
        plt.plot(mv.time, np.array(mv.reset)*reset_max, color='orange', linestyle='-', label='reset'+suffix)
        plt.plot(mv1.time, np.array(mv1.reset)*reset_max, color='cyan', linestyle='--', label='reset'+suffix1)
        plt.legend(loc=1)
        plt.subplot(338)
        plt.plot(mv.time, mv.vcd_dot, color='green', linestyle='-', label='vcd_dot' + suffix)
        plt.plot(mv1.time, mv1.vcd_dot, color='black', linestyle='--', label='vcd_dot' + suffix1)
        plt.legend(loc=1)
        plt.subplot(339)
        plt.plot(mv.time, mv.soc, color='green', linestyle='-', label='soc' + suffix)
        plt.plot(mv1.time, mv1.soc, color='black', linestyle='--', label='soc' + suffix1)
        plt.plot(mv.time, mv.soc_ekf, color='magenta', linestyle='-.', label='soc_ekf'+suffix)
        plt.plot(mv1.time, mv1.soc_ekf, color='blue', linestyle=':', label='soc_ekf'+suffix1)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()
        fig_list += 1
        plt.subplot(321)
        plt.title(plot_title + ' Battover 2')
        print('Battover 2', end=':  ')
        plt.plot(mv.time, mv.ib, color='green',   linestyle='-', label='ib'+suffix)
        plt.plot(mv1.time, mv1.ib, color='black', linestyle='--', label='ib' + suffix1)
        plt.plot(mv.time, mv.ioc, color='magenta', linestyle='-.', label='ioc'+suffix)
        plt.plot(mv1.time, mv1.ioc, color='blue', linestyle=':', label='ioc' + suffix1)
        plt.legend(loc=1)
        plt.subplot(322)
        plt.plot(mv.time, mv.dv_dyn, color='green', linestyle='-', label='dv_dyn'+suffix)
        plt.plot(mv1.time, mv1.dv_dyn, color='black', linestyle='--', label='dv_dyn'+suffix1)
        plt.legend(loc=1)
        plt.subplot(323)
        plt.plot(mv.time, mv.dv_hys, color='green', linestyle='-', label='dv_hys'+suffix)
        plt.plot(mv1.time, mv1.dv_hys, color='black', linestyle='--', label='dv_hys'+suffix1)
        plt.legend(loc=1)
        plt.subplot(324)
        plt.plot(mv.time, mv.tau_hys, color='green', linestyle='-', label='tau_hys'+suffix)
        plt.plot(mv1.time, mv1.tau_hys, color='black', linestyle='--', label='tau_hys'+suffix1)
        plt.legend(loc=1)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

        plt.figure()
        fig_list += 1
        plt.subplot(331)
        plt.title(plot_title + ' **EKF' + 'Battover 3')
        print('Battover 3', end=':  ')
        plt.plot(mv.time, mv.x_ekf, color='green', linestyle='-', label='x ekf' + suffix)
        plt.plot(mv1.time, mv1.x_ekf, color='black', linestyle='--', label='x ekf' + suffix1)
        plt.legend(loc=4)
        plt.subplot(332)
        plt.plot(mv.time, mv.hx, color='green', linestyle='-', label='hx ekf' + suffix)
        plt.plot(mv1.time, mv1.hx, color='black', linestyle='--', label='hx ekf' + suffix1)
        plt.plot(mv.time, mv.z_ekf, color='magenta', linestyle='-.', label='z ekf' + suffix)
        plt.plot(mv1.time, mv1.z_ekf, color='blue', linestyle=':', label='z ekf' + suffix1)
        plt.legend(loc=4)
        plt.subplot(333)
        plt.plot(mv.time, mv.y_ekf, color='green', linestyle='-', label='y ekf' + suffix)
        plt.plot(mv1.time, mv1.y_ekf, color='black', linestyle='--', label='y ekf' + suffix1)
        plt.plot(mv.time, mv.y_filt2, color='magenta', linestyle='-.', label='y filt2' + suffix)
        plt.plot(mv1.time, mv1.y_filt2, color='blue', linestyle=':', label='y filt2' + suffix1)
        plt.legend(loc=4)
        plt.subplot(334)
        plt.plot(mv.time, mv.H, color='green', linestyle='-', label='H ekf' + suffix)
        plt.plot(mv1.time, mv1.H, color='black', linestyle='--', label='H ekf' + suffix1)
        plt.ylim(0, 150)
        plt.legend(loc=3)
        plt.subplot(335)
        plt.plot(mv.time, mv.P, color='green', linestyle='-', label='P ekf' + suffix)
        plt.plot(mv1.time, mv1.P, color='black', linestyle='--', label='P ekf' + suffix1)
        plt.legend(loc=3)
        plt.subplot(336)
        plt.plot(mv.time, mv.Fx, color='green', linestyle='-', label='Fx ekf' + suffix)
        plt.plot(mv1.time, mv1.Fx, color='black', linestyle='--', label='Fx ekf' + suffix1)
        plt.legend(loc=2)
        plt.subplot(337)
        plt.plot(mv.time, mv.Bu, color='green', linestyle='-', label='Bu ekf' + suffix)
        plt.plot(mv1.time, mv1.Bu, color='black', linestyle='--', label='Bu ekf' + suffix1)
        plt.legend(loc=2)
        plt.subplot(338)
        plt.plot(mv.time, mv.K, color='green', linestyle='-', label='K ekf' + suffix)
        plt.plot(mv1.time, mv1.K, color='black', linestyle='--', label='K ekf' + suffix1)
        plt.legend(loc=4)
        plt.subplot(339)
        plt.plot(mv.time, mv.e_voc_ekf, color='green', linestyle='-', label='e_voc' + suffix)
        plt.plot(mv1.time, mv1.e_voc_ekf, color='black', linestyle='--', label='e_voc' + suffix1)
        plt.plot(mv.time, mv.e_soc_ekf, color='magenta', linestyle='-.', label='e_soc_ekf' + suffix)
        plt.plot(mv1.time, mv1.e_soc_ekf, color='blue', linestyle=':', label='e_soc_ekf' + suffix1)
        # plt.ylim(-0.01, 0.01)
        plt.legend(loc=2)
        fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
        fig_files.append(fig_file_name)
        plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files


class SavedS:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self):
        self.time_run = None
        self.time = []
        self.time_min = []
        self.time_day = []
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
