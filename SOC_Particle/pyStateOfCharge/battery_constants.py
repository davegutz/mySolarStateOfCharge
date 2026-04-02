# battery_constants.py - Battery class-level constant declarations
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

"""Mixin class holding all Battery class-level constant declarations.
All attributes are initialized to None and populated at runtime by the
configuration loader, except NOM_UNIT_CAP which has a fixed default."""


class BatteryConstants:
    AMP_WRAP_TRIM_GAIN = None
    ap_cc_diff_slr = None
    ap_dc_dc_on = None  # Truck charging
    ap_disab_ib_fa = None
    ap_disab_tb_fa = None
    ap_disab_vb_fa_lt = None
    ap_ds_voc_soc = None
    ap_dv_voc_soc = None
    ap_eframe_mult = None
    ap_ewhi_slr = None
    ap_ewlo_slr = None
    ap_hys_scale = None
    ap_ib_diff_slr = None
    ap_ib_quiet_slr = None
    CHEM = None
    D_SOC_S = 0.  # Bias on soc to voc-soc lookup to simulate error in estimation, esp cold battery near 0 C
    DF2 = None
    DISAB_LO_RESET = None
    DISAB_LO_SET = None
    EKF_CONV = None
    EKF_NOM_DT = None
    EKF_Q_SD_NORM = None
    EKF_R_SD_NORM = None
    EKF_T_CONV = None
    EKF_T_RESET = None
    EWHI_SLR = None
    EWHI_TRM_SLR = None
    EWLO_SLR = None
    EWLO_TRM_SLR = None
    F_MAX_T_WRAP = None
    HDB_VB = None
    hdwe_ib_hi_lo = None
    HDWE_IB_HI_LO_AMP_HI = None
    HDWE_IB_HI_LO_AMP_LO = None
    HDWE_IB_HI_LO_NOA_HI = None
    HDWE_IB_HI_LO_NOA_LO = None
    HYS_IB_THR = None
    HYS_SOC_MIN_MARG = None
    IB_ABS_MAX_AMP = None
    IB_ABS_MAX_NOA = None
    IB_DIFF_SLR = None
    IB_MIN_UP = None
    IBATT_DISAGREE_THRESH = None
    IMAX_NUM = None
    KF_Q_STD = None
    KF_R_STD = None
    MAX_TRIM_RATE = None
    MAX_WRAP_ERR_FILT = None
    MAX_Y_FILT = None
    MIN_Y_FILT = None
    MXEPS = None
    NOA_WRAP_TRIM_GAIN = None
    NOM_UNIT_CAP = 108.4
    NOMINAL_VB = None
    NP = None
    NS = None
    RATED_TEMP = None
    SHUNT_AMP_GAIN = None
    SHUNT_NOA_GAIN = None
    skip_battery = None
    sp_cutback_gain_slr = None
    sp_Dw = None
    sp_ib_disch_slr = None
    sp_s_cap_mon = None
    sp_s_cap_sim = None
    sp_vsat_add = None
    T_RLIM = None
    TAU_Y_FILT = None
    TB_FILT = None
    TB_MAX = None
    TB_MIN = None
    TCHARGE_DISPLAY_DEADBAND = None
    TMAX_FILT = None
    VB_DC_DC = None
    VB_MAX = None
    VB_MIN = None
    VOC_STAT_FILT = None
    WN_Y_FILT = None
    WRAP_ERR_FILT = None
    WRAP_HI_AMP = None
    WRAP_HI_NOA = None
    WRAP_HI_R = None
    WRAP_HI_S = None
    WRAP_HI_SAT_MARG = None
    WRAP_LO_AMP = None
    WRAP_LO_NOA = None
    WRAP_LO_R = None
    WRAP_LO_S = None
    WRAP_MOD_C_RATE = None
    WRAP_SOC_HI_OFF = None
    WRAP_SOC_HI_SLR = None
    WRAP_SOC_LO_OFF_ABS = None
    WRAP_SOC_LO_OFF_REL = None
    WRAP_SOC_LO_SLR = None
    WRAP_SOC_MOD_OFF = None
    ZETA_Y_FILT = None
