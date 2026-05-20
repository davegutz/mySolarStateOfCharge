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


import numpy as np


class BatteryConstants:
    AMP_WRAP_TRIM_GAIN = 10.0
    ap_cc_diff_slr = 1.0
    ap_dc_dc_on = 0  # Truck charging
    ap_disab_ib_fa = 0
    ap_disab_tb_fa = 0
    ap_disab_vb_fa_lt = 0
    ap_ds_voc_soc = 0.0
    ap_dv_voc_soc = 0.0
    ap_eframe_mult = 1.0
    ap_ewhi_slr = 1.0
    ap_ewlo_slr = 1.0
    ap_hys_scale = 1.0
    ap_ib_diff_slr = 1.0
    ap_ib_quiet_slr = 1.0
    cp_ts = 1.0
    CHEM = 0
    D_SOC_S = 0.  # Bias on soc to voc-soc lookup to simulate error in estimation, esp cold battery near 0 C
    DF2 = 0.0
    EKF_CONV = 0.5
    EKF_NOM_DT = 0.1
    EKF_Q_SD_NORM = 0.0003
    EKF_R_SD_NORM = 0.1
    EKF_T_CONV = 1.
    EKF_T_RES = 2.
    EWHI_SLR = 1.0
    EWHI_TRM_SLR = 2.5
    EWLO_SLR = 1.0
    EWLO_TRM_SLR = 2.5
    F_MAX_T_WRAP = 10.0
    HDB_VB = 0.01
    hdwe_ib_hi_lo = 1
    HDWE_IB_HI_LO_AMP_HI = 10.0
    HDWE_IB_HI_LO_AMP_LO = -10.0
    HDWE_IB_HI_LO_NOA_HI = 11.0
    HDWE_IB_HI_LO_NOA_LO = -11.0
    HYS_IB_THR = 0.5
    HYS_SOC_MIN_MARG = 0.05
    IB_ABS_MAX_AMP = 100.0
    IB_ABS_MAX_NOA = 100.0
    IB_DIFF_SLR = 1.0
    IB_LO_ACTIVE_SET = 0.2
    IB_LO_ACTIVE_RES = 0.4
    IB_MIN_UP = 0.1
    IBATT_DISAGREE_THRESH = 1.0
    IMAX_NUM = 100
    KF_Q_STD = 0.0003
    KF_R_STD = 0.1
    MAX_TRIM_RATE = 1.0
    MAX_WRAP_ERR_FILT = 10.
    MAX_Y_FILT = 1.
    MIN_Y_FILT = 0.
    MXEPS = 1e-6
    NOA_WRAP_TRIM_GAIN = 10.0
    NOM_UNIT_CAP = 108.4
    NOMINAL_TB = 15.
    NOMINAL_VB = 13.2
    NP = 1
    NS = 1
    RATED_TEMP = 25.0
    SHUNT_AMP_GAIN = 6.0225902
    SHUNT_NOA_GAIN = 60.2259026
    skip_battery = 0
    sp_cutback_gain_slr = 1.0
    sp_Dw = 0.0
    sp_ib_disch_slr = 1.0
    sp_ib_disch_slr_z = None
    sp_s_cap_mon = 1.0
    sp_s_cap_sim = 1.0
    sp_vsat_add = 0.0
    T_RLIM = 0.017
    TAU_Y_FILT = 1.
    TB_FILT = 120.
    TB_MAX = 60.
    TB_MIN = -20.
    TCHARGE_DISPLAY_DEADBAND = 0.1
    TMAX_FILT = 1.
    VB_DC_DC = 12.
    VB_MAX = 14.8
    VB_MIN = 10.
    VOC_STAT_FILT = 100.
    WN_Y_FILT = 1.
    WRAP_ERR_FILT = 100.
    WRAP_HI_AMPV = 0.5
    WRAP_LO_AMPV = -0.5
    WRAP_HI_NOAV = 1.5
    WRAP_LO_NOAV = -1.5
    WRAP_HI_RES = 4.5
    WRAP_HI_SET = 9.0
    WRAP_HI_SETAT_MARG = 0.2
    WRAP_HI_SETAT_SLR = 2.0
    WRAP_LO_RES = 4.5
    WRAP_LO_SET = 9.0
    WRAP_MOD_C_RATE = 0.02
    WRAP_SOC_HI_OFF = 0.94
    WRAP_SOC_HI_SLR = 1000.0
    WRAP_SOC_LO_OFF_ABS = 0.35
    WRAP_SOC_LO_OFF_REL = 0.20
    WRAP_SOC_LO_SLR = 60.0
    WRAP_SOC_MOD_OFF = 0.85
    ZETA_Y_FILT = 0.707


# noinspection PyPep8Naming
def load_off_nominal_battery(Battery_to_add=None):
    # Load off-nominal Battery values.  Load Battery
    if Battery_to_add is not None:
        # Scroll through all off-nominals make dictionary
        Battery_off_dict = {}
        for field_name in Battery_to_add.dtype.names:
            # print(f"field_name {field_name}  ", end='')
            try:
                Battery_off_dict[field_name] = Battery_to_add[field_name][0]  # Use first entry only.  Discard the rest
            except IndexError:
                Battery_off_dict[field_name] = Battery_to_add[field_name]
                print(f"Battery_off field_name {field_name}   value {Battery_to_add[field_name]}")
        # print(self.Battery_off_dict)
        # Print affected values
        # print(f"dictionary to apply to Battery class")
        # if Battery_off_dict:
        #     for key in dir(Battery_to_add):
        #         if key in Battery_off_dict and not key.startswith('__'):
        #             print(f"Battery.{key} {getattr(Battery_to_add, key)} --> ", end='')
        #             print("Battery.{:s} = {:8.6g}".format(key, Battery_off_dict[key]))
        return Battery_off_dict
    else:
        return None


# noinspection PyPep8Naming
def apply_off_nominal_battery(Battery_, Battery_off_dict):
    print(f"dictionary to apply to immutable Battery class")
    if Battery_off_dict:
        # Check exist
        for key in Battery_off_dict:
            if not np.isnan(Battery_off_dict[key]):
                if not key.startswith('__')  and  key in dir(Battery_):
                    # print(f"Battery.{key} = {getattr(Battery_, key)} to be replaced")
                    pass
                else:
                    print(f"{key} MISSING  *****************")
                    # exit(1)
        # Make translation
        for key in dir(Battery_):
            if key in Battery_off_dict and not key.startswith('__'):
                # print(f"Battery.{key} {getattr(Battery_, key)} --> ", end='')
                setattr(Battery_, key, Battery_off_dict[key])
