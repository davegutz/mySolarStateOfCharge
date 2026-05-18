/*  Heart rate and pulseox calculation Constants

18-Dec-2020 	DA Gutz 	Created from MAXIM code.
// Copyright (C) 2026 - Dave Gutz
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

*/
#pragma once

// Hardware configuration
#undef HDWE_UNIT
#undef HDWE_BARE
#undef SOFT_SBAUD
#undef HDWE_IB_HI_LO
#undef HDWE_2WIRE
#undef HDWE_IB_HI_LO_NOA_LO
#undef HDWE_IB_HI_LO_AMP_LO
#undef HDWE_IB_HI_LO_AMP_HI
#undef HDWE_IB_HI_LO_NOA_HI
#undef CURR_BIAS_AMP
#undef CURR_SCALE_AMP
#undef CURR_BIAS_NOA
#undef CURR_SCALE_NOA
#undef CURR_SCALE_DISCH
#undef SHUNT_GAIN
#undef SHUNT_AMP_R1
#undef SHUNT_AMP_R2
#undef VB_SENSE_R_LO
#undef VB_SENSE_R_HI
#undef VB_SCALE
#undef VTAB_BIAS
#undef VOLT_BIAS
#undef CURR_BIAS_ALL
#undef TEMP_BIAS
#undef TB_MAX
#undef TB_MIN
#undef CHEM_NOM_VSAT
#undef VOC_STAT_FILT

// Software configuration
#undef SOFT_DEPLOY_PHOTON
#undef SOFT_DEBUG_QUEUE
#undef IB_FORCE
// #undef PLATFORM_ID

// Setup
#include "local_config.h"
#ifndef IB_FORCE
    #define IB_FORCE 0
#endif
#ifndef DISAB_TB_FA
    #define DISAB_TB_FA false
#endif
#ifndef DISAB_VB_FA_LT
    #define DISAB_VB_FA_LT false
#endif
// #ifndef PLATFORM_ID
//     #define PLATFORM_ID  32
// #endif

const char unit[] = version_str "_" HDWE_UNIT;

// Constants always defined
#define ONE_DAY_MILLIS        86400000UL// Number of milliseconds in one day (24*60*60*1000)
#define TALK_DELAY            313UL      // Talk wait, ms (313UL = 0.313 sec)
#define NO_SAVE_WARN          30000UL   // Unsaved retained param warning interval, ms (30000UL = 30 sec)
#define READ_DELAY            100UL     // Sensor read wait, ms (100UL = 0.1 sec) Dr
#define TEMP_DELAY            600UL     // Sensor read wait, ms (600UL = 0.6 sec)
#define SUMMARY_DELAY         1800000UL // Battery state tracking and reporting, ms (1800000UL = 30 min) Dh
#define SUMMARY_WAIT          60000UL   // Summarize alive time before first save, ms (60000UL = 1 min) Dh
#define DISPLAY_USER_DELAY    1200UL    // User display update (1200UL = 1.2 sec)
#define DP_MULT               4         // Multiples of read to capture data DP
#define VB_S                  1.0       // Vb sense scalar (1.0)
#define VB_A                  0.0       // Vb sense adder, V (0)
#define PHOTON_ADC_COUNT      4096      // Photon ADC range, counts (4096)
#define PHOTON_ADC_VOLT       3.3       // Photon ADC range, V (3.3)
#define TB_FILT               120.      // Temperature filter lag, s (120)
#define SCL_40                40.       // Data storage integer scaling
#define SCL_60                60.       // Data storage integer scaling
#define SCL_600               600.      // Data storage integer scaling
#define SCL_1200              1200.     // Data storage integer scaling
#define SCL_1500              1500.     // Data storage integer scaling
#define SCL_3000              3000.f    // Data storage integer scaling
#define SCL_16000             16000.    // Data storage integer scaling
#define SCL_30000             30000.    // Data storage integer scaling

// If NSUM too large, will get flashing red with auto reboot on 'Hs' or compile error `.data' will not fit in region `APP_FLASH'
// For all, there are 40 bytes for each unit of NSUM
// Baseline compile information 20251227
//   text    data     bss     dec     hex filename
// 292998  119852   10306  423168   67500 c:/Users/daveg/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/target/6.2.1/p2/SOC_Particle.elf

#define NFLT    7  // Number of saved SRAM fault data slices 10 s intervals (7)
#define NHIS   43  // Number of saved SRAM history data slices. If NFLT + NHIS too large will get compile error BACKUPSRAM, BACKUPSRAM_USER  (45)
#define NSUM 2000  // Number of saved summaries. If NFLT + NHIS + NSUM too large, will get compile error SRAM, or GUI FRAG msg (2845) or SOS 4 Bus Fault (2500)

#define HDB_VB                0.05      // Half deadband to filter Vb, V (0.05)
#ifndef HDWE_IB_HI_LO
    #define T_SAT                 22        // Saturation time, sec (>21 for no SAT with Dv0.82)
#else
    #define T_SAT                 24        // Saturation time, sec (>21 for no SAT with Dv0.82)
#endif
const float T_DESAT =         20;       // De-saturation time, sec
#ifndef TEMP_INIT_DELAY
    #define TEMP_INIT_DELAY 700         // Time after power on to start reading temp, ms (700)
#endif
#define CC_DIFF_LO_SOC_SLR    4.        // Large to disable cc_diff
#define TAU_ERR_FILT          5.        // Current sensor difference filter time constant, s (5.)
#define IB_LO_ACTIVE_SET      0.2       // Ib low range sensor is in-range persistence, s (0.2)
#define IB_LO_ACTIVE_RES      0.4       // Ib low range sensor is in-range reset persistence, s (0.4)
#define VB_MAX                17.       // Signal selection hard fault threshold, V (17. < VB_CONV_GAIN*4095)
#define VB_MIN                2.        // Signal selection hard fault threshold, V (0.  < 2. < 10 bms shutoff, reads ~3 without power when off)
#define VC_MAX                2.15      // Signal selection hard fault threshold, V (2.15, 1.85 too low)  // 1.65*1.3 is max ADC reading with 1.65v ref, but see 1.9v on truck with no power, so set at 2.15v to avoid false fault on truck when off
#define VC_MIN                1.1       // Signal selection hard fault threshold, V (1.1, 1.4 too high)
#define IB_MIN_UP             0.2       // Min up charge current for come alive, BMS logic, and fault
#define TB_MAX                60.       // Signal selection hard fault threshold 2wire only, C (60.)
#define TB_MIN               -40.       // Signal selection hard fault threshold 2wire only, C (-40.)
#define TB_HARD_SET           1.        // Signal selection Tb 2-wire range fail persistence, s (1.)
#define TB_HARD_RES           2.        // Signal selection Tb 2-wire range fail reset persistence, s (2.)
#define VB_HARD_SET           1.        // Signal selection volt range fail persistence, s (1.)
#define VB_HARD_RES           2.        // Signal selection volt range fail reset persistence, s (2.)
#define VC_HARD_SET           1.        // Signal selection volt range fail persistence, s (1.)
#define VC_HARD_RES           2.        // Signal selection volt range fail reset persistence, s (2.)
#define TB_NOISE              0.        // Tb added noise amplitude, deg C pk-pk
#define TB_NOISE_SEED         0xe2      // Tb added noise seed 0-255 = 0x00-0xFF (0xe2) 
#define VB_NOISE              0.        // Vb added noise amplitude, V pk-pk
#define VB_NOISE_SEED         0xb2      // Vb added noise seed 0-255 = 0x00-0xFF (0xb2)
#define IB_AMP_NOISE          0.        // Ib amplified sensor added noise amplitude, A pk-pk
#define IB_NOA_NOISE          0.        // Ib non-amplified sensor added noise amplitude, A pk-pk
#define IB_AMP_NOISE_SEED     0x01      // Ib amplified sensor added noise seed 0-255 = 0x00-0xFF (0x01) 
#define IB_NOA_NOISE_SEED     0x0a      // Ib non-amplified sensor added noise seed 0-255 = 0x00-0xFF (0x0a) 
#define WRAP_ERR_FILT         4.        // Wrap error filter time constant, s (4)
#define F_MAX_T_WRAP          2.8       // Maximum update time of Wrap filter for stability at WRAP_ERR_FILT (0.7*T for Tustin), s (2.8)
#define MAX_WRAP_ERR_FILT     10.       // Anti-windup wrap error filter, V (10)
const float WRAP_LO_SET =      9.;      // Wrap low failure set time, sec (9) // 9 is legacy must be quicker than SAT test
const float WRAP_LO_RES = (WRAP_LO_SET/2.); // Wrap low failure reset time, sec ('up 1, down 2')
const float WRAP_HI_SET = WRAP_LO_SET;      // Wrap high failure set time, sec (WRAP_LO_SET)
const float WRAP_HI_RES = (WRAP_HI_SET/2.); // Wrap high failure reset time, sec ('up 1, down 2')
#define WRAP_HI_AMPV        0.5         // Wrap high voltage threshold amplified, V (1.5)
#define WRAP_LO_AMPV       -0.5         // Wrap low voltage threshold amplified, V (-1.5)
#define WRAP_HI_NOAV        0.8         // Wrap high voltage threshold non-amplified, V (0.8)
#define WRAP_LO_NOAV       -0.8         // Wrap low voltage threshold non-amplified, V (-0.8)
#define WRAP_HI_SETAT_MARG  0.2         // Wrap voltage margin to saturation, V (0.2)
#define WRAP_HI_SETAT_SLR   2.0         // Wrap voltage margin scalar when saturated (2.0)
#ifdef HDWE_IB_HI_LO
    #define IBATT_DISAGREE_THRESH 3.    // Signal selection threshold for current disagree test, A (3.)
#else
    #define IBATT_DISAGREE_THRESH 10.   // Signal selection threshold for current disagree test, A (10.)
#endif
const float IBATT_DISAGREE_SET = (WRAP_LO_SET-1.); // Signal selection current disagree fail persistence, s (WRAP_LO_SET-1) // must be quicker than wrap lo
#define IBATT_INST_DIFF_SET   0.2       // Persistence on instantaneous current difference, s (0.2)
#define IBATT_INST_DIFF_RES   0.0       // Persistence reset on instantaneous current difference, s (0.0)
#define IBATT_DISAGREE_RES    2.0       // Signal selection current disagree reset persistence, s (2.)
#define TAU_Q_FILT      0.5             // Quiet rate time constant, sec (0.5)
#define MIN_Q_FILT      -5.0            // Quiet filter minimum, V (-0.5)
#define MAX_Q_FILT      5.0             // Quiet filter maximum, V (0.5)
#define WN_Q_FILT       1.0             // Quiet filter-2 natural frequency, r/s (1.0)
#define ZETA_Q_FILT     0.9             // Quiet fiter-2 damping factor (0.9)
#define MAX_T_Q_FILT    0.2             // Quiet filter max update time (0.2)
#define QUIET_A         0.005           // Quiet set threshold, sec (0.005, 0.01 too large in truck)
#define LOW_A           1.0             // Currents are very small, A (1.0)
#define QUIET_SET         60.           // Quiet set persistence, sec (60.)
const float QUIET_RES (QUIET_SET/10.);  // Quiet reset persistence, sec ('up 1 down 10')
#define NOMINAL_TB      15.             // Middle of the road Tb for decent reversionary operation, deg C (15.)
#define NOMINAL_VB   (13.*NS)           // Middle of the road Vb for decent reversionary operation, V (13.)
#define IMAX_NUM        100000.         // Simulation limit to prevent NaN, A (1e5)
#ifndef WRAP_SOC_HI_OFF
    #define WRAP_SOC_HI_OFF     0.97        // Disable e_wrap_hi when saturated (0.97)
#endif
#define WRAP_SOC_HI_SLR     1000.       // Huge to disable e_wrap (1000)
#define WRAP_SOC_LO_OFF_ABS 0.35        // Disable e_wrap when near empty (soc lo any Tb, 0.35)
#define WRAP_SOC_LO_OFF_REL 0.2         // Disable e_wrap when near empty (soc lo for high Tb where soc_min=.2, voltage cutback, 0.2)
#define WRAP_SOC_LO_SLR     60.         // Large to disable e_wrap (60. for startup)
#define WRAP_MOD_C_RATE     .02         // Moderate charge rate threshold to engage wrap threshold (0.02 to prevent trip near saturation .05 too large)
#define WRAP_SOC_MOD_OFF    0.85        // Disable e_wrap_lo when nearing saturated and moderate C_rate (0.85)
#define AMP_WRAP_TRIM_GAIN  0.015       // Amp looparound trim gain r/s (0.015)
#define NOA_WRAP_TRIM_GAIN  0.015       // Non-Amp looparound trim gain r/s (0.015)
#define VC_S                1.0         // Vc sense scalar (1.0)
#define VO_S                1.0         // Vo sense scalar (1.0)
#define VTB_S               1.0         // VTb sense scalar (1.0)
#define AMP_FILT_TAU        4.0         // Ib filters time constant for calibration only, s (4.0)
#define V3V3                3.3         // Theoretical nominal V3v3, V (3.3)
#define HALF_V3V3         (V3V3/2.)     // Theoretical center of differential TSC2010
#define HDWE_RS_2WIRE   15000.          // 2-wire sense resistor, ohm (15000.)
#define HDWE_M_2WIRE    -58.96          // 2-wire thermistor characteristic, data fit (-58.96; see '2-wireRTD.ods')
#define HDWE_B_2WIRE    262.79          // 2-wire thermistor characteristic, data fit (262.79; see '2-wireRTD.ods')
#define HDWE_SHA_2WIRE  9.8194e-4       // 2-wire thermistor characteristic, Steinhart-Hart (9.8194e-4; see '2-wireRTD.ods')
#define HDWE_SHB_2WIRE  2.4775e-4       // 2-wire thermistor characteristic, Steinhart-Hart (2.4775e-4; see '2-wireRTD.ods')
#define HDWE_SHC_2WIRE  1.0265e-7       // 2-wire thermistor characteristic, Steinhart-Hart (1.0265e-7; see '2-wireRTD.ods')
#define SIZE_MARG         1.05          // Threshold margin, scalar (1.05)
#define CC_DIFF_RES         2.0         // Signal selection cc_diff ekf test reset persistence, s (2.)
#define CC_DIFF_SET         5.0         // Signal selection cc_diff ekf test set persistence, s (5. to handle sawtooth action on cc_diff)
#define MAX_TRIM_RATE     0.005         // Max allowable amp e_wraptrim rate, V/s (0.005)

// Default values for constants that can be overridden
#if !defined(IB_HARD_SET)
    #define IB_HARD_SET        1.0          // Signal selection volt range fail persistence, s (1.)
#endif
#if !defined(IB_HARD_RES)
    #define IB_HARD_RES        2.0          // Signal selection volt range fail reset persistence, s (2.)
#endif
#if !defined(NOM_DS)
    #define NOM_DS             0.0          // Nominal VOC(SOC) del soc (Ds) 0.0)
#endif
#if !defined(NOM_DY)
    #define NOM_DY             0.0          // Nominal Dy Sim table bias (Dy) (0.0)
#endif
#if !defined(VV)
    #define VV                   0          // Nominal verbosity (vv) (0)
#endif
#if !defined(TEMP_BIAS)
    #define TEMP_BIAS          0.0          // Nominal bias on Tb (D^) (0.0)
#endif
#if !defined(NOM_VB_ADD)
    #define NOM_VB_ADD         0.0          // Nominal bias on Vb (Dv) (0.0)
#endif
#if !defined(NOM_VC_ADD)
    #define NOM_VC_ADD         0.0          // Nominal bias on Vc (D3) (0.0)
#endif
#if !defined(IB_ABS_MAX_AMP)
    #define IB_ABS_MAX_AMP (float(NOM_UNIT_CAP)*float(NP))
#endif
#if !defined(IB_ABS_MAX_NOA)
    #define IB_ABS_MAX_NOA (float(NOM_UNIT_CAP)*float(NP))
#endif
#if !defined(HDWE_IB_HI_LO_AMP_LO)
    #define HDWE_IB_HI_LO_AMP_LO (-float(NOM_UNIT_CAP)*float(NP))
#endif
#if !defined(HDWE_IB_HI_LO_AMP_HI)
    #define HDWE_IB_HI_LO_AMP_HI (float(NOM_UNIT_CAP)*float(NP))
#endif
#if !defined(HDWE_IB_HI_LO_NOA_LO)
    #define HDWE_IB_HI_LO_NOA_LO (HDWE_IB_HI_LO_AMP_LO - 1.)
#endif
#if !defined(HDWE_IB_HI_LO_NOA_HI)
    #define HDWE_IB_HI_LO_NOA_HI (HDWE_IB_HI_LO_AMP_HI + 1.)
#endif
#if !defined(CURR_BIAS_AMP)
    #define CURR_BIAS_AMP 0.
#endif
#if !defined(CURR_BIAS_NOA)
    #define CURR_BIAS_NOA 0.
#endif
#if !defined(CURR_SCALE_AMP)
    #define CURR_SCALE_AMP 1.
#endif
#if !defined(CURR_SCALE_NOA)
    #define CURR_SCALE_NOA 1.
#endif
#if !defined(CURR_SCALE_DISCH)
    #define CURR_SCALE_DISCH 1.
#endif
#if !defined(VB_SCALE)
    #define VB_SCALE 1.
#endif
#if !defined(VOLT_BIAS)
    #define VOLT_BIAS 0.
#endif
#if !defined(CURR_BIAS_ALL)
    #define CURR_BIAS_ALL 0.
#endif
#if !defined(SHUNT_GAIN)
    #define SHUNT_GAIN            1333. // Shunt V2A gain (scale with 'SA' and 'SB'), A/V (1333 is 100A/0.075V)
#endif
#if !defined(KF_USE_AMP)
    #define KF_USE_AMP        false
#endif
#if !defined(HDWE_2WIRE)
    #define HDWE_2WIRE        true
#endif
#if !defined(KF_USE_NOA)
    #define KF_USE_NOA        true
#endif
#if !defined(KF_Q_STD)
    #define KF_Q_STD        0.0003 // Shunt KF process uncertainty this combo gives 10:1 attenuation  tune 2025128 like ishunt_cal_filt F_W_I=0.5, F_Z_I=0.8
#endif
#if !defined(KF_R_STD)
    #define KF_R_STD        0.1000 // Shunt KF state uncertainty  tune 2025128 like ishunt_cal_filt F_W_I=0.5, F_Z_I=0.8
#endif
#if !defined(EWLO_TRM_SLR)
    #define EWLO_TRM_SLR    25. // Amp looparound low error trim scalar.  Should provide hysteresis (~1 V) authority (25.)
#endif
#if !defined(EWHI_TRM_SLR)
    #define EWHI_TRM_SLR    25. // Amp looparound high error trim scalar.  Should provide hysteresis (~1 V) authority (25.)
#endif
#if !defined(FI_NOM)
    #define FI_NOM    1.0 // Hi wrap threshold nominal scalar (1.0)  // Fi
#endif
#if !defined(FO_NOM)
    #define FO_NOM    1.0 // Lo wrap threshold nominal scalar (1.0)  // Fo
#endif
#if !defined(VSAT_ADD)
    #define VSAT_ADD    0.0f // Bias on nominal vsat (0.0f)
#endif
#if !defined(VTAB_BIAS)
    #define VTAB_BIAS   0.0 // Bias on voc_soc table (* 'Dw'), V (0.0)
#endif

// Ib hardware
#if !defined(SHUNT_AMP_R1)
    #define SHUNT_AMP_R1    1500. // Internal amp resistance 196x, ohms (1500)
#endif
#if !defined(SHUNT_AMP_R2)
    #define SHUNT_AMP_R2    332000. // Internal amp resistance 196x, ohms (332000)
#endif
#if !defined(SHUNT_NOA_R1)
    #define SHUNT_NOA_R1    1500. // Internal amp resistance 29.4x, ohms (1500)
#endif
#if !defined(SHUNT_NOA_R2)
    #define SHUNT_NOA_R2    33200. // Internal amp resistance 29.4x, ohms (33200)
#endif
#if !defined(VRAW_BARE_DETECTED)
  #define VRAW_BARE_DETECTED 500 // Level of common voltage to declare circuit unconnected, V (50UL)
#endif

// Vb Hardware
#if !defined(VB_SENSE_R_LO)
    #define VB_SENSE_R_LO   4700 // Vb low sense resistor, ohm (4700)
#endif
#if !defined(VB_SENSE_R_HI)
    #define VB_SENSE_R_HI   22000 // Vb high sense resistor, ohm (22000)
#endif

// Chemistry and modeling
#if !defined(COULOMBIC_EFF_SCALE)
    #define COULOMBIC_EFF_SCALE 1.0 // Scalar on Coulombic efficiency of battery, fraction of charge that gets used (1.0)
#endif
#if !defined(CHEM)
    #define CHEM    0 // Chemistry monitor code integer, 0=Battleborn (0)
#endif
#if !defined(NOM_UNIT_CAP)
    #define NOM_UNIT_CAP    108.4 // Nominal battery unit capacity (* 'Sc' or 'BS'/'BP'), Ah (108.4)
#endif
#if !defined(CHEM_NOM_VSAT)
    #define CHEM_NOM_VSAT   13.85 // Nominal saturation voltage at 25C, V (13.85)
#endif
#if !defined(HYS_SCALE)
    #define HYS_SCALE   1.0 // Scalar on hysteresis (1.0)
#endif

// Bank configuration
#if !defined(NS)
    #define NS  1.0 // Number of series batteries in bank (* 'BS') (1.0)
#endif
#if !defined(NP)
    #define NP  1.0 // Number of parallel batteries in bank (* 'BP') (1.0)
#endif

// Faults
#if !defined(FAKE_FAULTS)
    #define FAKE_FAULTS false // Detect and display faults but don't change signals (false)
#endif
#if !defined(CC_DIFF_SOC_DIS_THRESH)
    #define CC_DIFF_SOC_DIS_THRESH  0.5 // Signal selection threshold for Coulomb counter EKF disagree test (0.5)
#endif
#if !defined(EKF_Q_SD_NORM)
    #define EKF_Q_SD_NORM   0.0015 // Standard deviation of normal EKF process uncertainty, V (0.0015)
#endif
#if !defined(EKF_R_SD_NORM)
    #define EKF_R_SD_NORM   0.5 // Standard deviation of normal EKF state uncertainty, fraction (0-1) (0.5)
#endif
#if !defined(EKF_EFRAME_MULT)
    #define EKF_EFRAME_MULT 20 // EKF multiframe: READ executes per EKF execute (20)
#endif
#if !defined(VOC_STAT_FILT)
    #define VOC_STAT_FILT   120. // voc_stat_f_ filtering time constant for EKF, s (120)
#endif
#if !defined(EKF_CONV)
    #define EKF_CONV    0.025 // EKF tracking error indicating convergence, V (0.025)
#endif
#if !defined(ASK_DURING_BOOT)
    #define ASK_DURING_BOOT       0   // Flag to ask for application of this file to * retained adjustements 0=retain,1=ask,2=force default
#endif
#if !defined(SNAP_WAIT)
    #define SNAP_WAIT             10000ULL  // Interval between fault snapshots (10000ULL = 10 sec)
#endif
#if !defined(RAW_BARE_SET)
    #define RAW_BARE_SET   1. // Raw bare set persistence, s (1)
#endif
#if !defined(RAW_BARE_RES)
    #define RAW_BARE_RES   2. // Raw bare reset persistence, s (2)
#endif
#if !defined(IN_SERVICE)
    #define IN_SERVICE      true // In service flag for testing (true)
#endif
#if !defined(TB_HDWE_MIN)
    #define TB_HDWE_MIN   -20.0
#endif
#if !defined(TB_HDWE_MAX)   
    #define TB_HDWE_MAX   150.0
#endif

// Conversion gains
// Voltage measurement gains
#if !defined(HDWE_BARE)
    const float VB_CONV_GAIN = float(PHOTON_ADC_VOLT) * float(VB_SENSE_R_HI + VB_SENSE_R_LO) /
                                  float(VB_SENSE_R_LO) / float(PHOTON_ADC_COUNT) * float(VB_S);
    const float VB_RAW_CONV_GAIN = float(PHOTON_ADC_VOLT) / float(PHOTON_ADC_COUNT) * float(VB_S);
#endif
const float VC_CONV_GAIN = float(PHOTON_ADC_VOLT) / float(PHOTON_ADC_COUNT) * float(VC_S);
const float VO_CONV_GAIN = float(PHOTON_ADC_VOLT) / float(PHOTON_ADC_COUNT) * float(VO_S);
#if defined(HDWE_IB_HI_LO) & !defined(HDWE_BARE)
    const float SHUNT_AMP_GAIN = SHUNT_GAIN * SHUNT_AMP_R1 / SHUNT_AMP_R2;
    const float SHUNT_NOA_GAIN = SHUNT_GAIN * SHUNT_NOA_R1 / SHUNT_NOA_R2;
#elif !defined(HDWE_BARE)
    const float SHUNT_AMP_GAIN = SHUNT_GAIN * SHUNT_AMP_R1 / SHUNT_AMP_R2;
    const float SHUNT_NOA_GAIN = SHUNT_GAIN;
#else
    const float SHUNT_AMP_GAIN = SHUNT_GAIN * 220;
    const float SHUNT_NOA_GAIN = SHUNT_GAIN * 22;
#endif

const float VH3V3_CONV_GAIN = float(PHOTON_ADC_VOLT) / float(PHOTON_ADC_COUNT);
const float VTB_CONV_GAIN = float(PHOTON_ADC_VOLT) / float(PHOTON_ADC_COUNT) * float(VTB_S);

// Time scalar for modifying fault logic time delays when deliberately running slowly for regression testing
const double NOM_READ_DELAY_S = double(READ_DELAY) / 1000.;
