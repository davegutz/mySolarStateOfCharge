#ifndef pro0p_h
#define pro0p_h

#include "version.h"

// Features config
#define HDWE_UNIT                    "pro0p"
#define SOFT_SBAUD              230400      // Default Serial baud 
#define SOFT_S1BAUD             230400      // Default Serial1 baud when able to run AT to set it using AT+BAUD9
#define HDWE_PHOTON
#define HDWE_SSD1306_OLED
#define HDWE_ADS1013_AMP_NOA
#define HDWE_DS18B20_SWIRE
#define SOFT_DEPLOY_PHOTON
// #define SOFT_DEBUG_QUEUE
// #define DEBUG_DETAIL                    // Use this to debug initialization using 'v-1;' *****Not possible Photon.  Use Argon or Photon2
// #define LOGHANDLE

// * = SRAM EEPROM adjustments, retained on power reset

// Miscellaneous
#define ASK_DURING_BOOT       1   // Flag to ask for application of this file to * retained adjustements
#define MODELING              247 // Nominal modeling bitmap (* 'Xm'), 0=all hdwe, 1+=Tb, 2+=Vb, 4+=Ib, 7=all model.  +240 for discn

// Sensor biases
#define CURR_BIAS_AMP         -0.94 // Calibration of amplified shunt sensor (* 'DA'), A
#define CURR_SCALE_AMP        0.968 // Hardware to match data (* 'SA')
#define CURR_BIAS_NOA         -0.17 // Calibration of non-amplified shunt sensor (* 'DB'), A
#define CURR_SCALE_NOA        1.016 // Hardware to match data (* 'SB')
#define CURR_SCALE_DISCH      1.0   // Scale discharge to account for asymetric inverter action only on discharge (* 'SD'), slr
#define SHUNT_GAIN            1333. // Shunt V2A gain (scale with * 'SA' and 'SB'), A/V (-1333 is -100A/0.075V)
#define SHUNT_AMP_R1          5600.     // Amplifed shunt ADS resistance, ohms
#define SHUNT_AMP_R2          27000.    // Amplifed shunt ADS resistance, ohms
#define CURR_BIAS_ALL         0.0   // Bias on all shunt sensors (* 'DI'), A
#define VOLT_BIAS             1.8   // Bias on Vb sensor (* 'Dc'), V
#define TEMP_BIAS             0.0   // Bias on Tb sensor (* 'Dt'), deg C
#define VB_SENSE_R_LO      4700      // Vb low sense resistor, ohm (4700)
#define VB_SENSE_R_HI      20000     // Vb high sense resistor, ohm (20000)
#define VB_SCALE              1.017  // Scale Vb sensor (* 'SV')
#define VTAB_BIAS             0.0    // Bias on voc_soc table (* 'Dw'), V

// Battery.  One 12 V 100 Ah battery bank would have NOM_UNIT_CAP 100, NS 1, and NP 1
// Two 12 V 100 Ah series battery bank would have NOM_UNIT_CAP 100, NS 2, and NP 1
// Four 12 V 200 Ah with two in parallel joined with two more in series
//   would have  NOM_UNIT_CAP 200, NS 2, and NP 2
#define COULOMBIC_EFF_SCALE   1.0     // Scalar on Coulombic efficiency of battery, fraction of charge that gets used (1.0)
#define CHEM                  1       // Chemistry monitor code integer, 0=Battleborn, 1=CHINS-guest room, 2=CHINS-garage
#define NOM_UNIT_CAP          112.7   // Nominal battery unit capacity at RATED_TEMP.  (* 'Sc' or '*BS'/'*BP'), Ah
#define CHEM_NOM_VSAT       13.85   // Nominal saturation voltage at 25C, V (13.85)
#define HYS_SCALE             1.0     // Scalar on hysteresis (1.0)
#define NS                    1.0     // Number of series batteries in bank.  Fractions scale and remember NOM_UNIT_CAP (* 'BS')
#define NP                    2.0     // Number of parallel batteries in bank.  Fractions scale and remember NOM_UNIT_CAP (* 'BP')

// Faults
#define FAKE_FAULTS           false   // What to do with faults, T=detect and display them but don't change signals
#define CC_DIFF_SOC_DIS_THRESH  0.5   // Signal selection threshold for Coulomb counter EKF disagree test (0.2, 0.1 too small on truck.   0.5 CHINS)

#endif
