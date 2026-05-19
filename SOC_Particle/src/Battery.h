//
// MIT License
//
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


#pragma once

#include "myLibrary/myTables.h"
#include "myLibrary/EKF_1x1.h"
#include "Coulombs.h"
#include "myLibrary/injection.h"
#include "myLibrary/myFilters.h"
#include "constants.h"
#include "myLibrary/iterate.h"
#include "Hysteresis.h"
#include "Variable.h"

class Sensors;

#define RATED_TEMP      25.       // Temperature at NOM_UNIT_CAP, deg C (25)
#define TCHARGE_DISPLAY_DEADBAND  0.1 // Inside this +/- deadband, charge time is displayed '---', A
#define T_RLIM         0.00085    // Temperature sensor rate limit to minimize jumps in Coulomb counting, deg C/s (0.00085 allows 0.05 deg for 1 minute)
const float VB_DC_DC = 13.5;      // DC-DC charger estimated voltage, V (13.5 < v_sat = 13.85)
#ifndef EKF_CONV  // allow override in config file
  #define EKF_CONV        1.5e-3    // EKF tracking error indicating convergence, V (1.5e-3)
#endif
#define EKF_T_CONV      30.       // EKF set convergence test time, sec (30.)
const float EKF_T_RES = (EKF_T_CONV/2.); // EKF reset retest time, sec ('up 1, down 2')
#ifndef VOC_STAT_FILT  // allow override in config file
  #define VOC_STAT_FILT 120.  // voc_stat_f_ filtering for EKF (120) VF
#endif
#ifndef EKF_Q_SD_NORM  // allow override in config file
  #define EKF_Q_SD_NORM   0.0015    // Standard deviation of normal EKF process uncertainty, V (0.0015)
#endif
#ifndef EKF_R_SD_NORM  // allow override in config file
  #define EKF_R_SD_NORM   0.5       // Standard deviation of normal EKF state uncertainty, fraction (0-1) (0.5)
#endif
#define NOM_DT          0.1       // Nominal update time, s (initialization; actual value varies)
#define EKF_NOM_DT      0.1       // EKF nominal update time, s (initialization; actual value varies)
#ifndef EKF_EFRAME_MULT  // allow override in config file
  #define EKF_EFRAME_MULT 20        // Multiframe rate consistent with READ_DELAY (20 for READ_DELAY=100) ED
#endif
#define DF2             1.2       // Threshold to resest Coulomb Counter if different from ekf, fraction (0.20)
#define TAU_Y_FILT      5.        // EKF y-filter time constant, sec (5.)
#define MIN_Y_FILT      -0.5      // EKF y-filter minimum, V (-0.5)
#define MAX_Y_FILT      0.5       // EKF y-filter maximum, V (0.5)
#define WN_Y_FILT       0.1       // EKF y-filter-2 natural frequency, r/s (0.1)
#define ZETA_Y_FILT     0.9       // EKF y-fiter-2 damping factor (0.9)
#define TMAX_FILT       3.        // Maximum y-filter-2 sample time, s (3.)
#define SOLV_ERR        1e-9      // EKF initialization solver error bound, V (1e-6)
#define SOLV_MAX_COUNTS 30        // EKF initialization solver max iters (30)
#define SOLV_SUCC_COUNTS 6        // EKF initialization solver iters to switch from successive approximation to Newton-Rapheson (6)
#define SOLV_MAX_STEP   0.2       // EKF initialization solver max step size of soc, fraction (0.2)
#define HYS_INIT_COUNTS 30        // Maximum initialization iterations hysteresis (50)
#define HYS_INIT_TOL    1e-8      // Initialization tolerance hysteresis (1e-8)
// const float MXEPS = 1-1e-6;       // Level of soc that indicates mathematically saturated (threshold is lower for robustness) (1-1e-6) dag 8/3/2023
const float MXEPS = 1.05;         // Level of soc that indicates mathematically saturated (threshold is higher to catch volt failure modes) (1.05)

#define HYS_SOC_MIN_MARG 0.15     // Add to soc_min to set thr for detecting low endpoint condition for reset of hysteresis (0.15)
#define HYS_IB_THR      1.0       // Ignore reset if opposite situation exists, A (1.0)
#ifndef VM
  #define VM 0.0
#endif
#ifndef VS
  #define VS 0.0
#endif
#ifndef VTAB_BIAS
  #define VTAB_BIAS 0.0
#endif

// Battery Class
class Battery : public Coulombs
{
public:
  Battery();
  Battery(double *sp_delta_q, const float d_voc_soc, const float dx_voc, const float dy_voc,
                const float dz_voc);
  ~Battery();
  virtual double calc_soc_voc(const double soc, const double Tb_f, double *dv_dsoc);
  double calc_soc_voc_slope(double soc, double Tb_f);
  float calc_vsat(void);
  virtual float calculate(const double Tb_f, const double soc_frac, float curr_in, const double dt, const bool dc_dc_on);
  float chargeTransfer_a() { return ChargeTransfer_->a(); };
  float chargeTransfer_b() { return ChargeTransfer_->b(); };
  float chargeTransfer_c() { return ChargeTransfer_->c(); };
  float chargeTransfer_lstate() { return ChargeTransfer_->lstate(); };
  float chargeTransfer_rstate() { return ChargeTransfer_->rstate(); };
  float chargeTransfer_T() { return ChargeTransfer_->T(); };
  float chargeTransfer_tau() { return ChargeTransfer_->tau(); };
  double cTime() { return ctime_; };
  bool bms_off() { return bms_off_; };
  float C_rate() { return ib_ / NOM_UNIT_CAP; }
  String decode(const uint8_t mod);
  float dqdt() { return chem_.dqdt; };
  float dt() { return dt_; };
  double dv_dsoc() { return dv_dsoc_; };
  float dv_dyn() { return dv_dyn_; };
  float dv_hys() { return dv_hys_; };
  float ib() { return ib_; };
  float ib_dyn() { return ib_dyn_; };
  float ib_dyn_lstate() { return ChargeTransfer_->lstate(); };
  float ib_dyn_r() { return ChargeTransfer_->reset(); };
  float ib_dyn_rstate() { return ChargeTransfer_->rstate(); };
  float ib_dyn_T() { return ChargeTransfer_->T(); };
  bool initializing() { return initializing_; };
  float ibs() { return ibs_; };
  float ioc() { return ioc_; };
  virtual void pretty_print();
  void print_signal(const bool print) { print_now_ = print; };
  double Tb() { return Tb_; };
  double Tb_f() { return Tb_f_; };
  float vb() { return vb_; };
  double voc() { return voc_; };
  double voc_soc() { return voc_soc_; };
  double voc_soc_tab(const double soc, const double Tb_f);
  double voc_stat() { return voc_stat_; };
  bool voltage_low() { return voltage_low_; };
  float vsat() { return vsat_; };
protected:
  bool bms_charging_; // Indicator that battery is charging, T = charging, changing soc and voltage
  bool bms_off_;// Indicator that battery management system is off, T = off preventing current flow
  double ctime_;   // Current time, s
  float dt_;       // Update time, s
  double dv_dsoc_; // Derivative scaled, V/fraction
  float dv_dyn_;   // ib-induced back emf, V
  float dv_hys_;   // Hysteresis state, voc-voc_out, V
  float ib_;       // Battery terminal current, A
  float ibs_;      // Hysteresis input current, A
  float ib_dyn_;   // ib lagged by charge transfer, A
  bool initializing_;  // Flag to indicate initializingn
  float ioc_;      // Hysteresis output current, A
  float nom_vsat_; // Nominal saturation threshold at 25C, V
  bool print_now_; // Print command
  bool soft_reset_print_;  // Soft reset flag
  double Tb_;     // Battery temperature, deg C
  double Tb_f_;   // Filtered battery temperature, deg C
  float vb_;       // Battery terminal voltage, V
  double voc_;      // Static model open circuit voltage, V
  double voc_soc_;      // Raw table lookup of voc, V
  double voc_stat_; // Static, table lookup value of voc before applying hysteresis, V
  bool voltage_low_; // Battery below BMS, T = BMS will turn off
  float vsat_;     // Saturation threshold at temperature, V
  // EKF declarations
  LagExp *ChargeTransfer_; // ChargeTransfer model {ib, vb} --> {voc}, ioc=ib for Battery version
                        // ChargeTransfer model {ib, voc} --> {vb}, ioc=ib for BatterySim version
  double *rand_A_;  // ChargeTransfer model A
  double *rand_B_;  // ChargeTransfer model B
  double *rand_C_;  // ChargeTransfer model C
  double *rand_D_;  // ChargeTransfer model D
};


// BatteryMonitor: extend Battery to use as monitor object
class BatteryMonitor: public Battery, public EKF_1x1
{
public:
  BatteryMonitor(const float dx_voc, const float dy_voc, const float dz_voc);
  ~BatteryMonitor();
  float amp_hrs_remaining_ekf() { return amp_hrs_remaining_ekf_; };
  float amp_hrs_remaining_soc() { return amp_hrs_remaining_soc_; };
  float calc_charge_time(const double q, const double q_capacity, const float charge_curr, const double soc);
  virtual double calc_soc_voc(const double soc, const double Tb_f, double *dv_dsoc);
  float calculate(Sensors *Sen, const bool reset,  const bool reset_ekf);
  bool converged_ekf() { return ekf_conv_; };
  double delta_q_ekf() { return delta_q_ekf_; };
  double delta_q_ekf_;         // Charge deficit represented by charge calculated by ekf, C
  float dv_dyn() { return dv_dyn_; };
  double hx() { return hx_; };
  float ib_charge() { return ib_charge_; };
  void init_battery_mon(const bool reset, Sensors *Sen);
  void init_soc_ekf(const double soc);
  bool is_sat(const bool reset);
  double K_ekf() { return K_; };
  void pretty_print(Sensors *Sen);
  void regauge(const double Tb_f, Sensors *Sen);
  float r_sd ();
  float r_ss ();
  double soc_ekf() { return soc_ekf_; };
  bool solve_ekf(const bool reset, const bool reset_temp, Sensors *Sen);
  float tcharge() { return tcharge_; };
  float vb_model_rev() { return vb_model_rev_; };
  float voc_dead() { return voc_dead_; };
  float voc_stat_f() { return voc_stat_f_; };
  float vocStatFilt_a() { return VocStatFilt->a(); };
  float vocStatFilt_b() { return VocStatFilt->b(); };
  float vocStatFilt_c() { return VocStatFilt->c(); };
  float vocStatFilt_lstate() { return VocStatFilt->lstate(); };
  float vocStatFilt_rstate() { return VocStatFilt->rstate(); };
  float vocStatFilt_T() { return VocStatFilt->T(); };
  float vocStatFilt_tau() { return VocStatFilt->tau(); };
  double y_ekf() { return y_ekf_; };
  float y_ekf_f() { return y_ekf_f_; };
  float y_ekf_f_T() { return y_ekf_f_T_; };
  float y_ekf_f_tau() { return y_ekf_f_tau_; };
  float y_ekf_f_lstate() { return y_ekf_f_lstate_; };
protected:
  LagExp *Yfilt = new LagExp(EKF_NOM_DT, TAU_Y_FILT, MIN_Y_FILT, MAX_Y_FILT);  // actual update time provided run time
  SlidingDeadband *SdVb_;  // Sliding deadband filter for Vb
  TFDelay *EKF_converged;  // Time persistence
  Iterator *ice_;      // Iteration control for EKF solver
  LagExp *VocStatFilt = new LagExp(EKF_NOM_DT, VOC_STAT_FILT, VB_MIN, VB_MAX);  // actual update time provided run time
  float amp_hrs_remaining_ekf_;  // Discharge amp*time left if drain to q_ekf=0, A-h
  float amp_hrs_remaining_soc_;  // Discharge amp*time left if drain soc_ to 0, A-h
  uint8_t eframe_;     // Counter to run EKF slower than Coulomb Counter and ChargeTransfer models
  bool ekf_conv_;   // Check that EKF error is within tolerance (T=converged)
  float ib_charge_;    // Current input avaiable for charging, A
  float ib_past_;      // Past value of current to synchronize e_wrap dynamics with model, A
  double q_ekf_;       // Filtered charge calculated by ekf, C
  double soc_ekf_;     // Filtered state of charge from ekf (0-1)
  float tcharge_;      // Counted charging time to 100%, hr
  float tcharge_ekf_;  // Solved charging time to 100% from ekf, hr
  float vb_model_rev_; // Reversionary model of vb, V
  float voc_dead_;     // Deadband-filtered, static model open circuit voltage, V
  float voc_stat_f_;   // Filtered voc_stat for EKF use, V
  double y_ekf_;       // EKF y value, V
  float y_ekf_f_;      // Filtered EKF y value, V
  void ekf_predict(double *Fx_, double *Bu_);
  void ekf_update(double *hx, double *H, double *x, double *Tb_f);
  float y_ekf_f_T_;        // EKF filter
  float y_ekf_f_tau_;      // EKF filter
  float y_ekf_f_lstate_;   // EKF filter
};


// BatterySim: extend Battery to use as model object
class BatterySim: public Battery
{
public:
  BatterySim(const float dx_voc, const float dy_voc, const float dz_voc);
  ~BatterySim();
  float calc_inj(const uint64_t now, const uint8_t type, const float amp, const double freq);
  virtual double calc_soc_voc(const double soc, const double Tb_f, double *dv_dsoc);
  float calculate(Sensors *Sen, const bool dc_dc_on, const bool reset);
  double count_coulombs(Sensors *Sen, const bool reset, BatteryMonitor *Mon, const bool initializing_all);
  bool cutback() { return model_cutback_; };
  double delta_q() { return *sp_delta_q_; };
  double d_delta_q_s() { return d_delta_q_s_; };
  uint32_t dt_long(void) { return sample_time_ - sample_time_z_; };
  float hys_state() { return hys_->dv_hys(); };
  void hys_state(const float st) { hys_->dv_hys(st); };
  void hys_pretty_print () { hys_->pretty_print(0., 0., 0.); };
  float ib_charge() { return ib_charge_; };
  float ib_fut() { return ib_fut_; };
  float ib_in() { return ib_in_; };
  float ib_s() { return ib_; };
  void init_battery_sim(const bool reset, Sensors *Sen);
  void init_hys(const float hys) { hys_->init(hys); };
  void pretty_print(void);
  bool saturated() { return model_saturated_; };
  uint32_t sample_time(void) { return sample_time_; };
  double voc() { return voc_; };
  double voc_stat() { return voc_stat_; };
protected:
  SinInj *Sin_inj_;         // Class to create sine waves
  SqInj *Sq_inj_;           // Class to create square waves
  TriInj *Tri_inj_;         // Class to create triangle waves
  CosInj *Cos_inj_;         // Class to create cosine waves
  uint32_t duty_;           // Used in Test Mode to inject Fake shunt current (0 - uint32_t(255))
  double d_delta_q_s_;       // Charge rate, C/s
  float ib_charge_;         // Current input avaiable for charging, A
  float ib_fut_;            // Future value of limited current, A
  float ib_in_;             // Saved value of current input, A
  float ib_sat_;            // Threshold to declare saturation.  This regeneratively slows down charging so if too small takes too long, A
  bool model_cutback_;   // Indicate that modeled current being limited on saturation cutback, T = cutback limited
  bool model_saturated_; // Indicator of maximal cutback, T = cutback saturated
  double q_;                // Charge, C
  uint32_t sample_time_;       // Exact moment of hardware signal generation, ms
  uint32_t sample_time_z_;     // Exact moment of past hardware signal generation, ms
  float sat_cutback_gain_; // Gain to retard ib when voc exceeds vsat, dimensionless
  float sat_ib_max_;       // Current cutback to be applied to modeled ib output, A
  float sat_ib_null_;      // Current cutback value for voc=vsat, A
  Hysteresis *hys_;
};


// Methods