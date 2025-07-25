//
// MIT License
//
// Copyright (C) 2023 - Dave Gutz
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


#ifndef BATTERY_H_
#define BATTERY_H_

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
#define T_RLIM          0.017     // Temperature sensor rate limit to minimize jumps in Coulomb counting, deg C/s (0.017 allows 1 deg for 1 minute)
const float VB_DC_DC = 13.5;      // DC-DC charger estimated voltage, V (13.5 < v_sat = 13.85)
#ifndef EKF_CONV  // allow override in config file
  #define EKF_CONV        1.5e-3    // EKF tracking error indicating convergence, V (1.5e-3)
#endif
#define EKF_T_CONV      30.       // EKF set convergence test time, sec (30.)
const float EKF_T_RESET = (EKF_T_CONV/2.); // EKF reset retest time, sec ('up 1, down 2')
#ifndef VOC_STAT_FILT  // allow override in config file
  #define VOC_STAT_FILT 15.  // voc_stat_f_ filtering for EKF
#endif
#ifndef EKF_Q_SD_NORM  // allow override in config file
  #define EKF_Q_SD_NORM   0.0015    // Standard deviation of normal EKF process uncertainty, V (0.0015)
#endif
#ifndef EKF_R_SD_NORM  // allow override in config file
  #define EKF_R_SD_NORM   0.5       // Standard deviation of normal EKF state uncertainty, fraction (0-1) (0.5)
#endif
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
#define SOLV_ERR        1e-6      // EKF initialization solver error bound, V (1e-6)
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
  Battery(double *sp_delta_q, float *sp_t_last, const float d_voc_soc, const float dx_voc, const float dy_voc,
                const float dz_voc);
  ~Battery();
  // operators
  // functions
  boolean bms_off() { return bms_off_; };
  virtual float calc_soc_voc(const float soc, const float temp_c, float *dv_dsoc);
  float calc_soc_voc_slope(float soc, float temp_c);
  float calc_vsat(void);
  virtual float calculate(const float temp_C, const float soc_frac, float curr_in, const double dt, const boolean dc_dc_on);
  float C_rate() { return ib_ / NOM_UNIT_CAP; }
  String decode(const uint8_t mod);
  float dqdt() { return chem_.dqdt; };
  float dv_dsoc() { return dv_dsoc_; };
  float dv_dyn() { return dv_dyn_; };
  float ib() { return ib_; };            // Battery terminal current, A
  float ibs() { return ibs_; };          // Hysteresis input current, A
  float ioc() { return ioc_; };          // Hysteresis output current, A
  virtual void pretty_print();
  void print_signal(const boolean print) { print_now_ = print; };
  float temp_c() { return temp_c_; };    // Battery temperature, deg C
  float Tb() { return temp_c_; };        // Battery bank temperature, deg C
  float vb() { return vb_; };            // Battery terminal voltage, V
  float voc() { return voc_; };
  float voc_stat() { return voc_stat_; };
  float voc_soc_tab(const float soc, const float temp_c);
  float vsat() { return vsat_; };
protected:
  boolean bms_charging_; // Indicator that battery is charging, T = charging, changing soc and voltage
  boolean bms_off_; // Indicator that battery management system is off, T = off preventing current flow
  float dt_;       // Update time, s
  float dv_dsoc_;  // Derivative scaled, V/fraction
  float dv_dyn_;   // ib-induced back emf, V
  float dv_hys_;   // Hysteresis state, voc-voc_out, V
  float ib_;       // Battery terminal current, A
  float ibs_;      // Hysteresis input current, A
  float ioc_;      // Hysteresis output current, A
  float nom_vsat_; // Nominal saturation threshold at 25C, V
  boolean print_now_; // Print command
  float temp_c_;    // Battery temperature, deg C
  float vb_;       // Battery terminal voltage, V
  float voc_;      // Static model open circuit voltage, V
  float voc_stat_; // Static, table lookup value of voc before applying hysteresis, V
  boolean voltage_low_; // Battery below BMS, T = BMS will turn off
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
  // operators
  // functions
  float amp_hrs_remaining_ekf() { return amp_hrs_remaining_ekf_; };
  float amp_hrs_remaining_soc() { return amp_hrs_remaining_soc_; };
  float calc_charge_time(const double q, const float q_capacity, const float charge_curr, const float soc);
  virtual float calc_soc_voc(const float soc, const float temp_c, float *dv_dsoc);
  float calculate(Sensors *Sen, const boolean reset);
  boolean converged_ekf() { return EKF_converged->state(); };
  double delta_q_ekf() { return delta_q_ekf_; };
  float hx() { return hx_; };
  float ib_charge() { return ib_charge_; };
  void init_battery_mon(const boolean reset, Sensors *Sen);
  void init_soc_ekf(const float soc);
  boolean is_sat(const boolean reset);
  float K_ekf() { return K_; };
  void pretty_print(Sensors *Sen);
  void regauge(const float temp_c);
  float r_sd ();
  float r_ss ();
  float soc_ekf() { return soc_ekf_; };
  boolean solve_ekf(const boolean reset, const boolean reset_temp, Sensors *Sen);
  float tcharge() { return tcharge_; };
  float dv_dyn() { return dv_dyn_; };
  float vb_model_rev() { return vb_model_rev_; };
  float voc_filt() { return voc_filt_; };
  float voc_soc() { return voc_soc_; };
  float voc_stat_f() { return voc_stat_f_; };
  double y_ekf() { return y_; };
  double y_ekf_filt() { return y_filt_; };
  double delta_q_ekf_;         // Charge deficit represented by charge calculated by ekf, C
protected:
  LagTustin *y_filt = new LagTustin(2., WRAP_ERR_FILT, -MAX_WRAP_ERR_FILT, MAX_WRAP_ERR_FILT);  // actual update time provided run time
  SlidingDeadband *SdVb_;  // Sliding deadband filter for Vb
  TFDelay *EKF_converged;  // Time persistence
  RateLimit *T_RLim = new RateLimit();
  Iterator *ice_;      // Iteration control for EKF solver
  LagTustin *voc_stat_filt = new LagTustin(EKF_NOM_DT, VOC_STAT_FILT, VB_MIN, VB_MAX);  // actual update time provided run time
  float amp_hrs_remaining_ekf_;  // Discharge amp*time left if drain to q_ekf=0, A-h
  float amp_hrs_remaining_soc_;  // Discharge amp*time left if drain soc_ to 0, A-h
  double dt_eframe_;   // Update time for EKF major frame
  uint8_t eframe_;     // Counter to run EKF slower than Coulomb Counter and ChargeTransfer models
  float ib_charge_;    // Current input avaiable for charging, A
  float ib_past_;      // Past value of current to synchronize e_wrap dynamics with model, A
  double q_ekf_;       // Filtered charge calculated by ekf, C
  float soc_ekf_;      // Filtered state of charge from ekf (0-1)
  float tcharge_;      // Counted charging time to 100%, hr
  float tcharge_ekf_;  // Solved charging time to 100% from ekf, hr
  float vb_model_rev_; // Reversionary model of vb, V
  float voc_filt_;     // Filtered, static model open circuit voltage, V
  float voc_soc_;      // Raw table lookup of voc, V
  float voc_stat_f_;   // Filtered voc_stat for EKF use, V
  float y_filt_;       // Filtered EKF y value, V
  void ekf_predict(double *Fx, double *Bu);
  void ekf_update(double *hx, double *H);
};


// BatterySim: extend Battery to use as model object
class BatterySim: public Battery
{
public:
  BatterySim(const float dx_voc, const float dy_voc, const float dz_voc);
  ~BatterySim();
  // operators
  // functions
  float calculate(Sensors *Sen, const boolean dc_dc_on, const boolean reset);
  float calc_inj(const unsigned long long now, const uint8_t type, const float amp, const double freq);
  virtual float calc_soc_voc(const float soc, const float temp_c, float *dv_dsoc);
  float count_coulombs(Sensors *Sen, const boolean reset, BatteryMonitor *Mon, const boolean initializing_all);
  boolean cutback() { return model_cutback_; };
  double delta_q() { return *sp_delta_q_; };
  unsigned long int dt(void) { return sample_time_ - sample_time_z_; };
  void hys_pretty_print () { hys_->pretty_print(0., 0., 0.); };
  float hys_state() { return hys_->dv_hys(); };
  void hys_state(const float st) { hys_->dv_hys(st); };
  void init_hys(const float hys) { hys_->init(hys); };
  float ib_charge() { return ib_charge_; };
  float ib_fut() { return ib_fut_; };
  void init_battery_sim(const boolean reset, Sensors *Sen);
  void pretty_print(void);
  unsigned long int sample_time(void) { return sample_time_; };
  boolean saturated() { return model_saturated_; };
  float t_last() { return *sp_t_last_; };
  float voc() { return voc_; };
  float voc_stat() { return voc_stat_; };
protected:
  SinInj *Sin_inj_;         // Class to create sine waves
  SqInj *Sq_inj_;           // Class to create square waves
  TriInj *Tri_inj_;         // Class to create triangle waves
  CosInj *Cos_inj_;         // Class to create cosine waves
  uint32_t duty_;           // Used in Test Mode to inject Fake shunt current (0 - uint32_t(255))
  float ib_charge_;         // Current input avaiable for charging, A
  float ib_fut_;            // Future value of limited current, A
  float ib_in_;             // Saved value of current input, A
  float ib_sat_;            // Threshold to declare saturation.  This regeneratively slows down charging so if too small takes too long, A
  boolean model_cutback_;   // Indicate that modeled current being limited on saturation cutback, T = cutback limited
  boolean model_saturated_; // Indicator of maximal cutback, T = cutback saturated
  double q_;                // Charge, C
  unsigned long int sample_time_;       // Exact moment of hardware signal generation, ms
  unsigned long int sample_time_z_;     // Exact moment of past hardware signal generation, ms
  float sat_cutback_gain_; // Gain to retard ib when voc exceeds vsat, dimensionless
  float sat_ib_max_;       // Current cutback to be applied to modeled ib output, A
  float sat_ib_null_;      // Current cutback value for voc=vsat, A
  Hysteresis *hys_;
};


// Methods

#endif