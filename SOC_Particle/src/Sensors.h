//
// MIT License
//
// Copyright (C) 2024 - Dave Gutz
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

#include "myLibrary/myFilters.h"
#include "Battery.h"
#include "constants.h"
#include "Cloud.h"
#include "talk/chitchat.h"
#include "command.h"
#include "Sync.h"
#include "parameters.h"
#include "Fault.h"
#include "myLibrary/KF_1x1.h"

// AD
extern PublishPars pp;  // For publishing
extern CommandPars cp;  // Various parameters to be static at system level
extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level
struct Pins;

// #ifdef HDWE_IB_HI_LO
//   #define IB_SEL_STAT_DEF 0
//   #define TB_SEL_STAT_DEF 1
//   #define VB_SEL_STAT_DEF 1
// #else
//   #define IB_SEL_STAT_DEF 1
//   #define TB_SEL_STAT_DEF 1
//   #define VB_SEL_STAT_DEF 1
// #endif


// DS18-based temp sensor
class TempSensor
{
public:
  TempSensor();
  TempSensor(const uint16_t pin, const bool parasitic, const uint16_t conversion_delay);
  TempSensor(const uint16_t pin, const bool parasitic, const uint16_t conversion_delay, const uint16_t VTb_pin);
  ~TempSensor();
  // operators
  // functions
  boolean tb_stale_flt() { return tb_stale_flt_; };
  unsigned long long sample_time() { return sample_time_; };
  float sample(Sensors *Sen);
  float noise();
  float Tb_volt(){ return Tb_volt_; };
protected:
  SlidingDeadband *SdTb;
  boolean tb_stale_flt_;  // One-wire did not update last pass
  uint16_t VTb_pin_;      // Using 2wire
  double Tb_volt_;              // Sensed battery temp voltage from ADC, V
  unsigned long long sample_time_;  // Sample time
};


// Current reading class
class Shunt
{
public:
  Shunt();
  Shunt(const String name, const uint8_t port, float *sp_ib_scale, float *sp_Ib_bias, const float v2a_s,
    const uint8_t vc_pin, const uint8_t vo_pin, const uint8_t vh3v3_pin, const boolean using_opAmp, const boolean using_kf);
  ~Shunt();
  // operators
  // functions
  boolean bare_shunt() { return ( bare_shunt_ ); };
  void dscn_cmd(const boolean cmd) { dscn_cmd_ = cmd; };
  unsigned long long dt_ms() { return sample_time_ - sample_time_z_; }; // ms
  void convert(const boolean disconnect, const boolean reset, Sensors *Sen);
  float get_v() { return KF_->get_v(); };
  float Ishunt_cal() { return Ishunt_cal_; };
  float ishunt_cal() { return Ishunt_cal_ / ap.nP(); };
  float Ishunt_cal_kf() { return Ishunt_cal_kf_; };
  float ishunt_cal_kf() { return Ishunt_cal_kf_ / ap.nP(); };
  void kf_q_std(const double q) {KF_->q_std(q);};
  void kf_r_std(const double r) {KF_->r_std(r);};
  void print_serial_header(const char suffix);
  void print_serial();
  void pretty_print();
  void sample(const boolean reset_kf);
  void sample_combine();
  void sample_filter_kf(const boolean reset_kf);
  void sample_Vc();
  void sample_Vo();
  float scale() { return ( *sp_ib_scale_ ); };
  unsigned long long sample_time(void) { return sample_time_; };
  float v2a_s() { return v2a_s_ ; };
  float vshunt() { return vshunt_; };
  int16_t vshunt_int() { return vshunt_int_; };
  int16_t vshunt_int_0() { return vshunt_int_0_; };
  int16_t vshunt_int_1() { return vshunt_int_1_; };
  float Vc() { return Vc_; };
  float Vo() { return Vo_; };
  float Vo_Vc() { return Vo_Vc_; };
  float Vo_Vc_kf() { return vshunt_kf_; };
protected:
  String name_;         // For print statements, multiple instances
  uint8_t port_;        // Octal I2C port used by Acafruit_ADS1015
  boolean bare_shunt_;  // If ADS to be ignored
  float v2a_s_;         // Selected shunt conversion gain, A/V
  int16_t vshunt_int_;  // Sensed shunt voltage, count
  int16_t vshunt_int_0_;// Interim conversion, count
  int16_t vshunt_int_1_;// Interim conversion, count
  float vshunt_;        // Sensed shunt voltage, V
  float vshunt_kf_;     // Sensed kalman filtered shunt voltage, V
  float Ishunt_cal_;    // Sensed bank current, calibrated ADC, A
  float Ishunt_cal_kf_; // Sensed kalman filtered bank current, calibrated ADC, A
  float *sp_ib_bias_;   // Global bias, A
  float *sp_ib_scale_;  // Global scale, A
  boolean reset_;       // Status of reset command input
  unsigned long long sample_time_;   // Exact moment of hardware sample, ms
  unsigned long long sample_time_z_; // Exact moment of past hardware sample, ms
  boolean dscn_cmd_;    // User command to ignore hardware, T=ignore
  uint8_t vc_pin_;      // Common voltage pin, for !HDWE_ADS1013_AMP_NOA
  uint8_t vo_pin_;      // Output voltage pin, for !HDWE_ADS1013_AMP_NOA
  uint8_t vr_pin_;      // Reference voltage pin, for TSC1200 or INA181
  int Vc_raw_;          // Raw analog read, integer       
  float Vc_;            // Sensed Vc, common op amp voltage ref, V
  int Vo_raw_;          // Raw analog read, integer       
  float Vo_;            // Sensed Vo, output of op amp, V
  float Vo_Vc_;         // Sensed Vo-Vc, difference in output of op amps, V
  boolean using_opamp_; // Using differential hardware amp
  boolean using_kf_;    // Using Kalman Filter.  If not, set filter = input
  KalmanFilter *KF_;    // Noise filter
};


// Sensors (like a big struct with public access)
class Sensors
{
public:
  Sensors();
  Sensors(double T, double T_temp, Pins *pins, Sync *ReadSensors, Sync *ReadTemp, Sync *Talk, Sync *Summarize,
    unsigned long long time_now, unsigned long long millis, BatteryMonitor *Mon);
  ~Sensors();
  int Vb_raw;                 // Raw analog read, integer
  float Vb;                   // Selected battery bank voltage, V
  float Vb_f;                 // Selected filtered battery bank voltage, V
  float Vb_hdwe;              // Sensed battery bank voltage, V
  float Vb_hdwe_f;            // Sensed, filtered battery bank voltage, V
  float Vb_model;             // Modeled battery bank voltage, V
  float Vb_volt;              // Sensed battery bank voltage at ADC, V
  float Vc;                   // Selected common reference voltage, V
  float Vc_hdwe;              // Sensed common reference voltage, V
  double Tb;                  // Selected battery bank temp, C
  double Tb_f;                // Selected filtered battery bank temp, C
  double Tb_f_rate;           // Selected filtered battery bank temp rate, C/s
  double Tb_hdwe;             // Sensed battery temp, C
  double Tb_hdwe_filt;        // Filtered, sensed battery temp, C
  double Tb_hdwe_filt_rate;   // Filtered, sensed battery temp, C/s
  double Tb_model;            // Temperature used for battery bank temp in model, C
  double Tb_model_filt;       // Filtered, modeled battery bank temp, C
  double Tb_model_filt_rate;  // Filtered, modeled battery bank temp rate, C/s
  float Ib;                   // Selected battery bank current, A
  float Ib_f;                 // Selected filtered battery bank current, A
  float Ib_amp;               // Initial selected amp battery bank current, A
  float Ib_amp_hdwe;          // Sensed amp battery bank current, A
  float Ib_amp_hdwe_f;        // Sensed, filtered amp battery bank current, A
  float Ib_amp_hdwe_kf;       // Sensed, kalman filtered amp battery bank current, A
  float Ib_amp_model;         // Modeled amp battery bank current, A
  float Ib_amp_rms;           // Amp battery bank current noise RMS, A
  float Ib_hdwe_f;            // Sensed, selected filtered battery bank current, A
  float Ib_hdwe_kf;           // Sensed, selected kalman filtered battery bank current, A
  float Ib_hdwe_f_cal;        // Sensed, filtered selected battery bank current for cal display, A
  float Ib_noa;               // Initial selected noa battery bank current, A
  float Ib_noa_hdwe;          // Sensed noa battery bank current, A
  float Ib_noa_hdwe_f;        // Sensed, filtered noa battery bank current, A
  float Ib_noa_hdwe_kf;       // Sensed, kalman filtered noa battery bank current, A
  float Ib_noa_rms;           // Noa battery bank current noise RMS, A
  float Ib_noa_model;         // Modeled noa battery bank current, A
  float Ib_hdwe;              // Sensed battery bank current, A
  float Ib_hdwe_model;        // Selected model hardware signal, A
  float Ib_model;             // Modeled battery bank current, A
  float Ib_model_in;          // Battery bank current input to model (modified by cutback), A
  float Vb_rms;               // Battery bank voltage noise RMS, V
  float Vc_rms;               // Battery bank voltage noise RMS, V
  float Wb;                   // Sensed battery bank power, use to compare to other shunts, W
  unsigned long long now;     // Time at sample, ms
  unsigned long long now_temp;// Time at sample, ms
  double T;                   // Update time, s
  boolean reset;              // Reset flag, T = reset
  double T_filt;              // Filter update time, s
  double T_temp;              // Temperature update time, s
  Sync *ReadSensors;          // Handle to debug read time
  Sync *ReadTemp;             // Handle to debug read temperature time
  boolean sat;                // Battery potential saturation status based on Temp and VOC
  boolean saturated;          // Battery confirmed saturation status based on Temp and VOC
  Shunt *ShuntAmp;            // Ib sense amplified
  Shunt *ShuntNoAmp;          // Ib sense non-amplified
  TempSensor* SensorTb;       // Tb sense
  Sync *Summarize;            // Handle to debug read time
  Sync *Talk;                 // Handle to debug talk time
  LagExp* TbModelFilt;        // Linear filter for modeled Tb (with injected noise).
  LagExp* TbSenseFilt;        // Linear filter for Tb. There are 1 Hz AAFs in hardware for Vb and Ib
  SlidingDeadband *SdTb;      // Non-linear filter for Tb
  BatterySim *Sim;            // Used to model Vb and Ib.   Use Talk 'Xp?' to toggle model on/off
  unsigned long long elapsed_inj;  // Injection elapsed time, ms
  unsigned long long start_inj;// Start of calculated injection, ms
  unsigned long long stop_inj; // Stop of calculated injection, ms
  unsigned long long end_inj;  // End of print injection, ms
  double control_time;        // Decimal time, seconds since 1/1/2021
  boolean display;            // Use display
  boolean bms_off;            // Calculated by BatteryMonitor, battery off, low voltage, switched by battery management system?
  unsigned long long dt_ib(void) { return dt_ib_; }; // ms since last update of selected Ib sample
  void select_temp(BatteryMonitor *Mon);  // Make final signal selection
  void select_volt_and_current(BatteryMonitor *Mon);  // Make final signal selection
  float ib() { return Ib / ap.nP(); };                            // Battery unit current, A
  float ib_amp() { return Ib_amp / ap.nP(); };                    // Battery amp unit current, A
  float ib_amp_hdwe() { return Ib_amp_hdwe / ap.nP(); };          // Battery amp unit current, A
  float ib_amp_hdwe_f() { return Ib_amp_hdwe_f / ap.nP(); };      // Battery amp 2-pole filtered unit current, A
  float ib_amp_hdwe_kf() { return Ib_amp_hdwe_kf / ap.nP(); };    // Battery amp kalman filtered unit current, A
  float ib_amp_model() { return Ib_amp_model / ap.nP(); };        // Battery amp model unit current, A
  float ib_amp_vo_vc() { return ShuntAmp->Vo_Vc(); };             // Battery amp kalman filter input, V
  float ib_amp_vo_vc_f() { return ShuntAmp->Vo_Vc_kf(); };        // Battery amp kalman filter output, V
  float ib_hdwe() { return Ib_hdwe / ap.nP(); };                  // Battery select hardware unit current, A
  float ib_hdwe_model() { return Ib_hdwe_model / ap.nP(); };      // Battery select hardware model unit current, A
  float ib_model() { return Ib_model / ap.nP(); };                // Battery select model unit current, A
  float ib_model_in() { return Ib_model_in / ap.nP(); };          // Battery select model input unit current, A
  float ib_noa() { return Ib_noa / ap.nP(); };                    // Battery noa unit current, A
  float ib_noa_hdwe() { return Ib_noa_hdwe / ap.nP(); };          // Battery no amp unit current, A
  float ib_noa_hdwe_kf() { return Ib_noa_hdwe_kf / ap.nP(); };    // Battery no amp kalman filtered unit current, A
  float ib_noa_model() { return Ib_noa_model / ap.nP(); };        // Battery no amp model unit current, A
  float ib_noa_vo_vc() { return ShuntNoAmp->Vo_Vc(); };           // Battery no amp kalman filter input, V
  float ib_noa_vo_vc_f() { return ShuntNoAmp->Vo_Vc_kf(); };      // Battery no amp kalman filter output, V
  float Ib_amp_add();
  float Ib_amp_max();
  float Ib_amp_min();
  float Ib_noa_add();
  float Ib_noa_max();
  float Ib_noa_min();
  float Ib_amp_noise();
  float Ib_noa_noise();
  float Ib_noise();
  unsigned long long inst_millis() { return inst_millis_; };
  unsigned long long inst_time() { return inst_time_; };
  void pretty_print();
  void reset_temp(const boolean reset) { reset_temp_ = reset; };
  boolean reset_temp() { return ( reset_temp_ ); };
  unsigned long long sample_time_ib(void) { return sample_time_ib_; };
  unsigned long long sample_time_vb(void) { return sample_time_vb_; };
  void select_print(Sensors *Sen, BatteryMonitor *Mon);
  void shunt_print();         // Print selection result
  void shunt_select_initial(const boolean reset);   // Choose between shunts for model
  void temp_load_and_filter(Sensors *Sen, const boolean reset_temp);
  float Tb_noise();
  float vb() { return Vb / ap.nS(); };                            // Battery select unit voltage, V
  float vb_hdwe() { return Vb_hdwe / ap.nS(); };                  // Battery select hardware unit voltage, V
  float vb_hdwe_f() { return Vb_hdwe_f / ap.nS(); };              // Battery select hardware unit voltage filtered, V
  void vb_load(const uint16_t vb_pin, const boolean reset);       // Analog read of Vb
  float vb_model() { return (Vb_model / ap.nS()); };              // Battery select model unit voltage, V
  float Vb_add();
  float Vb_noise();
  void vb_print(void);                  // Print Vb result
  float vc_hdwe() { return Vc_hdwe; };  // Common select hardware unit voltage, V
  Fault *Flt;
  ScaleBrk *sel_brk_hdwe;                  // Active/active scale break
protected:
  LagExp *AmpFilt;      // Noise filter for calibration
  unsigned long long dt_ib_;                // Delta update of selected Ib sample, ms
  unsigned long long dt_ib_hdwe_;           // Delta update of Ib sample, ms
  RecursiveRMSMonitorFP *IbAmpRMS; // RMS noise monitor for amp
  RecursiveRMSMonitorFP *IbNoaRMS; // RMS noise monitor for noa
  void ib_choose_active_standby(void);   // Deliberate choice based on inputs and results
  void ib_choose_hi_lo(void);   // Deliberate choice based on inputs and results
  unsigned long long inst_millis_;          // millis offset to account for setup() time, ms
  unsigned long long inst_time_;            // UTC Zulu at instantiation, s
  LagExp *NoaFilt;      // Noise filter for calibration
  PRBS_7 *Prbn_Tb_;     // Tb noise generator model only
  PRBS_7 *Prbn_Vb_;     // Vb noise generator model only
  PRBS_7 *Prbn_Ib_amp_; // Ib amplified sensor noise generator model only
  PRBS_7 *Prbn_Ib_noa_; // Ib non-amplified sensor noise generator model only
  boolean reset_temp_;  // Keep track of temperature reset, stored for plotting, T=reset
  unsigned long long sample_time_ib_;       // Exact moment of selected Ib sample, ms
  unsigned long long sample_time_ib_hdwe_;  // Exact moment of Ib sample, ms
  unsigned long long sample_time_tb_;       // Exact moment of selected Tb sample, ms
  unsigned long long sample_time_vb_;       // Exact moment of selected Vb sample, ms
  unsigned long long sample_time_vb_hdwe_;  // Exact moment of Vb sample, ms
  LagExp *SelFiltCal;      // Noise filter for calibration
  LagExp *VbFilt;       // Noise filter for calibration
  RecursiveRMSMonitorFP *VbRMS; // RMS noise monitor for Vb
  RecursiveRMSMonitorFP *VcRMS; // RMS noise monitor for Vc
};
