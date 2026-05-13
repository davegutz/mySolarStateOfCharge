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


// Particle Photon 2 (P2) debounced analog read
class AnalogReadP2
{
public:
  AnalogReadP2() {};
  AnalogReadP2(const uint16_t pin): dead_(false), dead_prev_(false), lgv_(0), pin_(pin), 
    raw_(0), raw_debounced_(0), raw_low_(false), raw_low_prev_(false) {};
  ~AnalogReadP2() {};

  // operators

  // functions

  // Hold last-good-value on exactly the 2nd consecutive low read (glitch); pass raw_ otherwise.
  // 1st low: pass raw_  2nd consecutive low: return lgv_  3rd+ (persistent): pass raw_
  int32_t analogReadDebounced(const int32_t debounce_level, const bool reset, const String name)
  {
    raw_ = analogRead(pin_);
    raw_low_ = raw_ < debounce_level;
    if ( reset ) raw_low_prev_ = raw_low_;
    dead_ = raw_low_ && raw_low_prev_ && !reset;  // two consecutive low readings
    raw_low_prev_ = raw_low_;
    if ( !raw_low_ )
    {
      lgv_ = raw_;             // update last-good-value on every valid reading
      raw_debounced_ = raw_;   // pass through
    }
    else if ( dead_ && !dead_prev_ && !reset )
    {
      raw_debounced_ = lgv_;   // exactly 2nd consecutive low: hold last good value
      sendTxBuf(String::format("trip %s reset %d raw %d < %d raw_low %d dead %d raw_low_prev %d lgv_ %d raw_debounced %d \n",
        name.c_str(), reset, raw_, debounce_level, raw_low_, dead_, raw_low_prev_, lgv_, raw_debounced_), true, IN_SERVICE);
    }
    else
    {
      raw_debounced_ = raw_;   // 1st low, or persistent (3rd+): pass raw_ through
    }
    dead_prev_ = dead_;
    if ( sp.debug()==57 )
      sendTxBuf(String::format("not yet dead %s reset %d raw %d < %d raw_low %d dead %d raw_low_prev %d lgv_ %d raw_debounced %d \n",
        name.c_str(), reset, raw_, debounce_level, raw_low_, dead_, raw_low_prev_, lgv_, raw_debounced_), true, IN_SERVICE);

    return ( raw_debounced_ );
  }
  bool dead() { return dead_; }
protected:
  bool dead_;
  bool dead_prev_;
  int32_t lgv_;
  uint16_t pin_;
  int32_t raw_;
  int32_t raw_debounced_;
  bool raw_low_;
  bool raw_low_prev_;
};



// Current reading class
class Shunt
{
public:
  Shunt();
  Shunt(const String name, const uint8_t port, float *sp_ib_scale, float *sp_Ib_bias, const float v2a_s,
    const uint8_t vc_pin, const uint8_t vo_pin, const uint8_t vh3v3_pin, const bool using_opAmp, const bool using_kf);
  ~Shunt();
  // operators
  // functions
  bool bare_shunt() { return ( bare_shunt_ ); };
  void dscn_cmd(const bool cmd) { dscn_cmd_ = cmd; };
  uint64_t dt_ms() { return sample_time_ - sample_time_z_; }; // ms
  void convert(const bool disconnect, const bool reset, Sensors *Sen);
  float Ishunt_cal() { return Ishunt_cal_; };
  float ishunt_cal() { return Ishunt_cal_ / ap.nP(); };
  float Ishunt_cal_kf() { return Ishunt_cal_kf_; };
  float ishunt_cal_kf() { return Ishunt_cal_kf_ / ap.nP(); };
  void kf_q_std(const double q) {KF_->q_std(q);};
  void kf_r_std(const double r) {KF_->r_std(r);};
  float kf_v() { return KF_->kf_v(); };
  void print_serial_header(const char suffix);
  void print_serial();
  void pretty_print();
  void sample(const bool reset_kf);
  void sample_combine();
  void sample_filter_kf(const bool reset_kf);
  void sample_Vc();
  void sample_Vo();
  float scale() { return ( *sp_ib_scale_ ); };
  uint64_t sample_time(void) { return sample_time_; };
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
  bool bare_shunt_;  // If ADS to be ignored
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
  bool reset_;       // Status of reset command input
  uint64_t sample_time_;   // Exact moment of hardware sample, ms
  uint64_t sample_time_z_; // Exact moment of past hardware sample, ms
  bool dscn_cmd_;    // User command to ignore hardware, T=ignore
  uint8_t vc_pin_;      // Common voltage pin
  uint8_t vo_pin_;      // Output voltage pin
  uint8_t vr_pin_;      // Reference voltage pin, for TSC1200 or INA181
  int Vc_raw_;          // Raw analog read, integer
  float Vc_;            // Sensed Vc, common op amp voltage ref, V
  int Vo_raw_;          // Raw analog read, integer
  float Vo_;            // Sensed Vo, output of op amp, V
  float Vo_Vc_;         // Sensed Vo-Vc, difference in output of op amps, V
  bool using_opamp_; // Using differential hardware amp
  bool using_kf_;    // Using Kalman Filter.  If not, set filter = input
  KalmanFilter *KF_;    // Noise filter
  AnalogReadP2 *Vc_read_; // Debounced analog read for Vc
  AnalogReadP2 *Vo_read_; // Debounced analog read for Vo
  TFDelay *Bare_delay_;   // Persistence declaring bare_ after Vc_read_ transition
};


// Sensors (like a big struct with public access)
class Sensors
{
public:
  Sensors();
  Sensors(double T, double T_temp, Pins *pins, Sync *ReadSensors, Sync *ReadTemp, Sync *Talk, Sync *Summarize,
    uint64_t time_now, uint64_t millis, BatteryMonitor *Mon);
  ~Sensors();
  // Getters and setters for encapsulated member variables
  void cTime(const double input) { ctime_ = input; }
  double cTime() { return ctime_; }
  void cTime_temp(const double input) { ctime_temp_ = input; }
  double cTime_temp() { return ctime_temp_; }
  void Vb_raw(const int input) { Vb_raw_ = input; }
  int Vb_raw() { return Vb_raw_; }
  void Vb(const float input) { Vb_ = input; }
  float Vb() { return Vb_; }
  void Vb_f(const float input) { Vb_f_ = input; }
  float Vb_f() { return Vb_f_; }
  void Vb_hdwe(const float input) { Vb_hdwe_ = input; }
  float Vb_hdwe() { return Vb_hdwe_; }
  void Vb_hdwe_f(const float input) { Vb_hdwe_f_ = input; }
  float Vb_hdwe_f() { return Vb_hdwe_f_; }
  void Vb_model(const float input) { Vb_model_ = input; }
  float Vb_model() { return Vb_model_; }
  void Vb_volt(const float input) { Vb_volt_ = input; }
  float Vb_volt() { return Vb_volt_; }
  void Vc(const float input) { Vc_ = input; }
  float Vc() { return Vc_; }
  void Vc_hdwe(const float input) { Vc_hdwe_ = input; }
  float Vc_hdwe() { return Vc_hdwe_; }
  void Vc_hdwe_sum(const float input) { Vc_hdwe_sum_ = input; }
  float Vc_hdwe_sum() { return Vc_hdwe_sum_; }
  void Tb(const double input) { Tb_ = input; }
  double Tb() { return Tb_; }
  void Tb_f(const double input) { Tb_f_ = input; }
  double Tb_f() { return Tb_f_; }
  void Tb_f_rate(const double input) { Tb_f_rate_ = input; }
  double Tb_f_rate() { return Tb_f_rate_; }
  void Tb_hdwe(const double input) { Tb_hdwe_ = input; }
  double Tb_hdwe() { return Tb_hdwe_; }
  void Tb_hdwe_f(const double input) { Tb_hdwe_f_ = input; }
  double Tb_hdwe_f() { return Tb_hdwe_f_; }
  double Tb_hdwe_f_dt() { return Tb_hdwe_f_dt_; }
  void Tb_hdwe_f_rate(const double input) { Tb_hdwe_f_rate_ = input; }
  double Tb_hdwe_f_rate() { return Tb_hdwe_f_rate_; }
  void Tb_hdwe_f_rstate(const double input) { Tb_hdwe_f_rstate_ = input; }
  double Tb_hdwe_f_rstate() { return Tb_hdwe_f_rstate_; }
  void Tb_hdwe_f_lstate(const double input) { Tb_hdwe_f_lstate_ = input; }
  double Tb_hdwe_f_lstate() { return Tb_hdwe_f_lstate_; }
  double Tb_hdwe_f_tau() { return Tb_hdwe_f_tau_; }
  void Tb_model(const double input) { Tb_model_ = input; }
  double Tb_model() { return Tb_model_; }
  void Tb_model_f(const double input) { Tb_model_f_ = input; }
  double Tb_model_f() { return Tb_model_f_; }
  double Tb_model_f_dt() { return Tb_model_f_dt_; }
  void Tb_model_f_rate(const double input) { Tb_model_f_rate_ = input; }
  double Tb_model_f_rate() { return Tb_model_f_rate_; }
  void Tb_model_f_lstate(const double input) { Tb_model_f_lstate_ = input; }
  double Tb_model_f_lstate() { return Tb_model_f_lstate_; }
  void Tb_model_f_rstate(const double input) { Tb_model_f_rstate_ = input; }
  double Tb_model_f_rstate() { return Tb_model_f_rstate_; }
  double Tb_model_f_tau() { return Tb_model_f_tau_; }
  void Ib(const float input) { Ib_ = input; }
  float Ib() { return Ib_; }
  void Ib_f(const float input) { Ib_f_ = input; }
  float Ib_f() { return Ib_f_; }
  void Ib_amp(const float input) { Ib_amp_ = input; }
  float Ib_amp() { return Ib_amp_; }
  void Ib_amp_hdwe(const float input) { Ib_amp_hdwe_ = input; }
  float Ib_amp_hdwe() { return Ib_amp_hdwe_; }
  void Ib_amp_hdwe_f(const float input) { Ib_amp_hdwe_f_ = input; }
  float Ib_amp_hdwe_f() { return Ib_amp_hdwe_f_; }
  void Ib_amp_hdwe_kf(const float input) { Ib_amp_hdwe_kf_ = input; }
  float Ib_amp_hdwe_kf() { return Ib_amp_hdwe_kf_; }
  void Ib_amp_model(const float input) { Ib_amp_model_ = input; }
  float Ib_amp_model() { return Ib_amp_model_; }
  void Ib_amp_rms(const float input) { Ib_amp_rms_ = input; }
  float Ib_amp_rms() { return Ib_amp_rms_; }
  void Ib_hdwe_f(const float input) { Ib_hdwe_f_ = input; }
  float Ib_hdwe_f() { return Ib_hdwe_f_; }
  void Ib_hdwe_kf(const float input) { Ib_hdwe_kf_ = input; }
  float Ib_hdwe_kf() { return Ib_hdwe_kf_; }
  void Ib_hdwe_f_cal(const float input) { Ib_hdwe_f_cal_ = input; }
  float Ib_hdwe_f_cal() { return Ib_hdwe_f_cal_; }
  void Ib_noa(const float input) { Ib_noa_ = input; }
  float Ib_noa() { return Ib_noa_; }
  void Ib_noa_hdwe(const float input) { Ib_noa_hdwe_ = input; }
  float Ib_noa_hdwe() { return Ib_noa_hdwe_; }
  void Ib_noa_hdwe_f(const float input) { Ib_noa_hdwe_f_ = input; }
  float Ib_noa_hdwe_f() { return Ib_noa_hdwe_f_; }
  void Ib_noa_hdwe_kf(const float input) { Ib_noa_hdwe_kf_ = input; }
  float Ib_noa_hdwe_kf() { return Ib_noa_hdwe_kf_; }
  void Ib_noa_rms(const float input) { Ib_noa_rms_ = input; }
  float Ib_noa_rms() { return Ib_noa_rms_; }
  void Ib_noa_model(const float input) { Ib_noa_model_ = input; }
  float Ib_noa_model() { return Ib_noa_model_; }
  void Ib_hdwe(const float input) { Ib_hdwe_ = input; }
  float Ib_hdwe() { return Ib_hdwe_; }
  void Ib_hdwe_model(const float input) { Ib_hdwe_model_ = input; }
  float Ib_hdwe_model() { return Ib_hdwe_model_; }
  void Ib_model(const float input) { Ib_model_ = input; }
  float Ib_model() { return Ib_model_; }
  void Ib_model_in(const float input) { Ib_model_in_ = input; }
  float Ib_model_in() { return Ib_model_in_; }
  void Vb_rms(const float input) { Vb_rms_ = input; }
  float Vb_rms() { return Vb_rms_; }
  void Vc_rms(const float input) { Vc_rms_ = input; }
  float Vc_rms() { return Vc_rms_; }
  void Wb(const float input) { Wb_ = input; }
  float Wb() { return Wb_; }
  void now(const uint64_t input) { now_ = input; }
  uint64_t now() { return now_; }
  void now_temp(const uint64_t input) { now_temp_ = input; }
  uint64_t now_temp() { return now_temp_; }
  void T(const double input) { T_ = input; }
  double T() { return T_; }
  void reset(const bool input) { reset_ = input; }
  bool reset() { return reset_; }
  void T_filt(const double input) { T_filt_ = input; }
  double T_filt() { return T_filt_; }
  void T_temp(const double input) { T_temp_ = input; }
  double T_temp() { return T_temp_; }
  void elapsed_inj(const uint64_t input) { elapsed_inj_ = input; }
  uint64_t elapsed_inj() { return elapsed_inj_; }
  void start_inj(const uint64_t input) { start_inj_ = input; }
  uint64_t start_inj() { return start_inj_; }
  void stop_inj(const uint64_t input) { stop_inj_ = input; }
  uint64_t stop_inj() { return stop_inj_; }
  void end_inj(const uint64_t input) { end_inj_ = input; }
  uint64_t end_inj() { return end_inj_; }
  void control_time(const double input) { control_time_ = input; }
  double control_time() { return control_time_; }
  void display(const bool input) { display_ = input; }
  bool display() { return display_; }
  void bms_off(const bool input) { bms_off_ = input; }
  bool bms_off() { return bms_off_; }
  Sync *ReadSensors;          // Handle to debug read time
  Sync *ReadTemp;             // Handle to debug read temperature time
  void sat(const bool input) { sat_ = input; }
  bool sat() { return sat_; }
  void saturated(const bool input) { saturated_ = input; }
  bool saturated() { return saturated_; }
  Shunt *ShuntAmp;            // Ib sense amplified
  Shunt *ShuntNoAmp;          // Ib sense non-amplified
  Sync *Summarize;            // Handle to debug read time
  Sync *Talk;                 // Handle to debug talk time
  BatterySim *Sim;            // Used to model Vb and Ib.   Use Talk 'Xp?' to toggle model on/off
  uint64_t dt_ib(void) { return dt_ib_; }; // ms since last update of selected Ib sample
  void select_volt_and_current_and_temp(BatteryMonitor *Mon);      // Make final signal selection
  float ib() { return Ib_ / ap.nP(); };                            // Battery unit current, A
  float ib_amp() { return Ib_amp_ / ap.nP(); };                    // Battery amp unit current, A
  float ib_amp_hdwe() { return Ib_amp_hdwe_ / ap.nP(); };          // Battery amp unit current, A
  float ib_amp_hdwe_f() { return Ib_amp_hdwe_f_ / ap.nP(); };      // Battery amp 2-pole filtered unit current, A
  float ib_amp_hdwe_kf() { return Ib_amp_hdwe_kf_ / ap.nP(); };    // Battery amp kalman filtered unit current, A
  float ib_amp_model() { return Ib_amp_model_ / ap.nP(); };        // Battery amp model unit current, A
  float ib_amp_vo_vc() { return ShuntAmp->Vo_Vc(); };              // Battery amp kalman filter input, V
  float ib_amp_vo_vc_kf() { return ShuntAmp->Vo_Vc_kf(); };        // Battery amp kalman filter output, V
  float ib_hdwe() { return Ib_hdwe_ / ap.nP(); };                  // Battery select hardware unit current, A
  float ib_hdwe_model() { return Ib_hdwe_model_ / ap.nP(); };      // Battery select hardware model unit current, A
  float ib_model() { return Ib_model_ / ap.nP(); };                // Battery select model unit current, A
  float ib_model_in() { return Ib_model_in_ / ap.nP(); };          // Battery select model input unit current, A
  float ib_noa() { return Ib_noa_ / ap.nP(); };                    // Battery noa unit current, A
  float ib_noa_hdwe() { return Ib_noa_hdwe_ / ap.nP(); };          // Battery no amp unit current, A
  float ib_noa_hdwe_kf() { return Ib_noa_hdwe_kf_ / ap.nP(); };    // Battery no amp kalman filtered unit current, A
  float ib_noa_model() { return Ib_noa_model_ / ap.nP(); };        // Battery no amp model unit current, A
  float ib_noa_vo_vc() { return ShuntNoAmp->Vo_Vc(); };            // Battery no amp kalman filter input, V
  float ib_noa_vo_vc_kf() { return ShuntNoAmp->Vo_Vc_kf(); };      // Battery no amp kalman filter output, V
  float Ib_amp_add();
  float Ib_amp_max();
  float Ib_amp_min();
  float Ib_noa_add();
  float Ib_noa_max();
  float Ib_noa_min();
  float Ib_amp_noise();
  float Ib_noa_noise();
  uint64_t inst_millis() { return inst_millis_; };
  uint64_t inst_time() { return inst_time_; };
  void pretty_print();
  void reset_temp(const bool reset) { reset_temp_ = reset; };
  bool reset_temp() { return ( reset_temp_ ); };
  uint64_t sample_time_ib(void) { return sample_time_ib_; };
  uint64_t sample_time_vb(void) { return sample_time_vb_; };
  void select_print(Sensors *Sen, BatteryMonitor *Mon);
  void shunt_print();         // Print selection result
  void shunt_select_initial(const bool reset);   // Choose between shunts for model
  float Tb_noise();
  void Tb_load(const uint16_t vb_pin, const bool reset);           // Analog read of Tb
  void Tb_print(void);                                             // Print Tb result
  float vb() { return Vb_ / ap.nS(); };                            // Battery select unit voltage, V
  float vb_hdwe() { return Vb_hdwe_ / ap.nS(); };                  // Battery select hardware unit voltage, V
  float vb_hdwe_f() { return Vb_hdwe_f_ / ap.nS(); };              // Battery select hardware unit voltage filtered, V
  void vb_load(const uint16_t vb_pin, const bool reset);           // Analog read of Vb
  float vb_model() { return (Vb_model_ / ap.nS()); };              // Battery select model unit voltage, V
  float Vb_add();
  float Vb_noise();
  void vb_print(void);                  // Print Vb result
  Fault *Flt;
  ScaleBrk *sel_brk_hdwe;                  // Active/active scale break
protected:
  LagExp *AmpFilt;      // Noise filter for calibration
  uint64_t dt_ib_;                // Delta update of selected Ib sample, ms
  uint64_t dt_ib_hdwe_;           // Delta update of Ib sample, ms
  RecursiveRMSMonitorFP *IbAmpRMS; // RMS noise monitor for amp
  RecursiveRMSMonitorFP *IbNoaRMS; // RMS noise monitor for noa
  void ib_choose_active_standby(void);   // Deliberate choice based on inputs and results
  void ib_choose_hi_lo(void);   // Deliberate choice based on inputs and results
  uint64_t inst_millis_;          // millis offset to account for setup() time, ms
  uint64_t inst_time_;            // UTC Zulu at instantiation, s
  LagExp *NoaFilt;      // Noise filter for calibration
  PRBS_7 *Prbn_Tb_;     // Tb noise generator model only
  PRBS_7 *Prbn_Vb_;     // Vb noise generator model only
  PRBS_7 *Prbn_Ib_amp_; // Ib amplified sensor noise generator model only
  PRBS_7 *Prbn_Ib_noa_; // Ib non-amplified sensor noise generator model only
  bool reset_temp_;  // Keep track of temperature reset, stored for plotting, T=reset
  uint64_t sample_time_ib_;       // Exact moment of selected Ib sample, ms
  uint64_t sample_time_ib_hdwe_;  // Exact moment of Ib sample, ms
  uint64_t sample_time_Tb_;       // Exact moment of Tb sample, ms
  uint64_t sample_time_Tb_hdwe_;  // Exact moment of Tb sample, ms
  uint64_t sample_time_vb_;       // Exact moment of selected Vb sample, ms
  uint64_t sample_time_vb_hdwe_;  // Exact moment of Vb sample, ms
  LagExp *SelFiltCal;             // Noise filter for calibration
  LagExp *TbHdweFilt;                 // Noise filter for calibration
  LagExp *TbModelFilt;                // Noise filter for calibration
  LagExp *VbFilt;                 // Noise filter for calibration
  RecursiveRMSMonitorFP *VbRMS;   // RMS noise monitor for Vb
  RecursiveRMSMonitorFP *VcRMS;   // RMS noise monitor for Vc
  AnalogReadP2 *Tb_read_;      // Tb sense debounce
  AnalogReadP2 *Vb_read_;      // Vb sense debounce
  int Tb_raw_;                 // Raw analog read, integer
  float Tb_volt_;              // Sensed battery bank temperature at ADC, V
  int Vb_raw_;                 // Raw analog read, integer
  float Vb_;                   // Selected battery bank voltage, V
  float Vb_f_;                 // Selected filtered battery bank voltage, V
  float Vb_hdwe_;              // Sensed battery bank voltage, V
  float Vb_hdwe_f_;            // Sensed, filtered battery bank voltage, V
  float Vb_model_;             // Modeled battery bank voltage, V
  float Vb_volt_;              // Sensed battery bank voltage at ADC, V
  float Vc_;                   // Selected common reference voltage, V
  float Vc_hdwe_;              // Sensed common reference voltage, V
  float Vc_hdwe_sum_;          // Sensed common reference voltage sum, V
  double Tb_;                  // Selected battery bank temp, C
  double Tb_f_;                // Selected filtered battery bank temp, C
  double Tb_f_rate_;           // Selected filtered battery bank temp rate, C/s
  double Tb_hdwe_;             // Sensed battery temp, C
  double Tb_hdwe_f_;           // Filtered, sensed battery temp, C
  double Tb_hdwe_f_dt_;        // Battery hdwe temp filter update time, s
  double Tb_hdwe_f_rate_;      // Filtered, sensed battery temp rate, C/s
  double Tb_hdwe_f_rstate_;    // Filtered, sensed battery temp rate state, C/s
  double Tb_hdwe_f_lstate_;    // Filtered, sensed battery temp rate state, C/s
  double Tb_hdwe_f_tau_;       // Battery hdwe temp filter time constant, s
  float Tb_model_;             // Modeled battery bank temp, C
  float Tb_model_f_;           // Filtered, modeled battery bank temp, C
  double Tb_model_f_dt_;       // Battery model temp filter update time, s
  double Tb_model_f_rate_;     // Filtered, modeled battery bank temp rate, C/s
  double Tb_model_f_rstate_;   // Filtered, sensed battery temp rate state, C/s
  double Tb_model_f_lstate_;   // Filtered, sensed battery temp rate state, C/s
  double Tb_model_f_tau_;      // Battery model temp filter time constant, s
  float Ib_;                   // Selected battery bank current, A
  float Ib_f_;                 // Selected filtered battery bank current, A
  float Ib_amp_;               // Initial selected amp battery bank current, A
  float Ib_amp_hdwe_;          // Sensed amp battery bank current, A
  float Ib_amp_hdwe_f_;        // Sensed, filtered amp battery bank current, A
  float Ib_amp_hdwe_kf_;       // Sensed, kalman filtered amp battery bank current, A
  float Ib_amp_model_;         // Modeled amp battery bank current, A
  float Ib_amp_rms_;           // Amp battery bank current noise RMS, A
  float Ib_hdwe_f_;            // Sensed, selected filtered battery bank current, A
  float Ib_hdwe_kf_;           // Sensed, selected kalman filtered battery bank current, A
  float Ib_hdwe_f_cal_;        // Sensed, filtered selected battery bank current for cal display, A
  float Ib_noa_;               // Initial selected noa battery bank current, A
  float Ib_noa_hdwe_;          // Sensed noa battery bank current, A
  float Ib_noa_hdwe_f_;        // Sensed, filtered noa battery bank current, A
  float Ib_noa_hdwe_kf_;       // Sensed, kalman filtered noa battery bank current, A
  float Ib_noa_rms_;           // Noa battery bank current noise RMS, A
  float Ib_noa_model_;         // Modeled noa battery bank current, A
  float Ib_hdwe_;              // Sensed battery bank current, A
  float Ib_hdwe_model_;        // Selected model hardware signal, A
  float Ib_model_;             // Modeled battery bank current, A
  float Ib_model_in_;          // Battery bank current input to model (modified by cutback), A
  float Vb_rms_;               // Battery bank voltage noise RMS, V
  float Vc_rms_;               // Battery bank voltage noise RMS, V
  float Wb_;                   // Sensed battery bank power, use to compare to other shunts, W
  uint64_t now_;     // Time at sample, ms
  uint64_t now_temp_;// Time at sample, ms
  double ctime_;               // Decimal time, seconds since 1/1/2021
  double ctime_temp_;          // Decimal time at temp read, seconds since 1/1/2021
  double T_;                   // Update time, s
  bool reset_;                 // Reset flag, T = reset
  double T_filt_;              // Filter update time, s
  double T_temp_;              // Temperature update time, s
  uint64_t elapsed_inj_;  // Injection elapsed time, ms
  uint64_t start_inj_;// Start of calculated injection, ms
  uint64_t stop_inj_; // Stop of calculated injection, ms
  uint64_t end_inj_;  // End of print injection, ms
  double control_time_;        // Decimal time, seconds since 1/1/2021
  bool display_;            // Use display
  bool bms_off_;            // Calculated by BatteryMonitor, battery off, low voltage, switched by battery management system?
  bool sat_;                // Battery potential saturation status based on Temp and VOC
  bool saturated_;          // Battery confirmed saturation status based on Temp and VOC
};
