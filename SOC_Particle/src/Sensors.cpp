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

#include "application.h"
#include "command.h"
#include "Sensors.h"
#include "constants.h"
#include <math.h>
#include "debug.h"
#include "Summary.h"

extern CommandPars cp;  // Various parameters shared at system level
extern PrinterPars pr;  // Print buffer
extern PublishPars pp;  // For publishing
extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle



// class Shunt
// constructors
Shunt::Shunt()
: name_("None"), port_(0x00), bare_shunt_(false){}
Shunt::Shunt(const String name, const uint8_t port, float *sp_ib_scale,  float *sp_Ib_bias, const float v2a_s,
  const uint8_t vc_pin, const uint8_t vo_pin, const uint8_t vh3v3_pin, const bool using_opAmp, const bool using_kf)
: name_(name), port_(port), bare_shunt_(false), v2a_s_(v2a_s),
  vshunt_int_(0), vshunt_int_0_(0), vshunt_int_1_(0), vshunt_(0.), vshunt_kf_(0.), Ishunt_cal_(0.), Ishunt_cal_kf_(0.),
  sp_ib_bias_(sp_Ib_bias), sp_ib_scale_(sp_ib_scale), reset_(false), sample_time_(0UL), sample_time_z_(0UL), dscn_cmd_(false),
  vc_pin_(vc_pin), vo_pin_(vo_pin), vr_pin_(vh3v3_pin), Vc_raw_(HALF_V3V3/VH3V3_CONV_GAIN), Vc_(HALF_V3V3),
  Vo_raw_(0), Vo_(0.), Vo_Vc_(0.), using_opamp_(using_opAmp), using_kf_(using_kf)
{
  if ( using_opamp_ ) Serial.printf("Ib %s sense ADC pin %d started using OpAmp and 3V3 pin %d\n", name_.c_str(), vo_pin_, vr_pin_);
  else Serial.printf("Ib %s sense ADC pins %d and %d started\n", name_.c_str(), vo_pin_, vc_pin_);
  KF_ = new KalmanFilter(0.1, 0., KF_Q_STD, KF_R_STD);
  Vc_read_ = new AnalogReadP2(using_opamp_ ? vr_pin_ : vc_pin_);
  Vc_read_ = new AnalogReadP2(using_opamp_ ? vr_pin_ : vc_pin_);
  Vo_read_ = new AnalogReadP2(vo_pin_);
  Bare_delay_ = new TFDelay(false, RAW_BARE_SET, RAW_BARE_RES, sample_time_);
}
Shunt::~Shunt() {}
// operators
// functions

void Shunt::pretty_print()
{
#ifndef SOFT_DEPLOY_PHOTON
  Serial.printf(" reset %d;\n", reset_);
  Serial.printf(" *sp_Ib_bias%7.3f; A\n", *sp_ib_bias_);
  Serial.printf(" *sp_ib_scale%7.3f; A\n", *sp_ib_scale_);
  Serial.printf(" bare_shunt %d dscn_cmd %d\n", bare_shunt_, dscn_cmd_);
  Serial.printf(" Ishunt_cal%7.3f; A\n", Ishunt_cal_);
  Serial.printf(" Ishunt_cal_kf%7.3f; A\n", Ishunt_cal_kf_);
  Serial.printf(" port 0x%X;\n", port_);
  Serial.printf(" using_kf%d;", using_kf_);
  Serial.printf(" v2a_s%7.2f; A/V\n", v2a_s_);
  Serial.printf(" Vc%10.6f; V\n", Vc_);
  Serial.printf(" Vc_raw %d;\n", Vc_raw_);
  Serial.printf(" Vo%10.6f; V\n", Vo_);
  Serial.printf(" Vo-Vc%10.6f; V\n", Vo_Vc());
  Serial.printf(" Vo-Vc_kf%10.6f; V\n", Vo_Vc_kf());
  Serial.printf(" Vo_raw %d;\n", Vo_raw_);
  Serial.printf(" vshunt_int %d; count\n", vshunt_int_);
  Serial.printf("Shunt(%s)::\n", name_.c_str());
  if ( using_kf_ )
  {
    Serial.printf(" KF\n");
    KF_->pretty_print();
  }
  else
    Serial.printf(" not using KF\n");
#else
     Serial.printf("Shunt: silent DEPLOY\n");
#endif
}

// Convert sampled shunt data to Ib engineering units
void Shunt::convert(const bool disconnect, const bool reset, Sensors *Sen)
{
  reset_ = reset;
  #ifndef HDWE_BARE
    bare_shunt_ = Bare_delay_->calculate(Vc_read_->dead(), RAW_BARE_SET, RAW_BARE_RES, Sen->T(),reset_);
  #else
    bare_shunt_ = false;
  #endif
  if ( !bare_shunt_ && !dscn_cmd_ )
  {
    vshunt_ = Vo_Vc_;
    vshunt_int_0_ = 0; vshunt_int_1_ = 0; vshunt_int_ = 0;
  }
  else
  {
    vshunt_int_0_ = 0; vshunt_int_1_ = 0; vshunt_int_ = 0; vshunt_ = 0.; vshunt_kf_ = 0.;
    Vc_raw_ = 0; Vc_ = 0.; Vo_raw_ = 0; Vo_ = 0.;
    Ishunt_cal_ = 0.;
  }
  if ( disconnect )
  {
    Ishunt_cal_ = 0.;
    Ishunt_cal_kf_ = 0.;
  }
  else
  {
    Ishunt_cal_ = vshunt_*v2a_s_*(*sp_ib_scale_) + *sp_ib_bias_;
    Ishunt_cal_kf_ = vshunt_kf_*v2a_s_*(*sp_ib_scale_) + *sp_ib_bias_;
  }

  // One-sided scale factor if needed
  if ( Ishunt_cal_kf_ < 0. ) Ishunt_cal_ *= sp.ib_disch_slr();

}

// Sample and filter amplifier Vo-Vc
void Shunt::sample(const bool reset_kf)
{
  sample_Vo();
  sample_Vc();
  sample_combine();
  sample_filter_kf(reset_kf);
  if  ( sp.debug()==14 )Serial.printf("reset_kf %d ADCref %7.3f samp_t %lld vo_pin_%d V0_raw_%d Vo_%7.3f Vo_Vc_%7.3f vshunt_kf_%7.3f  Vc_%7.3f\n", reset_kf, (float)analogGetReference(), sample_time_, vo_pin_, Vo_raw_, Vo_, Vo_Vc_, vshunt_kf_, Vc_);
}

// Basic arithmetic
void Shunt::sample_combine()
{
  Vo_Vc_ = Vo_ - Vc_;
}

// Apply Kalman filter to Vo-Vc
void Shunt::sample_filter_kf(const bool reset_kf)
{
  if ( using_kf_ )
    vshunt_kf_ = KF_->calculate(reset_kf, dt_ms()/1000., Vo_Vc_);
  else
    vshunt_kf_ = Vo_Vc_;

}

// Sample Vc = Vr centering signal for amplifier
void Shunt::sample_Vc()
{
  Vc_raw_ = Vc_read_->analogReadDebounced(VRAW_BARE_DETECTED, reset_, name_);
  if ( using_opamp_ )
  {
    Vc_ =  float(Vc_raw_)*VH3V3_CONV_GAIN + ap.vc_add();
  }
  else
  {
    Vc_ =  float(Vc_raw_)*VC_CONV_GAIN + ap.vc_add();
  }
}

// Sample Vo output voltage of amplifier
void Shunt::sample_Vo()
{
  sample_time_z_ = sample_time_;
  sample_time_ = millis();
  Vo_raw_ = Vo_read_->analogReadDebounced(VRAW_BARE_DETECTED, reset_, name_);
  Vo_ =  float(Vo_raw_)*VO_CONV_GAIN;
}


// Class Sensors
Sensors::Sensors(double T, double T_temp, Pins *pins, Sync *ReadSensors, Sync *ReadTemp, Sync *Talk, Sync *Summarize,
  uint64_t time_now, uint64_t millis, BatteryMonitor *Mon):
  AmpFilt(nullptr), dt_ib_(0ULL), dt_ib_hdwe_(0ULL), IbAmpRMS(nullptr), IbNoaRMS(nullptr),
  inst_millis_(millis), inst_time_(time_now), NoaFilt(nullptr), Prbn_Tb_(nullptr), Prbn_Vb_(nullptr), Prbn_Ib_amp_(nullptr), Prbn_Ib_noa_(nullptr),
  reset_temp_(false), sample_time_ib_(0UL), sample_time_ib_hdwe_(0UL), sample_time_Tb_(0ULL), sample_time_Tb_hdwe_(0UL), sample_time_vb_(0UL), sample_time_vb_hdwe_(0UL),
  SelFiltCal(nullptr), TbHdweFilt(nullptr), TbModelFilt(nullptr), VbFilt(nullptr), VbRMS(nullptr), VcRMS(nullptr), Tb_read_(nullptr), Vb_read_(nullptr), Tb_raw_(0), Tb_volt_(NOMINAL_TB), Vb_raw_(0), Vb_(NOMINAL_VB), Vb_f_(NOMINAL_VB), Vb_hdwe_(NOMINAL_VB),
  Vb_hdwe_f_(NOMINAL_VB), Vb_model_(NOMINAL_VB), Vb_volt_(NOMINAL_VB), Vc_(0.), Vc_hdwe_(0.), Vc_hdwe_sum_(0.),
  Tb_(NOMINAL_TB), Tb_f_(NOMINAL_TB), Tb_f_rate_(0.), Tb_hdwe_(NOMINAL_TB), Tb_hdwe_f_(NOMINAL_TB), Tb_hdwe_f_dt_(0.), Tb_hdwe_f_rate_(0.), Tb_hdwe_f_rstate_(NOMINAL_TB), Tb_hdwe_f_lstate_(NOMINAL_TB), Tb_hdwe_f_tau_(0.),
  Tb_model_(NOMINAL_TB), Tb_model_f_(NOMINAL_TB), Tb_model_f_dt_(0.), Tb_model_f_rate_(0.), Tb_model_f_rstate_(NOMINAL_TB), Tb_model_f_lstate_(NOMINAL_TB), Tb_model_f_tau_(0.),
  Ib_(0.), Ib_f_(0.), Ib_amp_(0.), Ib_amp_hdwe_(0.), Ib_amp_hdwe_f_(0.), Ib_amp_hdwe_kf_(0.), Ib_amp_model_(0.), Ib_amp_rms_(0.),
  Ib_hdwe_f_(0.), Ib_hdwe_kf_(0.), Ib_hdwe_f_cal_(0.), Ib_noa_(0.), Ib_noa_hdwe_(0.), Ib_noa_hdwe_f_(0.), Ib_noa_hdwe_kf_(0.), Ib_noa_rms_(0.),
  Ib_noa_model_(0.), Ib_hdwe_(0.), Ib_hdwe_model_(0.), Ib_model_(0.), Ib_model_in_(0.),
  Vb_rms_(0.), Vc_rms_(0.), Wb_(0.), now_(0ULL), now_temp_(0ULL), T_(0.), reset_(false), T_filt_(0.), T_temp_(0.),
  elapsed_inj_(0ULL), start_inj_(0ULL), stop_inj_(0ULL), end_inj_(0ULL), control_time_(0.), display_(true), bms_off_(false), sat_(false), saturated_(false)
{
  T_ = T;
  T_filt_ = T;
  T_temp_ = T_temp;
  #if defined(HDWE_IB_HI_LO) || defined(HDWE_BARE)
    this->ShuntAmp = new Shunt("Amp", 0x49, ap.ib_scale_amp_ptr(), sp.ib_bias_amp_ptr(), SHUNT_AMP_GAIN, pins->Vcm_pin, pins->Vom_pin, pins->Vh3v3_pin, true, KF_USE_AMP);
    this->ShuntNoAmp = new Shunt("No Amp", 0x48, ap.ib_scale_noa_ptr(), sp.ib_bias_noa_ptr(), SHUNT_NOA_GAIN, pins->Vcn_pin, pins->Von_pin, pins->Vh3v3_pin, true, KF_USE_NOA);
  #else
    this->ShuntAmp = new Shunt("Amp", 0x49, ap.ib_scale_amp_ptr(), sp.ib_bias_amp_ptr(), SHUNT_AMP_GAIN, pins->Vcm_pin, pins->Vom_pin, pins->Vh3v3_pin, false, KF_USE_AMP);
    this->ShuntNoAmp = new Shunt("No Amp", 0x48, ap.ib_scale_noa_ptr(), sp.ib_bias_noa_ptr(), SHUNT_NOA_GAIN, pins->Vcn_pin, pins->Von_pin, pins->Vh3v3_pin, false, KF_USE_NOA);
  #endif
  this->Sim = new BatterySim(ap.ds_voc_soc(), 0., 0.);
  elapsed_inj_ = 0ULL;
  start_inj_ = 0ULL;
  stop_inj_ = 0ULL;
  end_inj_ = 0ULL;
  this->ReadSensors = ReadSensors;
  this->ReadTemp = ReadTemp;
  this->Summarize = Summarize;
  this->Talk = Talk;
  display_ = true;
  Ib_hdwe_model_ = 0.;
  Prbn_Tb_ = new PRBS_7(TB_NOISE_SEED);
  Prbn_Vb_ = new PRBS_7(VB_NOISE_SEED);
  Prbn_Ib_amp_ = new PRBS_7(IB_AMP_NOISE_SEED);
  Prbn_Ib_noa_ = new PRBS_7(IB_NOA_NOISE_SEED);
  Flt = new Fault(T, sp.preserving_ptr(), Mon, this);
  Serial.printf("Vb sense ADC pin started\n");
  AmpFilt = new LagExp(T, AMP_FILT_TAU, -NOM_UNIT_CAP*ap.nS()*ap.nP(), NOM_UNIT_CAP*ap.nS()*ap.nP());
  NoaFilt = new LagExp(T, AMP_FILT_TAU, -NOM_UNIT_CAP*ap.nS()*ap.nP(), NOM_UNIT_CAP*ap.nS()*ap.nP());
  SelFiltCal = new LagExp(T, AMP_FILT_TAU, -NOM_UNIT_CAP*ap.nS()*ap.nP(), NOM_UNIT_CAP*ap.nS()*ap.nP());
  TbHdweFilt = new LagExp(double(READ_DELAY)/1000., ap.Tb_filt(), TB_HDWE_MIN, TB_HDWE_MAX);
  TbModelFilt = new LagExp(double(READ_DELAY)/1000., ap.Tb_filt(), TB_HDWE_MIN, TB_HDWE_MAX);
  VbFilt = new LagExp(T, AMP_FILT_TAU, 0., NOMINAL_VB*2.5);
  Vb_read_ = new AnalogReadP2(pins->Vb_pin);
  Tb_read_ = new AnalogReadP2(pins->VTb_pin);
  IbAmpRMS = new RecursiveRMSMonitorFP();
  IbNoaRMS = new RecursiveRMSMonitorFP();
  VbRMS = new RecursiveRMSMonitorFP();
  VcRMS = new RecursiveRMSMonitorFP();
  #ifdef HDWE_IB_HI_LO
    sel_brk_hdwe = new ScaleBrk(HDWE_IB_HI_LO_NOA_LO, HDWE_IB_HI_LO_AMP_LO, HDWE_IB_HI_LO_AMP_HI, HDWE_IB_HI_LO_NOA_HI);
  #else
    sel_brk_hdwe = new ScaleBrk(0., 0., 0., 0.);
  #endif
}

// Deliberate choice based on results and inputs
// Inputs:  ib_sel_stat_, Ib_amp_hdwe_, Ib_noa_hdwe_, Ib_amp_model_, Ib_noa_model_
// Outputs:  Ib_hdwe_model_, Ib_hdwe_
void Sensors::ib_choose_active_standby()
{
  if ( Flt->ib_sel_stat()==1 )
  {
    Ib_hdwe_ = Ib_amp_hdwe_;
    Ib_hdwe_f_ = Ib_amp_hdwe_f_;
    Ib_hdwe_kf_ = Ib_amp_hdwe_kf_;
    Ib_hdwe_model_ = Ib_amp_model_;
    sample_time_ib_hdwe_ = ShuntAmp->sample_time();
    dt_ib_hdwe_ = ShuntAmp->dt_ms();
  }
  else if ( Flt->ib_sel_stat()==-1 )
  {
    Ib_hdwe_ = Ib_noa_hdwe_;
    Ib_hdwe_f_ = Ib_noa_hdwe_f_;
    Ib_hdwe_kf_ = Ib_noa_hdwe_kf_;
    Ib_hdwe_model_ = Ib_noa_model_;
    sample_time_ib_hdwe_ = ShuntNoAmp->sample_time();
    dt_ib_hdwe_ = ShuntNoAmp->dt_ms();
  }
  else
  {
    Ib_hdwe_ = 0.;
    Ib_hdwe_f_ = 0.;
    Ib_hdwe_model_ = 0.;
    sample_time_ib_hdwe_ = 0ULL;
    dt_ib_hdwe_ = 0ULL;
  }
}

// Deliberate choice based on results and inputs
// Inputs:  ib_choice_, Ib_noa_hdwe_, Ib_amp_hdwe_, Ib_noa_hdwe_, Ib_amp_model_, Ib_noa_model_
// Outputs:  Ib_hdwe_model_, Ib_hdwe_
void Sensors::ib_choose_hi_lo()
{
  int8_t sel_stat = 0;
  if ( Flt->ib_choice()==KeepTrying )
  {
    Ib_hdwe_ = scale_select(Ib_noa_hdwe_, sel_brk_hdwe, Ib_amp_hdwe_, Ib_noa_hdwe_, &sel_stat);
    Ib_hdwe_f_ = scale_select(Ib_noa_hdwe_, sel_brk_hdwe, Ib_amp_hdwe_f_, Ib_noa_hdwe_f_, &sel_stat);
    Ib_hdwe_kf_ = scale_select(Ib_noa_hdwe_, sel_brk_hdwe, Ib_amp_hdwe_kf_, Ib_noa_hdwe_kf_, &sel_stat);
    Ib_hdwe_model_ = scale_select(Ib_noa_model_, sel_brk_hdwe, Ib_amp_model_, Ib_noa_model_, &sel_stat);
    sample_time_ib_hdwe_ = ShuntNoAmp->sample_time();
    dt_ib_hdwe_ = ShuntNoAmp->dt_ms();
    Flt->ib_sel_stat(sel_stat);
  }
  else if ( Flt->ib_choice()==UsingNoa )
  {
    Ib_hdwe_ = Ib_noa_hdwe_;
    Ib_hdwe_f_ = Ib_noa_hdwe_f_;
    Ib_hdwe_kf_ = Ib_noa_hdwe_kf_;
    Ib_hdwe_model_ = Ib_noa_model_;
    sample_time_ib_hdwe_ = ShuntNoAmp->sample_time();
    dt_ib_hdwe_ = ShuntNoAmp->dt_ms();
    Flt->ib_sel_stat(-1);
  }
  else if ( Flt->ib_choice()==UsingAmp )
  {
    Ib_hdwe_ = Ib_amp_hdwe_;
    Ib_hdwe_f_ = Ib_amp_hdwe_f_;
    Ib_hdwe_kf_ = Ib_amp_hdwe_kf_;
    Ib_hdwe_model_ = Ib_amp_model_;
    sample_time_ib_hdwe_ = ShuntAmp->sample_time();
    dt_ib_hdwe_ = ShuntAmp->dt_ms();
    Flt->ib_sel_stat(1);
  }
  // UsingNone: both ib sensors hard-failed. Zero the current channels (so the EKF
  // and Coulomb counter integrate 0), but keep timing alive from ShuntAmp because
  // dt_ib_hdwe_=0 would zero T_ and corrupt now_/ctime_, stopping serial output.
  else
  {
    Ib_hdwe_ = 0.;
    Ib_hdwe_f_ = 0.;
    Ib_hdwe_kf_ = 0.;
    Ib_hdwe_model_ = 0.;
    sample_time_ib_hdwe_ = ShuntAmp->sample_time();
    dt_ib_hdwe_ = ShuntAmp->dt_ms();
    Flt->ib_sel_stat(0);
  }
}

// Pretty print
void Sensors::pretty_print()
{
  Serial.printf(" Tb_raw%d; cnt\n", Tb_raw_);
  Serial.printf(" Vb_raw%d; cnt\n", Vb_raw_);
  Serial.printf(" Vb%8.4f; V\n", Vb_);
  Serial.printf(" Vb_hdwe%8.4f; V\n", Vb_hdwe_);
  Serial.printf(" Vb_hdwe_f%8.4f; V\n", Vb_hdwe_f_);
  Serial.printf(" Vb_model%8.4f; V\n", Vb_model_);
  Serial.printf(" Vc%8.4f; V\n", Vc_);
  Serial.printf(" Vc_hdwe%8.4f; V\n", Vc_hdwe_);
  Serial.printf(" Vc_hdwe_sum%8.4f; V\n", Vc_hdwe_sum_);
  Serial.printf(" Tb%9.5f; C\n", Tb_);
  Serial.printf(" Tb_f%9.5f; C\n", Tb_f_);
  Serial.printf(" Tb_f_rate%11.8f; C/s\n", Tb_f_rate_);
  Serial.printf(" Tb_hdwe%9.5f; C\n", Tb_hdwe_);
  Serial.printf(" Tb_hdwe_f%9.5f; C\n", Tb_hdwe_f_);
  Serial.printf(" Tb_model%9.5f; C\n", Tb_model_);
  Serial.printf(" Tb_model_f%9.5f; C\n", Tb_model_f_);
}

// Make final assignemnts
void Sensors::select_volt_and_current_and_temp(BatteryMonitor *Mon)
{

  #ifdef HDWE_IB_HI_LO
    // Reselect ib since may be changed
    // Inputs:  ib_choice_, Ib_amp_hdwe_, Ib_noa_hdwe_, Ib_amp_model_(past), Ib_noa_model_(past)
    // Outputs:  Ib_hdwe_model_, Ib_hdwe_
    ib_choose_hi_lo();
  #else
    // Reselect ib since may be changed
    // Inputs:  ib_sel_stat_, Ib_amp_hdwe_, Ib_noa_hdwe_, Ib_amp_model_(past), Ib_noa_model_(past)
    // Outputs:  Ib_hdwe_model_, Ib_hdwe_
    ib_choose_active_standby();
  #endif

  // Final assignments
  // Tb
  if ( sp.mod_tb() )
  {
    if ( Flt->Tb_fa() && !ap.fake_faults() )
    {
      Tb_ = NOMINAL_TB;
      Tb_f_ = NOMINAL_TB;
      Tb_f_rate_ = 0.;
      sample_time_Tb_ = Sim->sample_time();
    }
    else if ( Flt->Tb_flt() && !ap.fake_faults() )  // last good value while flt resolved
    {
      sample_time_Tb_ = sample_time_Tb_hdwe_;
      // if ( sp.debug()==18 ) Serial.printf("SEL:  Tb_flt%2d fake%2d Tb_fa%2d Tb_%7.3f Tb_f_%7.3f\n", Flt->Tb_flt(), ap.fake_faults(), Flt->Tb_fa(),  Tb_, Tb_f_);
      return;
    }
    else
    {
      Tb_ = Tb_model_;
      Tb_f_ = Tb_model_f_;
      Tb_f_rate_ = Tb_model_f_rate_;
      sample_time_Tb_ = Sim->sample_time();
    }
    // if ( sp.debug()==18 ) Serial.printf("SEL:  Tb_flt%2d fake%2d Tb_fa%2d Tb_%7.3f Tb_f_%7.3f\n", Flt->Tb_flt(), ap.fake_faults(), Flt->Tb_fa(),  Tb_, Tb_f_);
  }
  else
  {
    if ( Flt->Tb_fa() && !ap.fake_faults() )
    {
      Tb_ = NOMINAL_TB;
      Tb_f_ = NOMINAL_TB;
      Tb_f_rate_ = 0.;
      sample_time_Tb_ = Sim->sample_time();
    }
    else if ( Flt->Tb_flt() && !ap.fake_faults() )
    {
      sample_time_Tb_ = sample_time_Tb_hdwe_;
      return;
    }
    else
    {
      Tb_ = Tb_hdwe_;
      Tb_f_ = Tb_hdwe_f_;
      Tb_f_rate_ = Tb_hdwe_f_rate_;
      sample_time_Tb_ = sample_time_Tb_hdwe_;
    }
  }

  // vb
  if ( sp.mod_vb() )
  {
    Vb_f_ = Vb_;
    if ( (Flt->wrap_vb_fa() || Flt->vb_fa_lt()) && !ap.fake_faults() )
    {
      Vb_ = Mon->vb_model_rev() * ap.nS();
      sample_time_vb_ = Sim->sample_time();
    }
    else
    {
      Vb_ = Vb_model_ + Vb_noise();
      sample_time_vb_ = Sim->sample_time();
    }
  }
  else
  {
    Vb_f_ = Vb_hdwe_f_;
    if ( (Flt->wrap_vb_fa() || Flt->vb_fa_lt()) && !ap.fake_faults() )
    {
      Vb_ = Mon->vb_model_rev() * ap.nS();
      sample_time_vb_ = Sim->sample_time();
    }
    else
    {
      Vb_ = Vb_hdwe_;
      sample_time_vb_ = sample_time_vb_hdwe_;
    }
  }
  Vb_rms_ = VbRMS->update(Vb_);
  Vc_rms_ = VcRMS->update(Vc_);


  // ib
  if ( sp.mod_ib() )
  {
    Ib_ = Ib_hdwe_model_;
    Ib_f_ = Ib_;
    Ib_amp_ = Ib_amp_model_;
    Ib_noa_ = Ib_noa_model_;
    Vc_ = HALF_V3V3;
    sample_time_ib_ = Sim->sample_time();
    dt_ib_ = Sim->dt_long();
  }
  else
  {
    Ib_ = Ib_hdwe_;
    Ib_f_ = Ib_hdwe_f_;
    Ib_amp_ = Ib_amp_hdwe_;
    Ib_noa_ = Ib_noa_hdwe_;
    Vc_ = Vc_hdwe_;
    sample_time_ib_ = sample_time_ib_hdwe_;
    dt_ib_ = dt_ib_hdwe_;
  }
  Ib_amp_rms_ = IbAmpRMS->update(Ib_amp_);
  Ib_noa_rms_ = IbNoaRMS->update(Ib_noa_);
  T_ =  double(dt_ib_)/1000.;  // s
  now_ = sample_time_ib_ - inst_millis_ + inst_time_*1000;
  ctime_ = double(now_)/1000.;
  // Log.info("    select_volt_and_current_and_temp now:  now_,%lld, cTime,%7.3f,", now_, double(now_)/1000.);

  if ( sp.debug()==62 ) Serial.printf(" ctime%12.3f T%6.3f Ib%7.3f Ib_hdwe%7.3f Ib_hdwe_model%7.3f Ib_amp%7.3f Ib_amp_model%7.3f Ib_amp_hdwe%7.3f Ib_noa%7.3f Ib_noa_model%7.3f Ib_noa_hdwe%7.3f\n",
   ctime_, T_, Ib_, Ib_hdwe_, Ib_hdwe_model_, Ib_amp_, Ib_amp_model_, Ib_amp_hdwe_, Ib_noa_, Ib_noa_model_, Ib_noa_hdwe_);

}

// Selection print debug
#ifdef DEBUG_INIT
  void Sensors::select_print(Sensors *Sen, BatteryMonitor *Mon)  // vv==62
  {
    Serial.printf("ib_ %7.3f                     vb_hdwe %7.3f                      Tb_hdwe %7.3f\n", ib_hdwe(), vb_hdwe(), Tb_hdwe_);
    Serial.printf("ib limits amp%7.3f noa %7.3f  diff %7.3f\n", ap.ib_amp_max(), ap.ib_noa_max(), Flt->ib_diff_thr());
    Serial.printf("ib_hdwe_?: %7.3f %7.3f ib_model_?: %7.3f %7.3f", ib_amp_hdwe(), ib_noa_hdwe(), ib_amp_model(), ib_noa_model());
    #ifdef HDWE_IB_HI_LO
      Serial.printf(" ib_choice_ %d ibmfa %d ibnfa %d ibdfa %d\n", Flt->ib_choice(), Flt->ib_amp_fa(), Flt->ib_noa_fa(), Flt->ib_diff_fa());
    #else
      Serial.printf(" ib_sel_stat_ %d ibmfa %d ibnfa %d ibdfa %d\n", Flt->ib_sel_stat(), Flt->ib_amp_fa(), Flt->ib_noa_fa(), Flt->ib_diff_fa());
    #endif
    Serial.printf("ib_hdwe:     %7.3f     ib_hdwe_model: %7.3f  modeling=%d\n", ib_hdwe(), ib_hdwe_model(), sp.mod_ib());
    Serial.printf("               ib:  %7.3f\n", ib());
    Serial.printf("     ");
    Serial.printf("ib_ %7.3f                     vb_hdwe %7.3f                      Tb_hdwe %7.3f\n", ib_hdwe(), vb_hdwe(), Tb_hdwe_);
    Serial.printf("ib limits amp%7.3f noa %7.3f  diff %7.3f\n", ap.ib_amp_max(), ap.ib_noa_max(), Flt->ib_diff_thr());
    Serial.printf("ib_hdwe_?: %7.3f %7.3f ib_model_?: %7.3f %7.3f", ib_amp_hdwe(), ib_noa_hdwe(), ib_amp_model(), ib_noa_model());
    Serial.printf("ib_hdwe:     %7.3f     ib_hdwe_model: %7.3f  modeling=%d\n", ib_hdwe(), ib_hdwe_model(), sp.mod_ib());
    Serial.printf("               ib:  %7.3f\n", ib());
    Serial.printf("     ");
  }
#endif

// Tb noise
float Sensors::Tb_noise()
{
  if ( ap.Tb_noise_amp()==0. ) return ( 0. );
  uint8_t raw = Prbn_Tb_->calculate();
  float noise = (float(raw)/127. - 0.5) * ap.Tb_noise_amp();
  return ( noise );
}

// Conversion.   Here to avoid circular reference to sp in headers.
float Sensors::Ib_amp_add() { return ( ap.ib_amp_add() * ap.nP() ); };
float Sensors::Ib_amp_max() { if (sp.tweak_test()) return ( __FLT_MAX__ ); else return ( ap.ib_amp_max() * ap.nP() ); };
float Sensors::Ib_amp_min() { if (sp.tweak_test()) return ( -__FLT_MAX__ ); else return ( ap.ib_amp_min() * ap.nP() ); };
float Sensors::Ib_noa_add() { return ( ap.ib_noa_add() * ap.nP() ); };
float Sensors::Ib_noa_max() { if (sp.tweak_test()) return ( __FLT_MAX__ ); else return ( ap.ib_noa_max() * ap.nP() ); };
float Sensors::Ib_noa_min() { if (sp.tweak_test()) return ( -__FLT_MAX__ ); else return ( ap.ib_noa_min() * ap.nP() ); };
float Sensors::Vb_add() { return ( ap.vb_add() * ap.nS() ); };

// Vb noise
float Sensors::Vb_noise()
{
  if ( ap.Vb_noise_amp()==0. ) return ( 0. );
  uint8_t raw = Prbn_Vb_->calculate();
  float noise = (float(raw)/127. - 0.5) * ap.Vb_noise_amp();
  return ( noise );
}

// Ib noise
float Sensors::Ib_amp_noise()
{
  if ( ap.Ib_amp_noise_amp()==0. ) return ( 0. );
  uint8_t raw = Prbn_Ib_amp_->calculate();
  float noise = (float(raw)/125. - 0.5) * ap.Ib_amp_noise_amp();
  return ( noise );
}
float Sensors::Ib_noa_noise()
{
  if ( ap.Ib_noa_noise_amp()==0. ) return ( 0. );
  uint8_t raw = Prbn_Ib_noa_->calculate();
  float noise = (float(raw)/125. - 0.5) * ap.Ib_noa_noise_amp();
  return ( noise );
}

// Print Shunt selection data
void Sensors::shunt_print()
{
    Serial.printf("reset,T,select,inj_bias,  vim,Vsm,Vcm,Vom,Ibhm,  vin,Vsn,Vcn,Von,Ibhn,  vi3,vh3, Ib_hdwe,T,Ib_amp_fault,Ib_amp_fail,Ib_noa_fault,Ib_noa_fail,=,    %d,%7.3f,%d,%7.3f,    %d,%7.3f,%7.3f,%7.3f,%7.3f,    %d,%7.3f,%7.3f,%7.3f,%7.3f,    %7.3f,%7.3f, %d,%d,  %d,%d,\n",
        reset_, T_, sp.ib_force(), sp.inj_bias(),
        ShuntAmp->vshunt_int(), ShuntAmp->vshunt(), ShuntAmp->Vc(), ShuntAmp->Vo(), ShuntAmp->Ishunt_cal(),
        ShuntNoAmp->vshunt_int(), ShuntNoAmp->vshunt(), ShuntNoAmp->Vc(), ShuntNoAmp->Vo(), ShuntNoAmp->Ishunt_cal(),
        Ib_hdwe_, T_,
        Flt->ib_amp_flt(), Flt->ib_amp_fa(), Flt->ib_noa_flt(), Flt->ib_noa_fa());
}

// Shunt selection.  Use Coulomb counter and EKF to sort three signals:  amp current, non-amp current, voltage
// Initial selection to charge the Sim for modeling currents on BMS cutback
// Inputs: sp.ib_force (user override), Mon (EKF status)
// States:  Ib_fail_noa_
// Outputs:  Ib_hdwe_, Ib_model_in_, Vb_sel_status_
void Sensors::shunt_select_initial(const bool reset)
{
    // Current signal selection, based on if there or not.
    // Over-ride 'permanent' with Talk(sp.ib_force) = Talk('s')

    // Hardware and model current assignments
    float hdwe_add, mod_add;
    if ( !sp.mod_ib() )
    {
      mod_add = 0.;
      hdwe_add = sp.ib_bias_all() + sp.inj_bias();
    }
    else
    {
      mod_add = sp.ib_bias_all() + sp.inj_bias();
      if ( sp.tweak_test() )
        hdwe_add = sp.inj_bias();
      else
        hdwe_add = 0.;
    }
    Ib_amp_model_ = max(min(Ib_amp_add() + mod_add, Ib_amp_max()), Ib_amp_min()); // uses past Ib.  Synthesized signal to use as substitute for sensor, Dm/Mm/Nm
    Ib_noa_model_ = max(min(Ib_noa_add() + mod_add, Ib_noa_max()), Ib_noa_min()); // uses past Ib.  Synthesized signal to use as substitute for sensor, Dn/Nx/Nm
    Ib_amp_hdwe_ = ShuntAmp->Ishunt_cal() + hdwe_add;    // Sense fault injection feeds logic, not model
    Ib_amp_hdwe_kf_ = ShuntAmp->Ishunt_cal_kf() + hdwe_add;    // Sense fault injection feeds logic, not model
    Ib_amp_hdwe_f_ = AmpFilt->calculate(Ib_amp_hdwe_, reset, AMP_FILT_TAU, T_);
    Vc_hdwe_ = max(ShuntAmp->Vc(), ShuntNoAmp->Vc());
    Vc_hdwe_sum_ = ShuntAmp->Vc() + ShuntNoAmp->Vc();
    Ib_noa_hdwe_ = ShuntNoAmp->Ishunt_cal() + hdwe_add;  // Sense fault injection feeds logic, not model
    Ib_noa_hdwe_kf_ = ShuntNoAmp->Ishunt_cal_kf() + hdwe_add;  // Sense fault injection feeds logic, not model
    Ib_noa_hdwe_f_ = NoaFilt->calculate(Ib_noa_hdwe_, reset, AMP_FILT_TAU, T_);
    Ib_hdwe_f_cal_ = SelFiltCal->calculate(Ib_hdwe_, reset, AMP_FILT_TAU, T_);

    // Initial choice
    // Inputs:  ib_choice/ib_sel_stat_, Ib_amp_hdwe_, Ib_noa_hdwe_, Ib_amp_model_(past), Ib_noa_model_(past)
    // Outputs:  Ib_hdwe_model_, Ib_hdwe_
    #ifdef HDWE_IB_HI_LO
      ib_choose_hi_lo();
    #else
      ib_choose_active_standby();
    #endif

    // When running normally the model tracks hdwe to synthesize reference information
    if ( !sp.mod_ib() )
    {
      Ib_model_in_ = Ib_hdwe_;
    }
    // Otherwise it generates signals for feedback into monitor
    else
    {
      Ib_model_in_ = mod_add;
    }
}

// Load analog voltage
void Sensors::Tb_load(const uint16_t tb_pin, const bool reset)
{
  float hdwe_add, mod_add;
  if ( !sp.mod_tb() )
  {
    mod_add = 0.;
    hdwe_add = sp.Tb_bias_hdwe();
  }
  else
  {
    mod_add = ap.Tb_bias_model() + Tb_noise();
    hdwe_add = 0.;
  }

  float res = Tb_volt_ * float(HDWE_RS_2WIRE) / (V3V3 - Tb_volt_);
  // Steinhart-Hart (see '2-wireRTD.ods')
  float lnres = log(res);
  if ( !sp.mod_tb_dscn() )
  {
    #if !defined(HDWE_BARE)
      Tb_raw_ = Tb_read_-> analogReadDebounced(VRAW_BARE_DETECTED, reset, "Tb");
      Tb_volt_ = float(Tb_raw_)*VTB_CONV_GAIN;
      Tb_hdwe_ = ( 1. / max( HDWE_SHA_2WIRE + (HDWE_SHB_2WIRE + HDWE_SHC_2WIRE *lnres*lnres) * lnres, 0.000001 ) ) - 273.;
      Tb_hdwe_ += hdwe_add;  // Fault injection
    #else
      Tb_raw_ = 0.;
      Tb_volt_ = 0.;
      Tb_hdwe_ = 0.;
    #endif
    Tb_model_ = NOMINAL_TB;
  }
  else
  {
    Tb_raw_ = 0;
    Tb_hdwe_ = NOMINAL_TB;
    Tb_model_ = NOMINAL_TB + mod_add;  // Fault injection
  }
  // if ( sp.debug()==18 )
  // {
  //   Serial.printf("\nTb_load: T_%7.3f sp.mod_tb() %2d,", T_, sp.mod_tb());
  //   Serial.printf(" Tb_raw_ %d Tb_volt_ %7.3f res %7.3f lnres %7.3f hdwe_add %7.3f Tb_hdwe_ %7.3f mod_add %7.3f Tb_model_ %7.3f\n",
  //     Tb_raw_, Tb_volt_, res, lnres, hdwe_add, Tb_hdwe_, mod_add, Tb_model_);
  // }
  Tb_hdwe_f_ = TbHdweFilt->calculate(Tb_hdwe_, reset || Flt->Tb_fa() || sp.mod_tb_dscn(), ap.Tb_filt(), T_, -T_RLIM, T_RLIM);
  Tb_hdwe_f_dt_ = TbHdweFilt->T();
  Tb_hdwe_f_tau_ = TbHdweFilt->tau();
  Tb_hdwe_f_rate_ = TbHdweFilt->rate();
  Tb_hdwe_f_rstate_ = TbHdweFilt->rstate();
  Tb_hdwe_f_lstate_ = TbHdweFilt->lstate();
  Tb_model_f_ = TbModelFilt->calculate(Tb_model_, reset || Flt->Tb_fa(), ap.Tb_filt(), T_, -T_RLIM, T_RLIM);
  Tb_model_f_dt_ = TbModelFilt->T();
  Tb_model_f_tau_ = TbModelFilt->tau();
  Tb_model_f_rate_ = TbModelFilt->rate();
  Tb_model_f_rstate_ = TbModelFilt->rstate();
  Tb_model_f_lstate_ = TbModelFilt->lstate();
  if ( sp.debug()==18 )
  {
    Serial.printf("\nTb_load: T_%7.3f", T_);
    Tb_print();
    Serial.printf("\n");
  }
  sample_time_Tb_hdwe_ = millis();
}

// Print analog voltage
void Sensors::Tb_print()
{
  Serial.printf("\nTb_print: tb_dscn%2d reset%2d stime%7.3f T%7.3f tb_dscn%2d Tb_raw%4d sp.Tb_bias_hdwe%7.3f Tb_hdwe%7.3f Tb_hdwe_f%7.3f Tb_hdwe_f_dt%7.3f Tb_hdwe_f_rate%7.3f Tb_hdwe_f_rstate%7.3f Tb_hdwe_f_lstate%7.3f tb_flt%2d tb_fa%2d\n",
    sp.mod_tb_dscn(), reset_, float(sample_time_Tb_hdwe_)/1000.f, T_, sp.mod_tb_dscn(), Tb_raw_, sp.Tb_bias_hdwe(), Tb_hdwe_, Tb_hdwe_f_, Tb_hdwe_f_dt_, Tb_hdwe_f_rate_, Tb_hdwe_f_rstate_,
    Tb_hdwe_f_lstate_, Flt->Tb_flt(), Flt->Tb_fa());
  Serial.printf("Tb_print:  reset%2d stime %7.3f T%7.3f tb_dscn%2d            ap.Tb_bias_model%7.3f Tb_model%7.3f Tb_model_f%7.3f Tb_model_f_dt%7.3f Tb_model_f_rate%7.3f Tb_model_f_rstate%7.3f Tb_model_f_lstate%7.3f tb_flt%2d tb_fa%2d\n",
    reset_, float(sample_time_Tb_hdwe_)/1000.f, T_, sp.mod_tb_dscn(), ap.Tb_bias_model(), Tb_model_, Tb_model_f_, Tb_model_f_dt_, Tb_model_f_rate_, Tb_model_f_rstate_,
    Tb_model_f_lstate_, Flt->Tb_flt(), Flt->Tb_fa());
  Serial.printf("Tb_print:  Tb_%7.3f Tb_f_%7.3f \n\n", Tb_, Tb_f_);
}

// Load analog voltage
void Sensors::vb_load(const uint16_t vb_pin, const bool reset)
{
  if ( !sp.mod_vb_dscn() )
  {
    #if !defined(HDWE_BARE)
      Vb_raw_ = Vb_read_-> analogReadDebounced(VRAW_BARE_DETECTED, reset, "Vb");
      Vb_volt_ = Vb_raw_ * VB_RAW_CONV_GAIN;
      Vb_hdwe_ =  float(Vb_raw_)*VB_CONV_GAIN*ap.Vb_scale() + float(VB_A) + sp.Vb_bias_hdwe();
    #endif
    Vb_hdwe_f_ = VbFilt->calculate(Vb_hdwe_, reset, AMP_FILT_TAU, T_);
  }
  else
  {
    Vb_raw_ = 0;
    Vb_hdwe_ = 0.;
  }
  sample_time_vb_hdwe_ = millis();
}

// Print analog voltage
void Sensors::vb_print()
{
  Serial.printf("reset, T, vb_dscn, Vb_raw, sp.Vb_bias_hdwe(), Vb_hdwe, vb_flt(), vb_fa_lt(), wv_fa=, %d, %7.3f, %d, %d, %7.3f,  %7.3f, %d, %d, %d,\n",
    reset_, T_, sp.mod_vb_dscn(), Vb_raw_, sp.Vb_bias_hdwe(), Vb_hdwe_, Flt->vb_flt(), Flt->vb_fa_lt(), Flt->wrap_vb_fa());
}
