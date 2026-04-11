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


// class TempSensor
// constructors
TempSensor::TempSensor(const uint16_t pin, const bool parasitic, const uint16_t conversion_delay)
: tb_stale_flt_(true)
{
   SdTb = new SlidingDeadband(HDB_TBATT);
   Serial.printf("Tb started\n");
}
TempSensor::TempSensor(const uint16_t pin, const bool parasitic, const uint16_t conversion_delay, const uint16_t VTb_pin)
: tb_stale_flt_(true), VTb_pin_(VTb_pin)
{
   SdTb = new SlidingDeadband(HDB_TBATT);
   Serial.printf("Tb started\n");
}
TempSensor::~TempSensor() {}
// operators
// functions
float TempSensor::sample(Sensors *Sen)
{
  // Log.info("  TempSensor::sample");
  // Read Sensor
  // MAXIM conversion 1-wire Tp plenum temperature
  static double Tb_hdwe = 0.;

  Tb_volt_ = float(analogRead(VTb_pin_))*VTB_CONV_GAIN;
  sample_time_ = millis();

  float res = Tb_volt_ * float(HDWE_RS_2WIRE) / (V3V3 - Tb_volt_);
  #ifdef USE_SH_2WIRE
    // Steinhart-Hart (see '2-wireRTD.ods')
    float lnres = log(res);
    Tb_hdwe = ( 1. / max( HDWE_SHA_2WIRE + (HDWE_SHB_2WIRE + HDWE_SHC_2WIRE *lnres*lnres) * lnres, 0.000001 ) ) - 273.;

  #else
    // Data fit (see '2-wireRTD.ods')
    Tb_hdwe = float(HDWE_M_2WIRE) * log10(res) + float(HDWE_B_2WIRE);

  #endif

  tb_stale_flt_ = false;
  if ( sp.debug()==16 ) Serial.printf("I 2wire:  volt=%7.3f Tb_hdwe=%9.5f,\n", Tb_volt_, Tb_hdwe);


  return ( Tb_hdwe );
}


// class Shunt
// constructors
Shunt::Shunt()
: name_("None"), port_(0x00), bare_shunt_(false){}
Shunt::Shunt(const String name, const uint8_t port, float *sp_ib_scale,  float *sp_Ib_bias, const float v2a_s,
  const uint8_t vc_pin, const uint8_t vo_pin, const uint8_t vh3v3_pin, const bool using_opAmp, const bool using_kf)
: name_(name), port_(port), bare_shunt_(false), v2a_s_(v2a_s),
  vshunt_int_(0), vshunt_int_0_(0), vshunt_int_1_(0), vshunt_(0), Ishunt_cal_(0),
  sp_ib_bias_(sp_Ib_bias), sp_ib_scale_(sp_ib_scale), sample_time_(0UL), sample_time_z_(0UL), dscn_cmd_(false),
  vc_pin_(vc_pin), vo_pin_(vo_pin), vr_pin_(vh3v3_pin), Vc_raw_(HALF_V3V3/VH3V3_CONV_GAIN), Vc_(HALF_V3V3),
  Vo_Vc_(0.), using_opamp_(using_opAmp), using_kf_(using_kf)
{
  if ( using_opamp_ ) Serial.printf("Ib %s sense ADC pin %d started using OpAmp and 3V3 pin %d\n", name_.c_str(), vo_pin_, vr_pin_);
  else Serial.printf("Ib %s sense ADC pins %d and %d started\n", name_.c_str(), vo_pin_, vc_pin_);
  KF_ = new KalmanFilter(0.1, 0., KF_Q_STD, KF_R_STD);
  Vc_read_ = new AnalogReadP2(using_opamp_ ? vr_pin_ : vc_pin_);
  Bare_delay_ = new TFDelay(false, RAW_BARE_S, RAW_BARE_R, sample_time_);
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
    bare_shunt_ = Bare_delay_->calculate(Vc_read_->dead(), RAW_BARE_S, RAW_BARE_R, Sen->T(),reset_);
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
  if ( using_opamp_ )
  {
    Vc_raw_ = Vc_read_->analogReadDebounced(VC_BARE_DETECTED, reset_, name_);
    Vc_ =  float(Vc_raw_)*VH3V3_CONV_GAIN + ap.vc_add();
  }
  else
  {
    Vc_raw_ = Vc_read_->analogReadDebounced(VC_BARE_DETECTED, reset_, name_);
    Vc_ =  float(Vc_raw_)*VC_CONV_GAIN + ap.vc_add();
  }
}

// Sample Vo output voltage of amplifier
void Shunt::sample_Vo()
{
  sample_time_z_ = sample_time_;
  sample_time_ = millis();
  Vo_raw_ = analogRead(vo_pin_);
  Vo_ =  float(Vo_raw_)*VO_CONV_GAIN;
}


// Class Sensors
Sensors::Sensors(double T, double T_temp, Pins *pins, Sync *ReadSensors, Sync *ReadTemp, Sync *Talk, Sync *Summarize,
  unsigned long long time_now, unsigned long long millis, BatteryMonitor *Mon):
  AmpFilt(nullptr), dt_ib_(0ULL), dt_ib_hdwe_(0ULL), IbAmpRMS(nullptr), IbNoaRMS(nullptr),
  inst_millis_(millis), inst_time_(time_now), NoaFilt(nullptr), Prbn_Tb_(nullptr), Prbn_Vb_(nullptr), Prbn_Ib_amp_(nullptr), Prbn_Ib_noa_(nullptr),
  reset_temp_(false), sample_time_ib_(0UL), sample_time_ib_hdwe_(0UL), sample_time_tb_(0UL), sample_time_vb_(0UL), sample_time_vb_hdwe_(0UL),
  SelFiltCal(nullptr), VbFilt(nullptr), VbRMS(nullptr), VcRMS(nullptr), Vb_raw_(0), Vb_(NOMINAL_VB), Vb_f_(NOMINAL_VB), Vb_hdwe_(NOMINAL_VB),
  Vb_hdwe_f_(NOMINAL_VB), Vb_model_(NOMINAL_VB), Vb_volt_(NOMINAL_VB), Vc_(0.), Vc_hdwe_(0.), Vc_hdwe_sum_(0.),
  Tb_(NOMINAL_TB), Tb_f_(NOMINAL_TB), Tb_f_rate_(0.), Tb_hdwe_(NOMINAL_TB), Tb_hdwe_filt_(NOMINAL_TB), Tb_hdwe_filt_rate_(0.),
  Tb_model_(NOMINAL_TB), Tb_model_filt_(NOMINAL_TB), Tb_model_filt_rate_(0.),
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
  #if !defined(HDWE_2WIRE) & !defined(HDWE_BARE)
    this->SensorTb = new TempSensor(pins->pin_1_wire, TEMP_PARASITIC, TEMP_DELAY_DS18);
  #elif !defined(HDWE_BARE)
    this->SensorTb = new TempSensor(pins->pin_1_wire, TEMP_PARASITIC, TEMP_DELAY_DS18, pins->VTb_pin);
  #endif
  this->TbModelFilt = new LagExp(double(READ_DELAY)/1000., TB_FILT, -20.0, 150.);
  this->TbSenseFilt = new LagExp(double(READ_DELAY)/1000., TB_FILT, -20.0, 150.);
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
  VbFilt = new LagExp(T, AMP_FILT_TAU, 0., NOMINAL_VB*2.5);
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
  else
  {
    Ib_hdwe_ = 0.;
    Ib_hdwe_f_ = 0.;
    Ib_hdwe_kf_ = 0.;
    Ib_hdwe_model_ = 0.;
    sample_time_ib_hdwe_ = 0ULL;
    dt_ib_hdwe_ = 0ULL;
    Flt->ib_sel_stat(0);
  }
}

// Pretty print
void Sensors::pretty_print()
{
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
  Serial.printf(" Tb_hdwe_filt%9.5f; C\n", Tb_hdwe_filt_);
  Serial.printf(" Tb_hdwe_filt_rate%11.8f; C\n", Tb_hdwe_filt_rate_);
  Serial.printf(" Tb_model%9.5f; C\n", Tb_model_);
  Serial.printf(" Tb_model_filt%9.5f; C\n", Tb_model_filt_);
}

// Make final assignemnts
void Sensors::select_temp(BatteryMonitor *Mon)
{
  // Final assignments
  // tb
  if ( sp.mod_tb() )
  {
    if ( Flt->tb_fa() )
    {
      Tb_ = NOMINAL_TB;
      Tb_f_ = NOMINAL_TB;
      Tb_f_rate_ = 0.;
    }
    else
    {
      Tb_ = NOMINAL_TB + Tb_noise() + ap.Tb_bias_model();
      Tb_model_ = Tb_;
      Tb_f_ = Tb_model_filt_;
      Tb_f_rate_ = Tb_model_filt_rate_;
    }
    if ( sp.debug()==16) Serial.printf("Tb_noise %9.5f Tb%9.5f Tb_f%9.5f Tb_f%9.5f tb_fa %d\n", Tb_noise(), Tb_, Tb_f_, Tb_f_, Flt->tb_fa());
  }
  else
  {
    if ( Flt->tb_fa() )
    {
      Tb_ = NOMINAL_TB;
      Tb_f_ = NOMINAL_TB;
      Tb_f_rate_ = 0.;
    }
    else
    {
      Tb_ = Tb_hdwe_;
      Tb_f_ = Tb_hdwe_filt_;
      Tb_f_rate_ = Tb_hdwe_filt_rate_;
      // Log.info("    select_volt_and_current:  Tb=Tb_hdwe=%9.5f Tb_f%9.5f Tb_f_rate%11.8f", Tb_hdwe_, Tb_f_, Tb_f_rate_);
    }
  }
  sample_time_tb_ = SensorTb->sample_time();
}

// Make final assignemnts
void Sensors::select_volt_and_current(BatteryMonitor *Mon)
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
  // Log.info("    select_volt_and_current now:  now_,%lld, cTime,%7.3f,", now_, double(now_)/1000.);

  if ( sp.debug()==62 ) Serial.printf(" Ib%7.3f Ib_hdwe%7.3f Ib_hdwe_model%7.3f Ib_amp%7.3f Ib_amp_model%7.3f Ib_amp_hdwe%7.3f Ib_noa%7.3f Ib_noa_model%7.3f Ib_noa_hdwe%7.3f\n",
   Ib_, Ib_hdwe_, Ib_hdwe_model_, Ib_amp_, Ib_amp_model_, Ib_amp_hdwe_, Ib_noa_, Ib_noa_model_, Ib_noa_hdwe_);

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
    Ib_amp_model_ = max(min(Ib_amp_add() + mod_add, Ib_amp_max()/SIZE_MARG), Ib_amp_min()/SIZE_MARG); // uses past Ib.  Synthesized signal to use as substitute for sensor, Dm/Mm/Nm
    Ib_noa_model_ = max(min(Ib_noa_add() + mod_add, Ib_noa_max()/SIZE_MARG), Ib_noa_min()/SIZE_MARG); // uses past Ib.  Synthesized signal to use as substitute for sensor, Dn/Nx/Nm
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

// Load and filter Tb
void Sensors::temp_load_and_filter(Sensors *Sen, const bool reset_temp)
{
  // Log.info("  temp_load_and_filter:  calling sample");
  reset_temp_ = reset_temp;
  #ifndef HDWE_BARE
    Tb_hdwe_ = SensorTb->sample(Sen);  // Must sample even if using model

    Tb_model_filt_ = TbModelFilt->calculate(Tb_model_, reset_temp_, ap.tb_filt(), min(T_temp_, F_MAX_T_TEMP), T_RLIM, -T_RLIM);
    Tb_model_filt_rate_ = TbModelFilt->rate();

    if ( sp.mod_tb() )
    {
      Tb_hdwe_ = Tb_model_;
    }
    now_temp_ = sample_time_tb_ - inst_millis_ + inst_time_*1000;
    // Log.info("    temp_load_and_filter:  now_temp_,%lld, cTime,%7.3f,", now_temp_, double(now_temp_)/1000.);
  #else
    Tb_hdwe_ = RATED_TEMP;
  #endif

  // Filter and add rate limited bias
  if ( reset_temp_ && Tb_hdwe_>TEMP_RANGE_CHECK_MAX )  // Bootup T=85.5 C
  {
      Tb_hdwe_ = RATED_TEMP;
  }
  Tb_hdwe_ += sp.Tb_bias_hdwe();  // Fault injection

  Tb_hdwe_filt_ = TbSenseFilt->calculate(Tb_hdwe_, reset_temp_, ap.tb_filt(), min(T_temp_, F_MAX_T_TEMP), T_RLIM, -T_RLIM);
  Tb_hdwe_filt_rate_ = TbSenseFilt->rate();

  if ( sp.debug()==16 ) Serial.printf("temp_load_and_filter: T_temp, Tb_hdwe, Tb_hdwe_filt, rstate, lstate %11.8f %11.8f %11.8f %11.8f %11.8f\n",
        T_temp_, Tb_hdwe_, Tb_hdwe_filt_, TbSenseFilt->rstate(), TbSenseFilt->lstate());

  if ( sp.debug()==16 || (sp.debug()==-1 && reset_temp_) ) Serial.printf("reset_temp_ T_temp Tb_bias_hdwe_loc, RATED_TEMP, Tb_hdwe, Tb_hdwe_filt, Tb_hdwe_filt_rate, ready, rstate, lstate %d %8.6f %11.8f %11.8f %11.8f %11.8f %11.8f %d %11.8f  %11.8f\n",
    reset_temp_, T_temp_, sp.Tb_bias_hdwe(), RATED_TEMP, Tb_hdwe_, Tb_hdwe_filt_, Tb_hdwe_filt_rate_, cp.tb_info.ready, TbSenseFilt->rstate(),  TbSenseFilt->lstate());

  #ifdef HDWE_2WIRE
    Flt->tb_check(Sen, TB_MIN, TB_MAX,  reset_temp_);
  #else
    Flt->tb_stale(reset_temp_, Sen);
  #endif
}

// Load analog voltage
void Sensors::vb_load(const uint16_t vb_pin, const bool reset)
{
  if ( !sp.mod_vb_dscn() )
  {
    #if !defined(HDWE_BARE)
      Vb_raw_ = analogRead(vb_pin);
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
