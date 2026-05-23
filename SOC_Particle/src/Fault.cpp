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
// #include "command.h"
#include "Fault.h"
// #include "constants.h"
// #include <math.h>
#include "debug.h"
// #include "Summary.h"

extern CommandPars cp;  // Various parameters shared at system level
extern PrinterPars pr;  // Print buffer
extern PublishPars pp;  // For publishing
extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle

// Print bitmap utility
String bitMapPrint(char *buf, const int16_t fw, const uint8_t num)
{
  for ( int i=0; i<num; i++ )
  {
    if ( bitRead(fw, i) ) buf[num-i-1] = '1';
    else  buf[num-i-1] = '0';
  }
  buf[num] = '\0';
  return ( String(buf) );
}


// Class Looparound
Looparound::Looparound(BatteryMonitor *Mon, Sensors *Sen, const float wrap_hi_volt, const float wrap_lo_volt, const double wrap_trim_gain,
    const float imin, const float imax, const float err_max):
  chem_(Mon->chem()), dv_dyn_(0.), e_wrap_(0.), e_wrap_filt_(0.), e_wrap_rate_(0.), e_wrap_trim_(0.), e_wrap_trimmed_(0.),
  ewhi_thr_(0.), ewhi_thr_base_(0.), ewlo_thr_(0.), ewlo_thr_base_(0.), hi_fail_(false), hi_fault_(false), 
  ib_(0.), ib_dyn_(0.), ib_past_(0.), imax_(imax), imin_(imin), lo_fail_(false), lo_fault_(false), Mon_(Mon), 
  reset_(false), Sen_(Sen), vb_(0.), voc_(0.), voc_soc_(0.), wrap_hi_volt_(wrap_hi_volt), wrap_lo_volt_(wrap_lo_volt),
  wrap_trim_gain_(wrap_trim_gain)
{
  ChargeTransfer_ = new LagExp(EKF_NOM_DT, chem_->tau_ct, -NOM_UNIT_CAP, NOM_UNIT_CAP);     // actual update time provided run time
  Trim_ = new TustinIntegrator(EKF_NOM_DT, -err_max*10., err_max*10.);          // actual update time provided run time
  WrapErrFilt_ = new LagExp(2., WRAP_ERR_FILT, -err_max, err_max);   // actual update time provided run time
  WrapHi_ = new TFDelay(false, WRAP_HI_SET, WRAP_HI_RES, EKF_NOM_DT);  // Wrap test persistence.  Initializes false
  WrapLo_ = new TFDelay(false, WRAP_LO_SET, WRAP_LO_RES, EKF_NOM_DT);  // Wrap test persistence.  Initializes false
}

// Update the loop
void Looparound::calculate(const bool reset, const bool disable_fault, const float ib, Sensors *Sen, const bool freeze)
{
  reset_ = reset || Sen_->Flt->reset_all_faults();
  double frozen = 1. - double(freeze);
  ib_ = ib;
  vb_ = Mon_->vb();
  voc_soc_ = Mon_->voc_soc();
  
  // Dynamic emf. vb_ is stale when running with model
  float ib_into_ct = ib_;
  if (sp.mod_vb()) 
  {
    ib_into_ct = ib_past_;
  }
  ib_dyn_ = ChargeTransfer_->calculate(ib_into_ct, reset_, chem_->tau_ct, Sen_->T());
  dv_dyn_ = ib_dyn_*chem_->r_ct*ap.slr_res() + ib_into_ct*chem_->r_0*ap.slr_res();
  voc_ = vb_ - dv_dyn_;
  e_wrap_ = voc_soc_ - voc_;

  // Trimmer using past values
  float trim_init = 0.;
  float trim_rate_lim = 0.;
  if ( wrap_trim_gain_ > 0. )
  {
    trim_init = -(Mon_->vb() - Mon_->voc_soc() - dv_dyn_);
    trim_rate_lim = max(min(e_wrap_filt_*wrap_trim_gain_*frozen, MAX_TRIM_RATE), -MAX_TRIM_RATE);
    e_wrap_trim_ = -Trim_->calculate(trim_rate_lim, min(Sen_->T(), F_MAX_T_WRAP), reset_, trim_init,
                                      -ewhi_thr_base_*EWHI_TRM_SLR, -ewlo_thr_base_*EWLO_TRM_SLR);
  }
  else
  {
    trim_init = 0.;
    trim_rate_lim = 0.;
    e_wrap_trim_ = 0.;
  }

  // e_wrap using present values
  e_wrap_trimmed_ = e_wrap_ + e_wrap_trim_;
  double e_wrap_filt_rate = __DBL_MAX__;
  if ( freeze )
  {
    e_wrap_filt_rate = 0.;
  }
  e_wrap_filt_ = WrapErrFilt_->calculate(e_wrap_trimmed_, reset_, Sen_->T(), -e_wrap_filt_rate, e_wrap_filt_rate);
  e_wrap_rate_ = WrapErrFilt_->rate();

  // Thresholds. Scalars are calculated by Flt->wrap_scalars()
  ewhi_thr_base_ = wrap_hi_volt_ * ap.ewhi_slr();
  ewhi_thr_ = ewhi_thr_base_ * Sen_->Flt->ewsat_slr() * Sen_->Flt->ewmin_slr();
  ewlo_thr_base_ = wrap_lo_volt_ * ap.ewlo_slr();
  ewlo_thr_ = ewlo_thr_base_ * Sen_->Flt->ewsat_slr() * Sen_->Flt->ewmin_slr();

  // sat logic screens out voc jumps when ib>0 when saturated
  // wrap_hi and wrap_lo don't latch because need them available to check next ib sensor selection for dual ib sensor
  // wrap_vb latches because vb is single sensor  faultAssign( (e_wrap_filt_ >= ewhi_thr_ && !Mon->sat()), WRAP_HI_FLT);

  hi_fault_ = (e_wrap_filt_ >= ewhi_thr_) && !disable_fault;
  if ( !disable_fault )  // freeze fail
    hi_fail_ = WrapHi_->calculate(hi_fault_, WRAP_HI_SET, WRAP_HI_RES, Sen_->T(), reset_) && !Sen_->Flt->vb_fa_lt();  // not latched

  lo_fault_ = (e_wrap_filt_ <= ewlo_thr_) && !disable_fault;
  if ( !disable_fault )  // freeze fail
    lo_fail_ = WrapLo_->calculate(lo_fault_, WRAP_LO_SET, WRAP_LO_RES, Sen_->T(), reset_) && !Sen_->Flt->vb_fa_lt();  // not latched

  if ( sp.debug()==71 ) Serial.printf("ib%7.3f reset%2d frz%4.1f ewlo_thr/e_wrap_filt/ewhi_thr  %7.3f/%7.3f/%7.3f trim%7.3f vb_fa_lt %d lo_fault/fail %d/%d hi_fault/fail %d/%d\n",
   ib_, reset_, frozen, ewlo_thr_, e_wrap_filt_, ewhi_thr_, e_wrap_trim_, Sen_->Flt->vb_fa_lt(), lo_fault_, lo_fail_, hi_fault_, hi_fail_);
  ib_past_ = ib_;
}

String Looparound::pretty_print(Sensors *Sen)
{
  String txBuf;
  txBuf = String::format(" reset %d\n", reset_) + 
    String::format(" ib%7.3f A\n", ib_) +
    String::format(" ib_dyn%7.3f A\n", ib_dyn_) +
    String::format(" dv_dyn%7.3f V\n", dv_dyn_) +
    String::format(" voc_soc%7.3f V\n", voc_soc_) +
    String::format(" voc%7.3f V\n", voc_) +
    String::format(" e_wrap (= voc_soc-voc) %7.3f V\n", e_wrap_) +
    String::format(" e_wrap_filt%7.3f V\n", e_wrap_filt_) +
    String::format(" ewhi_slr%7.3f\n",  ap.ewhi_slr()) +
    String::format(" ewlo_slr%7.3f\n",  ap.ewlo_slr()) +
    String::format(" ewmin_slr%7.3f\n",  Sen_->Flt->ewmin_slr()) +
    String::format(" ewsat_slr%7.3f\n", Sen_->Flt->ewsat_slr()) +
    String::format(" ewhi_thr_base%7.3f V\n", ewhi_thr_base_) +
    String::format(" ewlo_thr_base%7.3f V\n", ewlo_thr_base_) +
    String::format(" ewhi_thr (kicked)%7.3f V\n", ewhi_thr_) +
    String::format(" ewlo_thr (kicked)%7.3f V\n", ewlo_thr_) +
    String::format(" e_wrap_trim%7.3f V\n", e_wrap_trim_) +
    String::format(" e_wrap_trimmed%7.3f V\n", e_wrap_trimmed_) +
    String::format(" wrap_trim_gain%7.3f r/s\n", wrap_trim_gain_) +
    String::format(" hi_fault/fail %d/%d\n", hi_fault_, hi_fail_) +
    String::format(" lo_fault/fail %d/%d\n", lo_fault_, lo_fail_) +
    String::format(" ewlo_thr/ewhi_thr%7.3f/%7.3f V\n", ewlo_thr_, ewhi_thr_);
    return ( txBuf );
}


// Class Fault
Fault::Fault(const double T, uint8_t *preserving, BatteryMonitor *Mon, Sensors *Sen):
  cc_diff_thr_(0.), ib_diff_thr_(0.), ib_quiet_thr_(0.),
  cc_diff_(0.), cc_diff_fa_(false), cc_diff_empty_slr_(1.), disable_amp_fault_(false),
  ewmax_slr_(1.), ewmin_slr_(1.), ewsat_slr_(1.),
  e_wrap_(0.), e_wrap_filt_(0.), e_wrap_rate_(0.), fltw_(0UL), falw_(0UL),
  ib_amp_hi_(false), ib_amp_invalid_(false), ib_amp_lo_(false), ib_noa_rate_(0.),
  ib_choice_(KeepTrying), ib_choice_last_(KeepTrying), ib_decision_(0),
  ib_diff_(0.), ib_diff_f_(0.), ib_is_functional_(false), ib_is_quiet_(false), ib_lo_active_(true),
  ib_lo_limited_hi_(false), ib_lo_limited_lo_(false),
  ib_noa_hi_(false), ib_noa_invalid_(false), ib_noa_lo_(false), ib_quiet_(0.), ib_really_quiet_(false), ib_rate_(0.),
  ib_sel_stat_(IB_SEL_STAT_DEF), ib_sel_stat_last_(IB_SEL_STAT_DEF), latch_(false),
  latch_fake_(false), rate_amp_(false), rate_noa_(false), reset_all_faults_(false),
  reset_all_faults_print_(false), sp_preserving_(preserving), splrr_amp_(false), splrr_noa_(false),
  tb_sel_stat_(TB_SEL_STAT_DEF), tb_sel_stat_last_(TB_SEL_STAT_DEF), vb_sel_stat_(VB_SEL_STAT_DEF),
  vb_sel_stat_last_(VB_SEL_STAT_DEF), wrap_hi_volt_(WRAP_HI_AMPV), wrap_hi_noa_(WRAP_HI_NOAV),
  wrap_lo_volt_(WRAP_LO_AMPV), wrap_lo_noa_(WRAP_LO_NOAV)
{
  IbDiffFilt = new LagExp(T, TAU_ERR_FILT, -IBATT_DISAGREE_THRESH*1.5, IBATT_DISAGREE_THRESH*1.5);  // actual update time provided run time
  IbdPosPer = new TFDelay(false, IBATT_INST_DIFF_SET, IBATT_INST_DIFF_RES, T);
  IbdNegPer = new TFDelay(false, IBATT_INST_DIFF_SET, IBATT_INST_DIFF_RES, T);
  IbdHiPer = new TFDelay(false, IBATT_DISAGREE_SET, IBATT_DISAGREE_RES, T);
  IbdLoPer = new TFDelay(false, IBATT_DISAGREE_SET, IBATT_DISAGREE_RES, T);
  CcdiffPer  = new TFDelay(false, CC_DIFF_SET, CC_DIFF_RES, T);
  IbAmpHardFail  = new TFDelay(false, IB_HARD_SET, IB_HARD_RES, T);
  IbLoLimitedHi  = new TFDelay(true, IB_LO_ACTIVE_SET, IB_LO_ACTIVE_RES, T);
  IbLoLimitedLo  = new TFDelay(true, IB_LO_ACTIVE_SET, IB_LO_ACTIVE_RES, T);
  IbNoAmpHardFail  = new TFDelay(false, IB_HARD_SET, IB_HARD_RES, T);
  TbHardFail  = new TFDelay(false, TB_HARD_SET, TB_HARD_RES, T);
  VbHardFail  = new TFDelay(false, VB_HARD_SET, VB_HARD_RES, T);
  VcHardFail  = new TFDelay(false, VC_HARD_SET, VC_HARD_RES, T);
  QuietPer  = new TFDelay(false, QUIET_SET, QUIET_RES, T);
  if ( !sp.mod_ib() )
    QuietPerFunc  = new TFDelay(false, QUIET_SET, QUIET_RES, T);
  else
    QuietPerFunc  = new TFDelay(true, QUIET_SET, QUIET_RES, T);
  WrapErrFilt = new LagExp(T, WRAP_ERR_FILT, -MAX_WRAP_ERR_FILT, MAX_WRAP_ERR_FILT);  // actual update time provided run time
  WrapHi = new TFDelay(false, WRAP_HI_SET, WRAP_HI_RES, EKF_NOM_DT);  // Wrap test persistence.  Initializes false
  WrapLo = new TFDelay(false, WRAP_LO_SET, WRAP_LO_RES, EKF_NOM_DT);  // Wrap test persistence.  Initializes false
  QuietFilt = new General2_Pole(T, WN_Q_FILT, ZETA_Q_FILT, MIN_Q_FILT, MAX_Q_FILT);  // actual update time provided run time
  QuietRate = new RateLagExp(T, TAU_Q_FILT, MIN_Q_FILT, MAX_Q_FILT);
  WrapLoopAmp = new Looparound(Mon, Sen, wrap_hi_volt_, wrap_lo_volt_, AMP_WRAP_TRIM_GAIN, -IB_ABS_MAX_AMP, IB_ABS_MAX_AMP,
                              MAX_WRAP_ERR_FILT/(IB_ABS_MAX_NOA/IB_ABS_MAX_AMP));
  WrapLoopNoa = new Looparound(Mon, Sen, wrap_hi_noa_, wrap_lo_noa_, NOA_WRAP_TRIM_GAIN, -IB_ABS_MAX_NOA, IB_ABS_MAX_NOA,
                              MAX_WRAP_ERR_FILT);
}

// Coulomb Counter difference test - failure conditions track poorly
void Fault::cc_diff(const bool reset, Sensors *Sen, BatteryMonitor *Mon)
{
  cc_diff_ = Mon->soc_ekf() - Mon->soc(); // These are filtered in their construction (EKF is a dynamic filter and 
                                          // Coulomb counter is wrapa big integrator)
  if ( Mon->soc() <= max(Mon->soc_min()+WRAP_SOC_LO_OFF_REL, WRAP_SOC_LO_OFF_ABS) )
  {
    cc_diff_empty_slr_ = CC_DIFF_LO_SOC_SLR;
  }
  else
  {
    cc_diff_empty_slr_ = 1.;
  }
  // ewsat_slr_ used here because voc_soc map inaccurate on cold days
  cc_diff_thr_ = CC_DIFF_SOC_DIS_THRESH*ap.cc_diff_slr()*cc_diff_empty_slr_*ewsat_slr_;
  failAssign( CcdiffPer->calculate(abs(cc_diff_)>=cc_diff_thr_, CC_DIFF_SET, CC_DIFF_RES, Sen->T(), reset), CC_DIFF_FA ); // CC_DIFF_FA not latched
}

// Compare current sensors - failure conditions large difference
void Fault::ib_diff(const bool reset, Sensors *Sen, BatteryMonitor *Mon)
{
  bool reset_loc = reset || reset_all_faults_;
  if ( disable_amp_fault_ ) ib_diff_ = ib_diff_;
  else if ( ib_lo_limited_hi_ ) ib_diff_ = max(0., ib_diff_);  // limit error when low amp is pegged high
  else if ( ib_lo_limited_lo_ ) ib_diff_ = min(0., ib_diff_);  // limit error when low amp is pegged low
  ib_diff_f_ = IbDiffFilt->calculate(ib_diff_, reset_loc || disable_amp_fault_ || ib_lo_limited_hi_ || ib_lo_limited_lo_, Sen->T());
  ib_diff_thr_ = IBATT_DISAGREE_THRESH*ap.ib_diff_slr();
  faultAssign( IbdPosPer->calculate((ib_diff_f_>=ib_diff_thr_), IBATT_INST_DIFF_SET, IBATT_INST_DIFF_RES, Sen->T(), reset_loc),
    IB_DIFF_HI_FLT );
  faultAssign( IbdNegPer->calculate((ib_diff_f_<=-ib_diff_thr_), IBATT_INST_DIFF_SET, IBATT_INST_DIFF_RES, Sen->T(), reset_loc),
    IB_DIFF_LO_FLT );
  failAssign( IbdHiPer->calculate(ib_diff_hi_flt(), IBATT_DISAGREE_SET, IBATT_DISAGREE_RES, Sen->T(), reset_loc),
    IB_DIFF_HI_FA ); // IB_DIFF_FA not latched
  failAssign( IbdLoPer->calculate(ib_diff_lo_flt(), IBATT_DISAGREE_SET, IBATT_DISAGREE_RES, Sen->T(), reset_loc),
    IB_DIFF_LO_FA ); // IB_DIFF_FA not latched

  // if ( sp.debug()==2 || sp.debug()==4 ) Serial.printf("ib_diff_%7.3f reset_loc %d disable_amp_fault_ %d ib_diff_f_ %7.3f ib_diff_thr_ %7.3f ib_lo_active_ %d\n",
  //    ib_diff_, reset_loc, disable_amp_fault_, ib_diff_f_, ib_diff_thr_, ib_lo_active_);
}

// Compare current sensors - failure conditions large difference
void Fault::ib_logic(const bool reset, Sensors *Sen, BatteryMonitor *Mon)
{
  bool reset_loc = reset || reset_all_faults_;

  // Difference error, filter, check, persist, doesn't latch
  if ( sp.mod_ib() )
  {
    ib_diff_ = Sen->ib_amp_model() - Sen->ib_noa_model();
    #ifdef HDWE_IB_HI_LO
      ib_amp_hi_ = Sen->ib_amp_model() >= HDWE_IB_HI_LO_AMP_HI / ap.nP();
      ib_amp_lo_ = Sen->ib_amp_model() <= HDWE_IB_HI_LO_AMP_LO / ap.nP();
      ib_noa_hi_ = Sen->ib_noa_model() >= HDWE_IB_HI_LO_NOA_HI / ap.nP();
      ib_noa_lo_ = Sen->ib_noa_model() <= HDWE_IB_HI_LO_NOA_LO / ap.nP();
      ib_lo_limited_hi_ = IbLoLimitedHi->calculate(ib_amp_hi_, IB_LO_ACTIVE_SET*cp.ts, IB_LO_ACTIVE_RES*cp.ts,
                                                   Sen->T() , reset_loc);
      ib_lo_limited_lo_ = IbLoLimitedLo->calculate(ib_amp_lo_, IB_LO_ACTIVE_SET*cp.ts, IB_LO_ACTIVE_RES*cp.ts,
                                                   Sen->T() , reset_loc);
      ib_lo_active_ = !ib_lo_limited_hi_ && !ib_lo_limited_lo_;
    #else
      ib_amp_hi_ = false;
      ib_amp_lo_ = false;
      ib_noa_hi_ = false;
      ib_noa_lo_ = false;
      ib_lo_limited_hi_ = false;
      ib_lo_limited_lo_ = false;
      ib_lo_active_ = false;
    #endif
  }
  else
  {
    ib_diff_ = Sen->ib_amp_hdwe() - Sen->ib_noa_hdwe();
    #ifdef HDWE_IB_HI_LO
      ib_amp_hi_ = Sen->ib_amp_hdwe() >= HDWE_IB_HI_LO_AMP_HI / ap.nP();
      ib_amp_lo_ = Sen->ib_amp_hdwe() <= HDWE_IB_HI_LO_AMP_LO / ap.nP();
      ib_noa_hi_ = Sen->ib_noa_hdwe() >= HDWE_IB_HI_LO_NOA_HI / ap.nP();
      ib_noa_lo_ = Sen->ib_noa_hdwe() <= HDWE_IB_HI_LO_NOA_LO / ap.nP();
      ib_lo_limited_hi_ = IbLoLimitedHi->calculate(ib_amp_hi_, IB_LO_ACTIVE_SET*cp.ts, IB_LO_ACTIVE_RES*cp.ts,
                                                   Sen->T() , reset_loc);
      ib_lo_limited_lo_ = IbLoLimitedLo->calculate(ib_amp_lo_, IB_LO_ACTIVE_SET*cp.ts, IB_LO_ACTIVE_RES*cp.ts,
                                                   Sen->T() , reset_loc);
      ib_lo_active_ =    !ib_lo_limited_hi_ && !ib_lo_limited_lo_;
    #else
      ib_amp_hi_ = false;
      ib_amp_lo_ = false;
      ib_noa_hi_ = false;
      ib_noa_lo_ = false;
      ib_lo_active_ = false;
    #endif
  }
  disable_amp_fault_ = (ib_amp_hi_ && ib_noa_hi_) || (ib_amp_lo_ && ib_noa_lo_);

}

// Detect no signal present based on detection of quiet signal.
// Research by sound industry found that 2-pole filtering is the sweet spot between seeing noise
// and actual motion without 'guilding the lily'
void Fault::ib_quiet(const bool reset, Sensors *Sen)
{
  bool reset_loc = reset | reset_all_faults_;

  // Rate (has some filtering)
  if ( !sp.mod_ib() )
  {
    ib_rate_ = QuietRate->calculate(Sen->Ib_amp_hdwe() + Sen->Ib_noa_hdwe(), reset, min(Sen->T(), MAX_T_Q_FILT));
    // 2-pole filter
    ib_quiet_ = QuietFilt->calculate(ib_rate_, reset_loc, min(Sen->T(), MAX_T_Q_FILT));
    ib_quiet_thr_ = QUIET_A * ap.ib_quiet_slr();
    ib_is_quiet_ = abs(ib_quiet_)<=ib_quiet_thr_ && !reset_loc;
    ib_is_functional_ = QuietPerFunc->calculate(!ib_is_quiet_, QUIET_SET, QUIET_RES, Sen->T(), reset_loc);
    // Really Quiet logic added for robust (no faults) during BMS shutoff
    ib_really_quiet_ = ib_is_quiet_ && ( abs(Sen->Ib_amp_hdwe()+Sen->Ib_noa_hdwe()) < LOW_A );
  }
  else
  {
    ib_rate_ = QuietRate->calculate(Sen->Ib_amp_model() + Sen->Ib_noa_model(), reset, min(Sen->T(), MAX_T_Q_FILT));
    // 2-pole filter
    ib_quiet_ = QuietFilt->calculate(ib_rate_, reset_loc, min(Sen->T(), MAX_T_Q_FILT));
    ib_quiet_thr_ = QUIET_A * ap.ib_quiet_slr();
    ib_is_quiet_ = abs(ib_quiet_)<=ib_quiet_thr_ && !reset_loc;
    ib_is_functional_ = QuietPerFunc->calculate(!ib_is_quiet_, QUIET_SET, QUIET_RES, Sen->T(), reset_loc);
    // Really Quiet logic added for robust (no faults) during BMS shutoff
    ib_really_quiet_ = ib_is_quiet_ && ( abs(Sen->Ib_amp_model()+Sen->Ib_noa_model()) < LOW_A );
  }

  if ( sp.debug()==21 )
    sendTxBuf(String::format("Isum %8.3f ib_quiet %8.3f ib_quiet_thr %8.3f ib_is_quiet %d ib_is_func %d ib_really_quiet %d\n",
      Sen->Ib_amp_hdwe() + Sen->Ib_noa_hdwe(), ib_quiet_, ib_quiet_thr_, ib_is_quiet_, ib_is_functional_, ib_really_quiet_), true, IN_SERVICE);

      // Fault
  faultAssign( ib_is_quiet_, IB_DSCN_FLT );   // initializes false
  failAssign( QuietPer->calculate(dscn_flt(), QUIET_SET, QUIET_RES, Sen->T(), reset_loc), IB_DSCN_FA);
  debug_check_m13(Sen);
  debug_check_m23(Sen);
  debug_check_m24(Sen);
}

// Range checks latch
void Fault::ib_range(const bool reset, Sensors *Sen, BatteryMonitor *Mon)
{
  bool reset_loc = reset | reset_all_faults_;
  if ( reset_loc )
  {
    failAssign(false, IB_AMP_FA);
    failAssign(false, IB_NOA_FA);
  }
  faultAssign( Sen->ShuntAmp->bare_shunt(), IB_AMP_BARE);
  faultAssign( Sen->ShuntNoAmp->bare_shunt(), IB_NOA_BARE);

  // Range checks latch
  if ( sp.mod_ib() )
  {
    faultAssign( ( abs(Sen->ib_amp_model()) >= ap.ib_amp_max() ) && !ap.disab_ib_fa() && !sp.tweak_test(), IB_AMP_FLT );
    faultAssign( ( abs(Sen->ib_noa_model()) >= ap.ib_noa_max() ) && !ap.disab_ib_fa() && !sp.tweak_test(), IB_NOA_FLT );
  }
  else
  {
    #ifndef HDWE_BARE
      faultAssign( ( ib_amp_bare() || abs(Sen->ib_amp_hdwe()) >= ap.ib_amp_max() ) && !ap.disab_ib_fa() && !sp.tweak_test(), IB_AMP_FLT );
      faultAssign( ( ib_noa_bare() || abs(Sen->ib_noa_hdwe()) >= ap.ib_noa_max() ) && !ap.disab_ib_fa() && !sp.tweak_test(), IB_NOA_FLT );
    #else
      float current_max = NOM_UNIT_CAP * ap.nP();
      faultAssign( abs(Sen->ShuntAmp->Ishunt_cal()) >= current_max && !ap.disab_ib_fa() && !sp.tweak_test(), IB_AMP_FLT );
      faultAssign( abs(Sen->ShuntNoAmp->Ishunt_cal()) >= current_max && !ap.disab_ib_fa() && !sp.tweak_test(), IB_NOA_FLT );
    #endif
  }

  // Fail persistence
  if ( ap.disab_ib_fa() )
  {
    failAssign( false, IB_AMP_FA );
    failAssign( false, IB_NOA_FA);
  }
  else
  {
    failAssign( vc_fa() || ib_amp_bare() || ib_amp_fa() || IbAmpHardFail->calculate(ib_amp_flt(), IB_HARD_SET, IB_HARD_RES, Sen->T(), reset_loc), IB_AMP_FA );
    failAssign( vc_fa() || ib_noa_bare() || ib_noa_fa() || IbNoAmpHardFail->calculate(ib_noa_flt(), IB_HARD_SET, IB_HARD_RES, Sen->T(), reset_loc), IB_NOA_FA);
  }
  #ifdef DEBUG_INIT
    if ( sp.mod_ib() )
    {
      if ( sp.debug()==62 ) Serial.printf("ibnoamod%7.3f ibampmod%7.3f ib_lo_active %d\n", Sen->Ib_noa_model(), Sen->Ib_amp_model(), ib_lo_active_);
      if ( sp.debug()==62 ) Serial.printf("ib_amp_model %7.3f mx %7.3f ib_noa_model %7.3f nx %7.3f IB_AMP_FLT %d IB_NOA_FLT %d\n", Sen->ib_amp_model(), ap.ib_amp_max(), Sen->ib_noa_model(), ap.ib_noa_max(), ib_amp_flt(), ib_noa_flt());
    }
    else
    {
      if ( sp.debug()==62 ) Serial.printf("ibnoahdwe%7.3f ibamphdwe%7.3f ib_lo_active %d\n", Sen->Ib_noa_hdwe(), Sen->Ib_amp_hdwe(), ib_lo_active_);
      if ( sp.debug()==62 ) Serial.printf("ib_amp_bare=%d ib_noa_bare=%d ib_model%7.3f mx%7.3f ibn%7.3f nx%7.3f IB_AMP_FLT=%d IB_NOA_FLT%d ib_lo_active%d\n", ib_amp_bare(), ib_noa_bare(), Sen->ib_amp_hdwe(), ap.ib_amp_max(), Sen->ib_noa_hdwe(), ap.ib_noa_max(), ib_amp_flt(), ib_noa_flt(), ib_lo_active_);
    }
  #endif
}

// Voltage wraparound logic for current selection
// Avoid using hysteresis data for this test and accept more generous thresholds
void Fault::ib_wrap(const bool reset, Sensors *Sen, BatteryMonitor *Mon)
{
  bool reset_loc = reset | reset_all_faults_;
  if ( reset_loc )
  {
    failAssign(false, WRAP_VB_FA);
  }

  // Thresholds
  wrap_scalars(Mon);

  // ib section of wrap logic - separate because has multiple sensors and complex selection logic
  // HI_LO-Only Logic
  #ifdef HDWE_IB_HI_LO
    WrapLoopNoa->calculate(reset_loc, false, Sen->ib_noa(), Sen, false);
    WrapLoopAmp->calculate(reset_loc, disable_amp_fault_, Sen->ib_amp(), Sen, !ib_lo_active_);
    faultAssign( WrapLoopAmp->hi_fault(), WRAP_HI_M_FLT);
    failAssign( WrapLoopAmp->hi_fail(), WRAP_HI_M_FA);  // WRAP_HI_M_FA not latched
    faultAssign( WrapLoopAmp->lo_fault(), WRAP_LO_M_FLT);
    failAssign( WrapLoopAmp->lo_fail(), WRAP_LO_M_FA);  // WRAP_LO_M_FA not latched
    faultAssign( WrapLoopNoa->hi_fault(), WRAP_HI_N_FLT);
    failAssign( WrapLoopNoa->hi_fail(), WRAP_HI_N_FA);  // WRAP_HI_N_FA not latched
    faultAssign( WrapLoopNoa->lo_fault(), WRAP_LO_N_FLT);
    failAssign( WrapLoopNoa->lo_fail(), WRAP_LO_N_FA);  // WRAP_LO_N_FA not latched
  #endif

  // Overall wrap logic (separates amp/noa and hi/lo)
  #ifdef HDWE_IB_HI_LO
    e_wrap_ = scale_select(Sen->Ib_noa_hdwe(), Sen->sel_brk_hdwe, WrapLoopAmp->e_wrap(), WrapLoopNoa->e_wrap());
    e_wrap_filt_ = scale_select(Sen->Ib_noa_hdwe(), Sen->sel_brk_hdwe, WrapLoopAmp->e_wrap_filt(), WrapLoopNoa->e_wrap_filt());
    e_wrap_rate_ = scale_select(Sen->Ib_noa_hdwe(), Sen->sel_brk_hdwe, WrapLoopAmp->e_wrap_rate(), WrapLoopNoa->e_wrap_rate());
    faultAssign( ( wrap_hi_m_flt() && wrap_hi_n_flt() && !Mon->sat() ), WRAP_HI_FLT);
    faultAssign( ( wrap_lo_m_flt() && wrap_lo_n_flt() ), WRAP_LO_FLT);
    failAssign( ( wrap_hi_m_fa() && wrap_hi_n_fa() && !Mon->sat() ), WRAP_HI_FA);
    failAssign( ( wrap_lo_m_fa() && wrap_lo_n_fa() ), WRAP_LO_FA);
  #else
    e_wrap_ = Mon->voc_soc() - Mon->voc_stat();
    e_wrap_filt_ = WrapErrFilt->calculate(e_wrap_, reset_loc, Sen->T());
    e_wrap_rate_ = WrapErrFilt->rate();
    // sat logic screens out voc jumps when ib>0 when saturated
    // wrap_hi and wrap_lo don't latch because need them available to check next ib sensor selection for dual ib sensor
    // wrap_vb latches because vb is single sensor
    // Thresholds calculated by wrap_scalars()
    faultAssign( (e_wrap_filt_ >= ewhi_thr_ && !Mon->sat()), WRAP_HI_FLT);
    faultAssign( (e_wrap_filt_ <= ewlo_thr_), WRAP_LO_FLT);
    failAssign( (WrapHi->calculate(wrap_hi_flt(), WRAP_HI_SET, WRAP_HI_RES, Sen->T(), reset_loc) && !vb_fa_lt()), WRAP_HI_FA );  // not latched
    failAssign( (WrapLo->calculate(wrap_lo_flt(), WRAP_LO_SET, WRAP_LO_RES, Sen->T(), reset_loc) && !vb_fa_lt()), WRAP_LO_FA );  // not latched
  #endif

  // vb section of wrap logic - separate because vb is single sensor and can latch
  failAssign( ( wrap_vb_fa() && !reset_loc ) ||
              ( !ib_diff_fa() && wrap_m_and_n_fa() && ib_really_quiet() ),
               WRAP_VB_FA);    // WRAP_VB_FA latches latches because vb is single sensor
}

void Fault::pretty_print(Sensors *Sen, BatteryMonitor *Mon)
{
  String txBuf;

  txBuf = String::format("\nLooparound Amp:\n");
  sendTxBuf(txBuf, true, IN_SERVICE);
  txBuf = WrapLoopAmp->pretty_print(Sen);
  sendTxBuf(txBuf, true, IN_SERVICE);

  txBuf = String::format("\nLooparound Noa:\n");
  sendTxBuf(txBuf, true, IN_SERVICE);
  txBuf = WrapLoopNoa->pretty_print(Sen);
  sendTxBuf(txBuf, true, IN_SERVICE);

  txBuf = String::format("\nFault:\n") +
    String::format(" cc_diff%9.6f  thr%9.6f Fc^\n", cc_diff_, cc_diff_thr_) +
    String::format(" ib_lo_limited_hi %d\n", ib_lo_limited_hi_) +
    String::format(" ib_lo_active     %d\n", ib_lo_active_) +
    String::format(" ib_lo_limited_lo %d\n", ib_lo_limited_lo_) +
    String::format(" ib_diff%7.3f thr%7.3f Fd^\n", ib_diff_f_, ib_diff_thr_) +
    String::format(" e_wrap_filt%7.3f\n", e_wrap_filt_) +
    String::format(" ib_quiet%7.3f thr%7.3f Fq v\n", ib_quiet_, ib_quiet_thr_) +
    String::format(" sel_brk_hdwe:     ");
  sendTxBuf(txBuf, true, IN_SERVICE);

  txBuf = Sen->sel_brk_hdwe->pretty_print() +
    String::format("\n");
  sendTxBuf(txBuf, true, IN_SERVICE);

  txBuf = String::format(" soc%7.3f soc_inf%7.3f voc%7.3f  voc_soc%7.3f\n", Mon->soc(), Mon->soc_inf(), Mon->voc(), Mon->voc_soc()) +
    String::format(" dis_tb_fa %d  dis_vb_fa %d  dis_ib_fa %d\n", ap.disab_tb_fa(), ap.disab_vb_fa_lt(), ap.disab_ib_fa()) +
    String::format(" ib_is_quiet %d ib_really_quiet %d\n", ib_is_quiet_, ib_really_quiet_) +
    String::format(" bms_off  %d\n\n", Mon->bms_off()) +
    String::format(" wrap_m_and_n_fa %d\n", Sen->Flt->wrap_m_and_n_fa()) +
    String::format(" Tbh%9.5f Tbm=%9.5f sel%9.5f\n", Sen->Tb_hdwe(), Sen->Tb_model(), Sen->Tb()) +
    String::format(" Tbh%9.5f Tbm=%9.5f sel%9.5f\n", Sen->Tb_hdwe(), Sen->Tb_model(), Sen->Tb()) +
    String::format(" Vbh%7.3f Vbm %7.3f sel%7.3f\n", Sen->Vb_hdwe(), Sen->Vb_model(), Sen->Vb()) +
    String::format(" Vc_h_m%7.3f\n", Sen->ShuntAmp->Vc()) +
    String::format(" Vc_h_n%7.3f\n", Sen->ShuntNoAmp->Vc()) +
    String::format(" Imh%7.3f Imm %7.3f Ib%7.3f\n", Sen->Ib_amp_hdwe(), Sen->Ib_amp_model(), Sen->Ib()) +
    String::format(" Inh%7.3f Inm %7.3f Ib%7.3f\n", Sen->Ib_noa_hdwe(), Sen->Ib_noa_model(), Sen->Ib()) +
    String::format(" Ibh%7.3f Ibh %7.3f Ib%7.3f\n\n", Sen->Ib_hdwe(), Sen->Ib_hdwe_model(), Sen->Ib());
  sendTxBuf(txBuf, true, IN_SERVICE);

  // if ( ib_choice_ != ib_choice_last_ || vb_sel_stat_ != vb_sel_stat_last_ || tb_sel_stat_ != tb_sel_stat_last_ )
  debug_qs(Mon, Sen);
  
txBuf = String::format("") +
  #ifdef HDWE_IB_HI_LO
    String::format("HDWE_IB_HI_LO Decisions\n") +
  #else
    String::format(("Active/Standby Decisions\n") +
  #endif
    String::format("       Fault  Fail'\n") +
    String::format("1 wnl     %d  %d 'Fo ^'\n", wrap_lo_n_flt(), wrap_lo_n_fa()) +
    String::format("0 wnh     %d  %d 'Fi ^'\n", wrap_hi_n_flt(), wrap_hi_n_fa()) +
    String::format("F wml     %d  %d 'Fo ^'\n", wrap_lo_m_flt(), wrap_lo_m_fa()) +
    String::format("E wmh     %d  %d 'Fi ^'\n", wrap_hi_m_flt(), wrap_hi_m_fa()) +
    String::format("D vc      %d  %d 'FI 1'\n", vc_flt(), vc_fa()) +
    String::format("C bare n  %d  x \n", ib_noa_bare()) +
    String::format("B bare m  %d  x \n", ib_amp_bare()) +
    String::format("A ib_dsc  %d  %d 'Fq v'\n", ib_dscn_flt(), ib_dscn_fa()) +
    String::format("9 ibd_lo  %d  %d 'Fd ^ SA/SB'\n", ib_diff_lo_flt(), ib_diff_lo_fa()) +
    String::format("8 ibd_hi  %d  %d 'Fd ^ SA/SB'\n", ib_diff_hi_flt(), ib_diff_hi_fa()) +
    String::format("7 red wv  %d  %d   'Fd, Fi/Fo ^'\n",  red_loss(), wrap_vb_fa()) +
    String::format("6 wl      %d  %d 'Fo ^'\n", wrap_lo_flt(), wrap_lo_fa()) +
    String::format("5 wh      %d  %d 'Fi ^'\n", wrap_hi_flt(), wrap_hi_fa()) +
    String::format("4 xx | cc_dif x  %d 'x Fc ^'\n", cc_diff_fa()) +
    String::format("3 ib n    %d  %d 'FI 1'\n", ib_noa_flt(), ib_noa_fa()) +
    String::format("2 ib m    %d  %d 'FI 1'\n", ib_amp_flt(), ib_amp_fa()) +
    String::format("1 vb      %d  %d 'Fv 1  SV, *Dc/*Dv'.", vb_flt(), vb_fa_lt()) +
    String::format("  bms_off %d\n", Mon->bms_off()) +
    String::format("0 Tb      %d  %d 'Ft 1'\n\n", Tb_flt(), Tb_fa()) +
    // String::format("B-time_long%2d\n", dispRead(time_long)) +  // distracting
    String::format("A-accy     %2d\n", dispRead(accy)) +
    String::format("9-off      %2d\n", dispRead(off)) +
    String::format("8-SAT      %2d\n", dispRead(SAT)) +
    String::format("7-flt_ekf  %2d\n", dispRead(flt_ekf)) +
    String::format("6-flt_tb   %2d\n", dispRead(flt_tb)) +
    String::format("*5-fail_vb %2d\n", dispRead(fail_vb)) +
    String::format("*4-fail_ibm%2d\n", dispRead(fail_ibm)) +
    String::format("*3-fail_ib %2d\n", dispRead(fail_ib)) +
    String::format("2-red_loss %2d\n", dispRead(dispw::red_loss)) +
    String::format("1-diff_ib  %2d\n", dispRead(diff_ib)) +
    String::format("0-conn     %2d\n\n", dispRead(conn)); 
  sendTxBuf(txBuf, true, IN_SERVICE);
  // enum dispw {conn=0, diff_ib=1, red_loss=2, fail_ib=3, fail_ibm=4, fail_vb=5, flt_tb=6, flt_ekf=7, SAT=8, off=9, accy=10, time_long=11, Count};

  txBuf = bitMapPrint(pr.buff, fltw_, NUM_FLT) +
    String::format("   ") +
    bitMapPrint(pr.buff, falw_, NUM_FA) +
    String::format("   ") +
    bitMapPrint(pr.buff, cp.disp_word, static_cast<int>(dispw::Count)) +
    String::format("\n") +
    String::format("10FEDCBA9876543210   10FExxBA9876543210   BA9876543210\n\n") +
    String::format("  fltw=%8ld       falw=%8ld         dispw=%8ld\n",
      fltw_, falw_, cp.disp_word);
  sendTxBuf(txBuf, true, IN_SERVICE);

  if ( ap.fake_faults() )
  {
    txBuf = String::format("fake_faults=>redl\n");
    sendTxBuf(txBuf, true, IN_SERVICE);
  }

  if ( sp.Time_now() < 1746684000UL )
  {
    txBuf = String::format("\n\n////////////////// WARN set UT (h;) %lu < %lu\n\n", sp.Time_now(), 1746684000UL);
    sendTxBuf(txBuf, true, IN_SERVICE);
  }
}

// Calculate selection for ib_decision_
// Use model instead of sensors when running tests as user
// Equivalent to using voc(soc) as voter between two hardware currrents
// Over-ride sensed Ib, Vb and Tb with model when running tests
// Inputs:  Sen->Ib_model, Sen->Ib_hdwe,
//          Sen->Vb_model, Sen->Vb_hdwe,
//          ----------------, Sen->Tb_hdwe, Sen->Tb_hdwe_f
// Outputs: Ib,
//          Vb,

//          Tb, Tb_f
//          latch_
void Fault::select_all_logic(Sensors *Sen, BatteryMonitor *Mon, const bool reset)
{
  // Reset
  if ( reset_all_faults_ )
  {
    reset_all_faults_select();
    Serial.printf("reset ib flt\n");
    Serial.printf("reset vb flt\n");
  }

  // Ib decision tables
  #ifdef HDWE_IB_HI_LO
    ib_decision_hi_lo(Sen);
    if ( ap.fake_faults() )
    {
      latch_fake_ = latch_;
      latch_ = false;
      ib_choice_ = ibSel(sp.ib_force());
    }
  #else
    ib_decision_active_standby(Sen);
    if ( ap.fake_faults() )
    {
      latch_fake_ = latch_;
      latch_ = false;
      ib_sel_stat_ = sp.ib_force();
    }
  #endif

  // vb failure from wrap result
  if ( !ap.fake_faults() )
  {
    if ( !vb_sel_stat_last_ && !sp.mod_vb() )
    {
      vb_sel_stat_ = 0;   // Latches
      latch_ = true;
    }
    if (  wrap_vb_fa() || vb_fa_lt() )
    {
      vb_sel_stat_ = 0; // Latches
      latch_ = true;
    }
  }
  else  // fake_faults
  {
    if ( !vb_sel_stat_last_ )
    {
      latch_fake_ = true;
    }
    if (  wrap_vb_fa() || vb_fa_lt() )
    {
      latch_fake_ = true;
    }
  }

  // Tb failure from inactivity. Does not latch because can heal and failure not critical
  if ( reset_all_faults_ )
  {
    tb_sel_stat_last_ = 1;
    tb_sel_stat_ = 1;
    Serial.printf("reset tb flts\n");
    failAssign(false, TB_FA);
  }
  if ( Tb_fa() )  // Latches
  {
    tb_sel_stat_ = 0;
    latch_ = true;
  }
  else
    tb_sel_stat_ = 1;

  // Print
   if ( ib_choice_ != ib_choice_last_ || vb_sel_stat_ != vb_sel_stat_last_ || tb_sel_stat_ != tb_sel_stat_last_ )
    debug_qs(Mon, Sen);
  #ifndef HDWE_IB_HI_LO
    if ( ib_sel_stat_ != ib_sel_stat_last_ || vb_sel_stat_ != vb_sel_stat_last_ || tb_sel_stat_ != tb_sel_stat_last_ )
    {
      Serial.printf("Small reset\n");
      cp.cmd_reset();
    }
  #endif

  // Latch memory
  ib_choice_last_ = ib_choice_;
  ib_sel_stat_last_ = ib_sel_stat_;
  vb_sel_stat_last_ = vb_sel_stat_;
  tb_sel_stat_last_ = tb_sel_stat_;

  // Make sure Rf command gets executed at least once all fault logic.  Asynchronous hence counter
  static uint8_t count = 0;
  reset_all_faults_print_ = reset_all_faults_;
  if ( reset_all_faults_ )
  {
    if ( ( falw_==0 && fltw_==0 ) || count>1 )
    {
      reset_all_faults_ = false;
      latch_ = false;
      latch_fake_ = false;
      preserving(false);
      count = 0;
    }
    else
    {
      count++;
      Serial.printf("Rf%d\n", count);
    }
  }
}

// Select ib decision table active-standby
void Fault::ib_decision_active_standby(Sensors *Sen)
{
  if ( ap.fake_faults() )
  {
    ib_sel_stat_ = IB_SEL_STAT_DEF;
    latch_ = false;
    ib_decision_ = 10;
  }
  else if ( latch_ )
    // ib_decision_ = 0;
    {}
  else if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() )  // these separate inputs don't latch
  {
    ib_decision_ = 1;
    ib_sel_stat_ = 0;    // takes two not latched inputs to set and latch
    latch_ = true;
  }
  else if ( sp.ib_force()>0 && !Sen->Flt->ib_amp_fa() )
  {
    ib_decision_ = 2;
    ib_sel_stat_ = 1;
    latch_ = true;
  }
  else if ( ib_sel_stat_last_==-1 && !Sen->Flt->ib_noa_fa() && !reset_all_faults_ )  // latches
  {
    ib_decision_ = 3;
    ib_sel_stat_ = -1;
    latch_ = true;
  }
  else if ( sp.ib_force()<0 && !Sen->Flt->ib_noa_fa() && !reset_all_faults_)  // latches
  {
    ib_decision_ = 4;
    ib_sel_stat_ = -1;
    latch_ = true;
  }
  else if ( sp.ib_force()==0 )  // auto
  {
    if ( Sen->Flt->ib_amp_fa() && !Sen->Flt->ib_noa_fa() )  // these inputs don't latch
    {
      ib_decision_ = 5;
      ib_sel_stat_ = -1;
      latch_ = true;
    }
    else if ( ib_diff_fa() )  // this input doesn't latch
    {
      if ( vb_sel_stat_ && wrap_hi_or_lo_fa() )    // wrap_hi_or_lo_fa is not latched
      {
        ib_decision_ = 6;
        ib_sel_stat_ = -1;      // two not latched fails but result of 'and' with ib_diff_fa latches latched_fail
        latch_ = true;
      }
      else if ( cc_diff_fa() )  // this input doesn't latch but result of 'and' with ib_diff_fa latches latched_fail
      {
        ib_decision_ = 7;
        ib_sel_stat_ = -1;      // takes two not latched inputs to isolate ib failure and to set and latch ib_sel
        latch_ = true;
      }
    }
  }
  else if ( ( (sp.ib_force() <  0) && ib_sel_stat_last_>-1 ) ||
            ( (sp.ib_force() >= 0) && ib_sel_stat_last_< 1 )   )  // Latches.  Must reset to move out of no amp selection.  ==0 not reachable
  {
    ib_decision_ = 8;
    latch_ = true;
  }
  else
  {
    latch_ = false;
  }
  faultAssign(ib_sel_stat_!=1 || sp.ib_force()!=0  || ib_diff_fa() || ib_amp_fa() || ib_noa_fa() || vb_fail(), RED_LOSS); // for active-standby, redundancy loss anytime ib_sel_stat<0

  #ifdef DEBUG_INIT
    if ( sp.debug()==62 ) Serial.printf("fake_faults %d ib_force %d reset %d ib_sel_stat_last %d ib_amp_fa %d ib_noa_fa %d ib_diff_fa %d vb_sel_stat_last %d wrap_m_fa %d wrap_n_fa %d  cc_diff_fa %d latch_ %d ib_sel_stat %d ib_decision_ %d\n", ap.fake_faults(), sp.ib_force(), reset_all_faults_, ib_sel_stat_last_, ib_amp_fa(), ib_noa_fa(), ib_diff_fa(), vb_sel_stat_last_, wrap_m_fa(), wrap_n_fa(), cc_diff_fa(), latch_, ib_sel_stat_, ib_decision_);
  #endif
}

// Select ib decision table hi-lo
// Inputs:  ib_amp_fa, ib_noa_fa, ib_force, ib_diff_fa, vb_sel_stat_last_, wrap_m_fa, wrap_n_fa, cc_diff_fa, wrap_hi_or_lo_fa
// Outputs:  ib_decision_, ib_choice_, latch_
void Fault::ib_decision_hi_lo(Sensors *Sen)
{
  bool latch_last = latch_;
  // Dual hard-fail takes precedence over a prior single-failure latch: if the survivor
  // also fails hard, take both ib signals off-line (UsingNone) so downstream stages
  // zero ib into the Coulomb counter and the EKF.
  if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() )  // these separate inputs don't latch
  {
    ib_choice_ = UsingNone;
    latch_ = true;
    ib_decision_ = 1;
  }
  else if ( latch_ )
    // ib_decision_ = xx;   lgv
    {}
  else if ( sp.ib_force()>0 && !Sen->Flt->ib_noa_fa() )
  {
    ib_choice_ = UsingAmp;
    latch_ = true;
    ib_decision_ = 2;
  }
  else if ( sp.ib_force()<0 && !Sen->Flt->ib_noa_fa() && !reset_all_faults_)  // latches
  {
    ib_choice_ = UsingNoa;
    latch_ = true;
    ib_decision_ = 3;
  }
  else if ( sp.ib_force()==0 )  // auto section
  {
    if ( Sen->Flt->ib_amp_fa() && !Sen->Flt->ib_noa_fa() )  // these inputs don't latch
    {
      ib_choice_ = UsingNoa;
      latch_ = true;
      ib_decision_ = 4;
    }
    else if ( !Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() )  // these inputs don't latch
    {
      ib_choice_ = UsingAmp;
      latch_ = true;
      ib_decision_ = 5;
    }
    else if ( ib_diff_fa() )  // this input doesn't latch
    {
      if ( vb_sel_stat_last_ )
      {
        if ( Sen->Flt->wrap_m_fa() && !Sen->Flt->wrap_n_fa() )
        {
          ib_choice_ = UsingNoa;
          latch_ = true;
          ib_decision_ = 6;
        }
        else if ( !Sen->Flt->wrap_m_fa() && Sen->Flt->wrap_n_fa() )
        {
          ib_choice_ = UsingAmp;
          latch_ = true;
          ib_decision_ = 7;
        }
        else if ( Sen->Flt->wrap_m_fa() && Sen->Flt->wrap_n_fa() )
        {
          ib_choice_ = KeepTrying;  // ambiguous; keep trying
          latch_ = false;
          ib_decision_ = 8;
        }
        else if ( cc_diff_fa() ) // isolated
        {
          ib_choice_ = UsingNoa; 
          latch_ = true;
          ib_decision_ = 10;
        }
        else  // all's well
        {
          ib_choice_ = ib_choice_last_;
          latch_ = latch_last;
          ib_decision_ = 0;
        }
      }
      else if ( cc_diff_fa() )  // don't know how to isolate due to weighting of amp and noa
      {
        ib_choice_ = KeepTrying;  // ambiguous; keep trying
        latch_ = false;
        ib_decision_ = 10;
      }
      else  // all's well
      {
        ib_choice_ = ib_choice_last_;
        latch_ = latch_last;
        ib_decision_ = 0;
      }
    }
    else if ( cc_diff_fa() )  // don't know how to isolate due to weighting of amp and noa
    {
        ib_choice_ = KeepTrying;  // ambiguous; keep trying
        latch_ = false;
        ib_decision_ = 12;
    }
    else  // all's well
    {
      ib_choice_ = ib_choice_last_;
      latch_ = latch_last;
      ib_decision_ = 0;
    }
  }
  else if ( ( (sp.ib_force() <  0) && ib_choice_last_!=UsingNoa ) ||
            ( (sp.ib_force() >= 0) && ib_choice_last_!=UsingAmp )   )  // Latches.  Must reset to move out of no amp selection.  ==0 not reachable
  {
    latch_ = true;
    ib_decision_ = 14;
  }
  else
  {
    latch_ = false;
    ib_decision_ = 15;
  }
  faultAssign( (ib_choice_!=0 || vb_sel_stat_!=1) && !(sp.mod_ib() || sp.mod_vb()), RED_LOSS);  // hi_lo

  #ifdef DEBUG_INIT
    if ( sp.debug()==62 ) Serial.printf("latch_last %d fake_faults %d ib_force %d reset %d ib_choice_last %d ib_amp_fa %d ib_noa_fa %d ib_diff_fa %d vb_sel_stat_last %d wrap_m_fa %d wrap_n_fa %d  cc_diff_fa %d latch_ %d ib_choice_ %d ib_decision_ %d\n", latch_last, ap.fake_faults(), sp.ib_force(), reset_all_faults_, ib_choice_last_, ib_amp_fa(), ib_noa_fa(), ib_diff_fa(), vb_sel_stat_last_, wrap_m_fa(), wrap_n_fa(), cc_diff_fa(), latch_, ib_choice_, ib_decision_);
  #endif
}

// Select reset
void Fault::reset_all_faults_select()
{
  // ib
  #ifdef HDWE_IB_HI_LO
    ib_choice_ = ibSel(sp.ib_force());
    ib_choice_last_ = ib_choice_;
  #else
    if ( sp.ib_force() >= 0 )
      ib_sel_stat_ = 1;
    else
      ib_sel_stat_ = -1;
    ib_sel_stat_last_ =  ib_sel_stat_;
  #endif

  // Reset latch memory
  vb_sel_stat_last_ = 1;
  vb_sel_stat_ = 1;
}

// Checks analog current.  Latches
void Fault::shunt_check(Sensors *Sen, BatteryMonitor *Mon, const bool reset)
{
  bool reset_loc = reset | reset_all_faults_;
  if ( reset_loc )
  {
    failAssign(false, IB_AMP_FA);
    failAssign(false, IB_NOA_FA);
  }
  faultAssign( Sen->ShuntAmp->bare_shunt(), IB_AMP_BARE);
  faultAssign( Sen->ShuntNoAmp->bare_shunt(), IB_NOA_BARE);
  #ifndef HDWE_BARE
    faultAssign( ( ib_amp_bare() || abs(Sen->ShuntAmp->Ishunt_cal()) >= Sen->Ib_amp_max() ) && !ap.disab_ib_fa(), IB_AMP_FLT );
    faultAssign( ( ib_noa_bare() || abs(Sen->ShuntNoAmp->Ishunt_cal()) >= Sen->Ib_noa_max() ) && !ap.disab_ib_fa(), IB_NOA_FLT );
    #ifdef DEBUG_INIT
      if ( sp.debug()==62 ) Serial.printf("ib_amp_bare=%d ib_noa_bare=%d ib_model%7.3f mX%7.3f Ibn%7.3f nX%7.3f IB_AMP_FLT=%d IB_NOA_FLT=%d\n", ib_amp_bare(), ib_noa_bare(), Sen->ShuntAmp->Ishunt_cal(), Sen->Ib_amp_max(), Sen->ShuntNoAmp->Ishunt_cal(), Sen->Ib_noa_max(), IB_AMP_FLT, IB_NOA_FLT);
    #endif
  #else
    float current_max = NOM_UNIT_CAP * ap.nP();
    faultAssign( abs(Sen->ShuntAmp->Ishunt_cal()) >= current_max && !ap.disab_ib_fa(), IB_AMP_FLT );
    faultAssign( abs(Sen->ShuntNoAmp->Ishunt_cal()) >= current_max && !ap.disab_ib_fa(), IB_NOA_FLT );
  #endif
  if ( ap.disab_ib_fa() )
  {
    failAssign( false, IB_AMP_FA );
    failAssign( false, IB_NOA_FA);
  }
  else
  {
    failAssign( vc_fa() || ib_amp_bare() || ib_amp_fa() || IbAmpHardFail->calculate(ib_amp_flt(), IB_HARD_SET, IB_HARD_RES, Sen->T(), reset_loc), IB_AMP_FA );
    failAssign( vc_fa() || ib_noa_bare() || ib_noa_fa() || IbNoAmpHardFail->calculate(ib_noa_flt(), IB_HARD_SET, IB_HARD_RES, Sen->T(), reset_loc), IB_NOA_FA);
  }
}


// Check Tb 2-wire analog voltage.  Latches
void Fault::Tb_check(Sensors *Sen, const float _tb_min, const float _tb_max, const bool reset)
{
  bool reset_loc = reset | reset_all_faults_;
  if ( reset_loc )
  {
    failAssign(false, TB_FA);
  }
  if ( sp.mod_tb() )
  {
    faultAssign( ((Sen->Tb_model()<=_tb_min) || (Sen->Tb_model()>=_tb_max)) &&
                 !ap.disab_tb_fa(), TB_FLT);
    failAssign( Tb_fa() || TbHardFail->calculate(Tb_flt(), TB_HARD_SET, TB_HARD_RES, Sen->T(), reset_loc), TB_FA);
  }
  else if ( ap.disab_tb_fa() )
  {
    faultAssign( false, TB_FLT);
    failAssign( false, TB_FA); }
  else
  {
    faultAssign( (Sen->Tb_hdwe()<=_tb_min) || (Sen->Tb_hdwe()>=_tb_max), TB_FLT);
    failAssign( Tb_fa() || TbHardFail->calculate(Tb_flt(), TB_HARD_SET, TB_HARD_RES, Sen->T(), reset_loc), TB_FA);
  }
  // if ( sp.debug()==18 ) Serial.printf("Tb_check: mod_tb %d disab_tb_fa %d Sen->Tb_model() %7.3f Sen->Tb_model_f() %7.3f Sen->Tb_hdwe() %7.3f Sen->Tb_hdwe_f() %7.3f _tb_min%7.3f _tb_max%7.3f TB_FLT %d TB_FA %d\n",
  //   sp.mod_tb(), ap.disab_tb_fa(), Sen->Tb_model(), Sen->Tb_model_f(), Sen->Tb_hdwe(), Sen->Tb_hdwe_f(), _tb_min, _tb_max, Tb_flt(), Tb_fa());
}

// Check analog voltage.  Latches
void Fault::vb_check(Sensors *Sen, BatteryMonitor *Mon, const float _vb_min, const float _vb_max, const bool reset)
{
  bool reset_loc = reset | reset_all_faults_;
  if ( reset_loc )
  {
    failAssign(false, VB_FA_LT);
  }
  if ( ap.disab_vb_fa_lt() || sp.mod_vb() )
  {
    faultAssign(false, VB_FLT);
    failAssign( false, VB_FA_LT);
  }
  else
  {
    faultAssign( (Sen->vb_hdwe()<=_vb_min && Sen->ib_hdwe()*ap.nP()>IB_MIN_UP) || (Sen->vb_hdwe()>=_vb_max), VB_FLT);
    failAssign( vb_fa_lt() || VbHardFail->calculate(vb_flt(), VB_HARD_SET, VB_HARD_RES, Sen->T(), reset_loc), VB_FA_LT);
  }
}
void Fault::vc_check(Sensors *Sen, BatteryMonitor *Mon, const float _vc_min, const float _vc_max, const bool reset)
{
  bool reset_loc = reset | reset_all_faults_;
  if ( reset_loc )
  {
    failAssign(false, VC_FA);
  }
  if ( sp.mod_ib() || ap.disab_ib_fa() )
  {
    faultAssign(false, VC_FLT);
    failAssign( false, VC_FA);
  }
  else
  {
    faultAssign( ( ((Sen->Vc()<=_vc_min) || (Sen->Vc()>=_vc_max)) && !reset_loc ), VC_FLT);
    failAssign( vc_fa() || VcHardFail->calculate(vc_flt(), VC_HARD_SET, VC_HARD_RES, Sen->T(), reset_loc), VC_FA);
  }
}

// Wrap scalars
void Fault::wrap_scalars(BatteryMonitor *Mon)
{
  if ( Mon->soc()>=WRAP_SOC_HI_OFF )
  {
    ewsat_slr_ = WRAP_SOC_HI_SLR;
    ewmin_slr_ = 1.;
  }
  else if ( Mon->soc() <= max(Mon->soc_min()+WRAP_SOC_LO_OFF_REL, WRAP_SOC_LO_OFF_ABS)  )
  {
    ewsat_slr_ = 1.;
    ewmin_slr_ = WRAP_SOC_LO_SLR;
  }
  else if ( Mon->voc_soc()>(Mon->vsat()-WRAP_HI_SETAT_MARG) ||
          ( Mon->voc_stat()>(Mon->vsat()-WRAP_HI_SETAT_MARG) && Mon->C_rate()>WRAP_MOD_C_RATE && Mon->soc()>WRAP_SOC_MOD_OFF) ) // Use voc_stat to get some anticipation
  {
    ewsat_slr_ = WRAP_HI_SETAT_SLR;
    ewmin_slr_ = 1.;
  }
  else
  {
    ewsat_slr_ = 1.;
    ewmin_slr_ = 1.;
  }
}


/*
                  ^ scale
                  |
------            |          ------- 1.0 ==> all lg
       -          |        -
         -        |      -
      |    -------------             0.0 ==> all sm
      |    |      |     |    |
   n_lo   n_hi    |   p_lo   p_hi
                  |
                  |
                  v
n_d = n_hi - n_lo
p_d = p_hi - p_lo
*/
// Scale select between a large and small set of inputs.  Small might be a precise, amplified sensor and large might be the high range equivalent
float scale_select(const float in, const ScaleBrk *brk, const float sm, const float lg)
{
  
  if ( brk->n_hi <= in && in <= brk->p_lo )
  {
    return ( sm );
  }

  else if ( in <= brk->n_lo || in >= brk->p_hi )
  {
    return ( lg );
  }

  else if ( in < brk->n_hi )
  {
    return ( (in - brk->n_lo) / brk->n_d * (sm - lg) + lg );
  }

  else
  {
    return ( (in - brk->p_lo) / brk->p_d * (lg - sm) + sm );
  }

}
float scale_select(const float in, const ScaleBrk *brk, const float sm, const float lg, int8_t *sel_stat)
{
  
  if ( brk->n_hi <= in && in <= brk->p_lo )
  {
    *sel_stat = 1;
    return ( sm );
  }

  else if ( in <= brk->n_lo || in >= brk->p_hi )
  {
    *sel_stat = -1;
    return ( lg );
  }

  else if ( in < brk->n_hi )
  {
    *sel_stat = 0;
    return ( (in - brk->n_lo) / brk->n_d * (sm - lg) + lg );
  }

  else
  {
    *sel_stat = 0;
    return ( (in - brk->p_lo) / brk->p_d * (lg - sm) + sm );
  }

}
