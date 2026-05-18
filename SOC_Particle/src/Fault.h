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

#include "application.h"
#include "myLibrary/myFilters.h"
// #include "Battery.h"
#include "constants.h"
// #include "Cloud.h"
// #include "talk/chitchat.h"
#include "command.h"
// #include "Sync.h"
#include "parameters.h"
// #include "myLibrary/KF_1x1.h"

// Bit manipulation macros (if not provided by Particle/Arduino)
#ifndef bitSet
  #define bitSet(value, bit) ((value) |= (1UL << (bit)))
#endif
#ifndef bitClear
  #define bitClear(value, bit) ((value) &= ~(1UL << (bit)))
#endif
#ifndef bitRead
  #define bitRead(value, bit) (((value) >> (bit)) & 1)
#endif
#ifndef bitWrite
  #define bitWrite(value, bit, bitvalue) (bitvalue ? bitSet(value, bit) : bitClear(value, bit))
#endif

// AD
// extern PublishPars pp;  // For publishing
// extern CommandPars cp;  // Various parameters to be static at system level
extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level

#ifdef HDWE_IB_HI_LO
  #define IB_SEL_STAT_DEF 0
  #define TB_SEL_STAT_DEF 1
  #define VB_SEL_STAT_DEF 1
#else
  #define IB_SEL_STAT_DEF 1
  #define TB_SEL_STAT_DEF 1
  #define VB_SEL_STAT_DEF 1
#endif

enum ibSel {UsingNoa=-1, KeepTrying=0, UsingAmp=1, UsingNone=2};
enum dispw {conn=0, diff_ib=1, red_loss=2, fail_ib=3, fail_ibm=4, fail_vb=5, flt_tb=6, flt_ekf=7, SAT=8, off=9, accy=10, time_long=11, Count};

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
*/
struct ScaleBrk
{
  float n_lo;
  float n_d;
  float n_hi;
  float p_lo;
  float p_d;
  float p_hi;
  ScaleBrk(const float n_l, const float n_h, const float p_l, const float p_h) : n_lo(n_l), n_hi(n_h), p_lo(p_l), p_hi(p_h)
  {
    n_d = n_hi - n_lo;
    p_d = p_hi - p_lo;
  }
  String pretty_print()
  {
    String txBuf = String::format("ScaleBrk  [%7.3f %7.3f  %7.3f %7.3f]\n", n_lo, n_hi, p_lo, p_hi);
    return ( txBuf );
  }
};

// Fault word bits.   All faults heal
#define TB_FLT        0   // Momentary isolation of Tb failure, T=faulted
#define VB_FLT        1   // Momentary isolation of Vb failure, T=faulted
#define IB_AMP_FLT    2   // Momentary isolation of Ib amp failure, T=faulted 
#define IB_NOA_FLT    3   // Momentary isolation of Ib no amp failure, T=faulted 
//                  4
#define WRAP_HI_FLT   5   // Wrap isolates to Ib high fault
#define WRAP_LO_FLT   6   // Wrap isolates to Ib low fault
#define RED_LOSS      7   // Loss of current knowledge redundancy, T = fault
#define IB_DIFF_HI_FLT 8   // Faulted sensor difference error, T = fault
#define IB_DIFF_LO_FLT 9   // Faulted sensor difference error, T = fault
#define IB_DSCN_FLT   10  // Dual faulted quiet error, T = disconnected shunt
#define IB_AMP_BARE   11  // Unconnected ib bus, T = bare bus
#define IB_NOA_BARE   12  // Unconnected ib bus, T = bare bus
#define VC_FLT        13  // Momentary isolation of Vc failure, T=faulted
#define WRAP_HI_M_FLT 14  // Wrap reports Vb lo / Ib amp high fault
#define WRAP_LO_M_FLT 15  // Wrap reports Vb hi / Ib amp low fault
#define WRAP_HI_N_FLT 16  // Wrap reports Vb lo / Ib noa high fault
#define WRAP_LO_N_FLT 17  // Wrap reports Vb hi / Ib noa low fault
#define NUM_FLT       20  // Number of these

// Fail word bits.   A couple don't latch because single sensor fail in dual sensor system
#define TB_FA         0   // Peristed, latched isolation of Tb failure, heals because soft type, T=failed
#define VB_FA_LT      1   // Peristed, latched isolation of Vb failure, latches because hard type, T=failed
#define IB_AMP_FA     2   // Amp sensor selection memory, latches because hard type, T = amp failed
#define IB_NOA_FA     3   // Noamp sensor selection memory, latches because hard type, T = no amp failed
#define CC_DIFF_FA    4   // Accumulated Coulomb Counter difference used to isolate IB differences, heals functional type, T = faulted=failed 
#define WRAP_HI_FA    5   // Wrap isolates to Ib high fail, heals because dual sensor (no latch)
#define WRAP_LO_FA    6   // Wrap isolates to Ib low fail, heals because dual sensor (no latch)
#define WRAP_VB_FA    7   // Wrap isolates to Vb fail, latches because single sensor (latch)
#define IB_DIFF_HI_FA 8   // Persisted sensor difference error, latches because hard type, T = fail
#define IB_DIFF_LO_FA 9   // Persisted sensor difference error, latches because hard type, T = fail
#define IB_DSCN_FA    10  // Dual persisted quiet error, heals functional type, T = disconnected shunt
#define VC_FA         13  // Peristed, latched isolation of Vc failure, latches because hard type, T=failed
#define WRAP_HI_M_FA  14  // Wrap isolates to Ib amp high fail, heals because dual sensor (no latch)
#define WRAP_LO_M_FA  15  // Wrap isolates to Ib amp low fail, heals because dual sensor (no latch)
#define WRAP_HI_N_FA  16  // Wrap isolates to Ib amp high fail, heals because dual sensor (no latch)
#define WRAP_LO_N_FA  17  // Wrap isolates to Ib amp low fail, heals because dual sensor (no latch)
#define NUM_FA        20  // Number of these

#define faultRead(bit) ( (fltw_ >> bit) & 1 )
#define failRead(bit) ( (falw_ >> bit) & 1 )
#define dispRead(bit) ( (cp.disp_word >> bit) & 1 )
#define faultAssign(bval, bit) if (bval) bitSet(fltw_, bit); else bitClear(fltw_, bit)
#define failAssign(bval, bit) if (bval) bitSet(falw_, bit); else bitClear(falw_, bit)
#define dispAssign(bval, bit) if (bval) bitSet(cp.disp_word, bit); else bitClear(cp.disp_word, bit)

String bitMapPrint(char *buf, const int16_t fw, const uint8_t num);


// Model-based fault detector
// The trim integrator path is under threat for underflows and needs double resolution
// and possibly infrequent calls
class Looparound
{
public:
  Looparound();
  Looparound(BatteryMonitor *Mon, Sensors *Sen, const float wrap_hi_volt, const float wrap_lo_volt, const double wrap_trim_gain,
    const float imin, const float imax, const float err_max);
  ~Looparound();
  void calculate(const bool reset, const bool disable_fault, const float ib, Sensors *Sen, const bool freeze);
  float dv_dyn() { return dv_dyn_; };
  float e_wrap() { return e_wrap_; };
  float e_wrap_filt() { return e_wrap_filt_; };
  float e_wrap_rate() { return e_wrap_rate_; };
  float e_wrap_trim() { return e_wrap_trim_; };
  float e_wrap_trimmed() { return e_wrap_trimmed_; };
  float ewhi_thr() { return ewhi_thr_; };
  float ewlo_thr() { return ewlo_thr_; };
  uint8_t hi_fail() { return hi_fail_; };
  uint8_t hi_fault() { return hi_fault_; };
  float ib_dyn() { return ib_dyn_; };
  float ib_dyn_a() { return ChargeTransfer_->a(); };
  float ib_dyn_b() { return ChargeTransfer_->b(); };
  float ib_dyn_c() { return ChargeTransfer_->c(); };
  float ib_dyn_lstate() { return ChargeTransfer_->lstate(); };
  float ib_dyn_r() { return ChargeTransfer_->reset(); };
  float ib_dyn_rstate() { return ChargeTransfer_->rstate(); };
  float ib_dyn_T() { return ChargeTransfer_->T(); };
  float ib_dyn_tau() { return ChargeTransfer_->tau(); };
  float ib_wrp_a() { return WrapErrFilt_->a(); };
  float ib_wrp_b() { return WrapErrFilt_->b(); };
  float ib_wrp_rate() { return WrapErrFilt_->rate(); };
  float ib_wrp_state() { return WrapErrFilt_->lstate(); };
  float ib_wrp_T() { return WrapErrFilt_->T(); };
  float ib_wrp_tau() { return WrapErrFilt_->tau(); };
  uint8_t lo_fail() { return lo_fail_; };
  uint8_t lo_fault() { return lo_fault_; };
  String pretty_print(Sensors *Sen);
  bool reset() { return reset_; };
  float vb() { return vb_; };
  float voc() { return voc_; };
  float voc_soc() { return voc_soc_; };
protected:
  Chemistry *chem_;         // Chemistry
  LagExp *ChargeTransfer_;  // ChargeTransfer model {ib, vb} --> {voc}, ioc=ib for Battery version
  float dv_dyn_;            // Effective Randles voltage drop, V
  float e_wrap_;            // Wrap error, V
  float e_wrap_filt_;       // Wrap error, V
  float e_wrap_rate_;       // Wrap error rate, V/s
  float e_wrap_trim_;       // Trimmer, V
  float e_wrap_trimmed_;    // Trimmer applied to e_wrap_, V
  float ewhi_thr_;   // Threshold e_wrap failed high kicked, V
  float ewhi_thr_base_;     // Threshold e_wrap failed high base, V
  float ewlo_thr_;   // Threshold e_wrap failed low kicked, V
  float ewlo_thr_base_;     // Threshold e_wrap failed low base, V
  uint8_t hi_fail_;         // Fail bit
  uint8_t hi_fault_;        // Fault bit
  float ib_;                // Sensed unit shunt current, A
  float ib_dyn_;            // Effective Randles unit shunt current, A
  float ib_past_;           // Sensed unit shunt current past value, A
  float imax_;              // Current range max, A
  float imin_;              // Current range min, A
  uint8_t lo_fail_;         // Fail bit
  uint8_t lo_fault_;        // Fault bit
  BatteryMonitor *Mon_;     // Monitor ptr
  bool reset_;           // If resetting or not
  Sensors *Sen_;            // Sensors ptr
  TustinIntegrator *Trim_;  // Trim integrator
  float vb_;                // Looparound version of vb, V
  float voc_;               // Looparound version of voc, V 
  float voc_soc_;           // Looparound version of voc_soc, V 
  LagExp *WrapErrFilt_;  // Noise filter for voltage wrap
  TFDelay *WrapHi_;         // Wrap test persistence
  TFDelay *WrapLo_;         // Wrap test persistence
  float wrap_hi_volt_;      // Wrap high voltage (hyst +), V
  float wrap_lo_volt_;      // Wrap low voltage (hyst -), V
  double wrap_trim_gain_;   // Trim gain, r/s
};


// Detect faults and manage selection
class Fault
{
public:
  Fault();
  Fault(const double T, uint8_t *sp_preserving, BatteryMonitor *Mon, Sensors *Sen);
  ~Fault();
  bool cc_diff_fa() { return failRead(CC_DIFF_FA); };
  float cc_diff() { return cc_diff_; };
  void cc_diff(const bool reset, Sensors *Sen, BatteryMonitor *Mon);
  float cc_diff_thr() { return cc_diff_thr_; };
  float cc_diff_thr_;     // Threshold Coulomb Counters difference faults, soc fraction
  bool disable_amp_fault() { return disable_amp_fault_; };
  bool dscn_fa() { return failRead(IB_DSCN_FA); };
  bool dscn_flt() { return faultRead(IB_DSCN_FLT); };
  float dv_dyn_m() { return WrapLoopAmp->dv_dyn(); };
  float dv_dyn_n() { return WrapLoopNoa->dv_dyn(); };
  float e_wrap() { return e_wrap_; };
  float e_wrap_filt() { return e_wrap_filt_; };
  float e_wrap_m() { return WrapLoopAmp->e_wrap(); };
  float e_wrap_m_filt() { return WrapLoopAmp->e_wrap_filt(); };
  bool e_wrap_m_r() { return WrapLoopAmp->reset(); };
  float e_wrap_m_trim() { return WrapLoopAmp->e_wrap_trim(); };
  float e_wrap_n() { return WrapLoopNoa->e_wrap(); };
  float e_wrap_n_filt() { return WrapLoopNoa->e_wrap_filt(); };
  float e_wrap_rate() { return e_wrap_rate_; };
  float ewmin_slr() { return ewmin_slr_; };
  float ewsat_slr() { return ewsat_slr_; };
  uint32_t falw() { return falw_; };
  uint32_t fltw() { return fltw_; };
  bool ib_amp_bare() { return faultRead(IB_AMP_BARE);  };
  bool ib_amp_fa() { return failRead(IB_AMP_FA); };
  bool ib_amp_flt() { return faultRead(IB_AMP_FLT);  };
  bool ib_amp_hi() { return ib_amp_hi_; };
  bool ib_amp_invalid() { return ib_amp_invalid_; };
  bool ib_amp_lo() { return ib_amp_lo_; };
  ibSel ib_choice() { return ib_choice_; };
  ibSel ib_choice_past() { return ib_choice_last_; };
  uint16_t ib_decision() { return ib_decision_;  };
  void ib_decision_active_standby(Sensors *Sen);
  void ib_decision_hi_lo(Sensors *Sen);
  void ib_diff(const bool reset, Sensors *Sen, BatteryMonitor *Mon);
  float ib_diff() { return ( ib_diff_ ); };
  bool ib_diff_fa() { return ( failRead(IB_DIFF_HI_FA) || failRead(IB_DIFF_LO_FA) ); };
  float ib_diff_f() { return ( ib_diff_f_ ); };
  bool ib_diff_hi_fa() { return failRead(IB_DIFF_HI_FA); };
  bool ib_diff_hi_flt() { return faultRead(IB_DIFF_HI_FLT); };
  bool ib_diff_lo_fa() { return failRead(IB_DIFF_LO_FA); };
  bool ib_diff_lo_flt() { return faultRead(IB_DIFF_LO_FLT); };
  float ib_diff_thr() { return ib_diff_thr_; };
  float ib_diff_thr_;     // Threshold current difference faults, A
  bool ib_dscn_fa() { return failRead(IB_DSCN_FA); };
  bool ib_dscn_flt() { return faultRead(IB_DSCN_FLT); };
  float ib_dyn_m() { return WrapLoopAmp->ib_dyn(); };
  float ib_dyn_n() { return WrapLoopNoa->ib_dyn(); };
  bool ib_is_functional() { return ib_is_functional_; };
  bool ib_is_quiet() { return ib_is_quiet_; };
  bool ib_lo_active() { return ib_lo_active_; };
  void ib_logic(const bool reset, Sensors *Sen, BatteryMonitor *Mon);
  bool ib_noa_bare() { return faultRead(IB_NOA_BARE); };
  bool ib_noa_fa() { return failRead(IB_NOA_FA); };
  bool ib_noa_flt() { return faultRead(IB_NOA_FLT); };
  bool ib_noa_hi() { return ib_noa_hi_; };
  bool ib_noa_invalid() { return ib_noa_invalid_; };
  bool ib_noa_lo() { return ib_noa_lo_; };
  void ib_quiet(const bool reset, Sensors *Sen);
  float ib_quiet() { return ib_quiet_; };
  float ib_quiet_thr() { return ib_quiet_thr_; };
  float ib_quiet_thr_;     // Threshold below which ib is quiet, A pk
  float ib_rate() { return ib_rate_; };
  void ib_range(const bool reset, Sensors *Sen, BatteryMonitor *Mon);
  bool ib_really_quiet() { return ib_really_quiet_; };
  int8_t ib_sel_stat() { return ib_sel_stat_; };
  void ib_sel_stat(const int sel_stat) { ib_sel_stat_ = sel_stat; };
  void ib_wrap(const bool reset, Sensors *Sen, BatteryMonitor *Mon);
  TFDelay *IbLoLimitedHi;   // Persistence low amp limited high active status
  TFDelay *IbLoLimitedLo;   // Persistence low amp limited low active status
  int8_t latch_fake() { return latch_fake_; };
  void latch_fake(const bool cmd) { latch_fake_ = cmd; };
  int8_t latched_fail() { return latch_; };
  void latched_fail(const bool cmd) { latch_ = cmd; };
  Looparound *WrapLoopAmp;    // Looparound for Ib amp
  Looparound *WrapLoopNoa;    // Looparound for Ib noa
  bool no_fails() { return !latch_; };
  bool no_fails_fake() { return !latch_fake_; };
  bool preserving() { return *sp_preserving_; };
  void preserving(const bool cmd) {  sp.put_Preserving(cmd); };
  void pretty_print(Sensors *Sen, BatteryMonitor *Mon);
  bool record() { if ( ap.fake_faults() ) return no_fails_fake(); else return no_fails(); };
  bool red_loss() { return faultRead(RED_LOSS); };
  bool reset_all_faults() { return reset_all_faults_; };
  void reset_all_faults(const bool cmd) { reset_all_faults_ = cmd; };
  bool reset_all_faults_print() { return reset_all_faults_print_; };
  void reset_all_faults_select();
  void select_all_logic(Sensors *Sen, BatteryMonitor *Mon, const bool reset);
  void shunt_check(Sensors *Sen, BatteryMonitor *Mon, const bool reset);  // Range check Ib signals
  void shunt_select_initial(const bool reset);   // Choose between shunts for model
  void Tb_check(Sensors *Sen, const float _tb_min, const float _tb_max, const bool reset);  // Range check Tb
  bool Tb_fa() { return failRead(TB_FA); };
  bool Tb_flt() { return faultRead(TB_FLT); };
  int8_t tb_sel_stat_past() { return tb_sel_stat_last_; };
  int8_t tb_sel_status() { return tb_sel_stat_; };
  void vb_check(Sensors *Sen, BatteryMonitor *Mon, const float _vb_min, const float _vb_max, const bool reset);  // Range check Vb
  bool vb_clean() { return ( !vb_fail() ); };
  bool vb_fail() { return ( vb_fa_lt() || vb_sel_stat_==0 ); };
  bool vb_fa_lt() { return failRead(VB_FA_LT); };
  bool vb_flt() { return faultRead(VB_FLT); };
  int8_t vb_sel_stat() { return vb_sel_stat_; };
  int8_t vb_sel_stat_past() { return vb_sel_stat_last_; };
  void vc_check(Sensors *Sen, BatteryMonitor *Mon, const float _vc_min, const float _vc_max, const bool reset);  // Range check Vc
  bool vc_fa() { return failRead(VC_FA); };
  bool vc_flt() { return faultRead(VC_FLT); };
  void wrap_err_filt_state(const float in) { WrapErrFilt->lstate(in); }
  bool wrap_hi_and_lo_fa() { return ( failRead(WRAP_HI_FA) && failRead(WRAP_LO_FA) ); };
  bool wrap_hi_fa() { return failRead(WRAP_HI_FA); };
  bool wrap_hi_flt() { return faultRead(WRAP_HI_FLT); };
  bool wrap_hi_m_fa() { return failRead(WRAP_HI_M_FA); };
  bool wrap_hi_m_flt() { return faultRead(WRAP_HI_M_FLT); };
  bool wrap_hi_n_fa() { return failRead(WRAP_HI_N_FA); };
  bool wrap_hi_n_flt() { return faultRead(WRAP_HI_N_FLT); };
  bool wrap_hi_or_lo_fa() { return ( failRead(WRAP_HI_FA) || failRead(WRAP_LO_FA) ); };
  bool wrap_lo_fa() { return failRead(WRAP_LO_FA); };
  bool wrap_lo_flt() { return faultRead(WRAP_LO_FLT);  };
  bool wrap_lo_m_fa() { return failRead(WRAP_LO_M_FA); };
  bool wrap_lo_m_flt() { return faultRead(WRAP_LO_M_FLT); };
  bool wrap_lo_n_fa() { return failRead(WRAP_LO_N_FA); };
  bool wrap_lo_n_flt() { return faultRead(WRAP_LO_N_FLT); };
  bool wrap_m_and_n_fa() { return ( (failRead(WRAP_LO_M_FA) && failRead(WRAP_LO_N_FA)) ||
                                       (failRead(WRAP_HI_M_FA) && failRead(WRAP_HI_N_FA))  ); };
  bool wrap_m_fa() { return failRead(WRAP_LO_M_FA) || failRead(WRAP_HI_M_FA); };
  bool wrap_n_fa() { return failRead(WRAP_LO_N_FA) || failRead(WRAP_HI_N_FA); };
  void wrap_scalars(BatteryMonitor *Mon);
  bool wrap_vb_fa() { return failRead(WRAP_VB_FA); };
protected:
  TFDelay *CcdiffPer;       // Persistence cc_diff ekf fail amp
  TFDelay *IbAmpHardFail;   // Persistence ib hard fail amp
  TFDelay *IbdPosPer;       // Persistence ib diff hi instantaneous
  TFDelay *IbdNegPer;       // Persistence ib diff lo instantaneous
  TFDelay *IbdHiPer;        // Persistence ib diff hi
  TFDelay *IbdLoPer;        // Persistence ib diff lo
  LagExp *IbDiffFilt;       // Noise filter for signal selection
  TFDelay *IbNoAmpHardFail; // Persistence ib hard fail noa
  General2_Pole *QuietFilt; // Linear filter to test for quiet
  TFDelay *QuietPer;        // Persistence ib quiet disconnect detection
  TFDelay *QuietPerFunc;    // Persistence ib quiet normal functional detection
  RateLagExp *QuietRate;    // Linear filter to calculate rate for quiet
  TFDelay *TbHardFail;      // Persistence Tb hard fail
  TFDelay *VbHardFail;      // Persistence vb hard fail
  TFDelay *VcHardFail;      // Persistence vc hard fail
  LagExp *WrapErrFilt;      // Noise filter for voltage wrap
  TFDelay *WrapHi;          // Time high wrap fail persistence
  TFDelay *WrapLo;          // Time low wrap fail persistence
  float cc_diff_;           // EKF tracking error, C
  bool cc_diff_fa_;         // EKF tested disagree, T = error
  float cc_diff_empty_slr_; // Scale cc_diff when soc low, scalar
  bool disable_amp_fault_;  // Disable amp faults (both sensors agree), T=disable
  float ewmax_slr_;         // Scale wrap detection thresh when voc(soc) greater than max, scalar
  float ewmin_slr_;         // Scale wrap detection thresh when voc(soc) less than min, scalar
  float ewsat_slr_;         // Scale wrap detection thresh when voc(soc) saturated, scalar
  float e_wrap_;            // Wrap error, V
  float e_wrap_filt_;       // Wrap error, V
  float e_wrap_rate_;       // Wrap error rate, V/s
  uint32_t fltw_;           // Bitmapped faults
  uint32_t falw_;           // Bitmapped fails
  bool ib_amp_hi_;          // ib amp near it's range limit, T=near hi
  bool ib_amp_invalid_;     // Battery amp is invalid (hard failed)
  bool ib_amp_lo_;          // ib amp near it's range limit, T=near lo
  float ib_noa_rate_;       // ib amp rate, A/s
  ibSel ib_choice_;         // ib signal selection
  ibSel ib_choice_last_;    // ib signal selection
  uint16_t ib_decision_;    // ib_decision_hi_lo_, code (stops 0, stops on last decision)
  float ib_diff_;           // Current sensor difference error, A
  float ib_diff_f_;         // Filtered sensor difference error, A
  bool ib_is_functional_;   // Ib is active, T=functional
  bool ib_is_quiet_;        // Ib is found to be quiet, T=quiet
  bool ib_lo_active_;       // Battery low amp is in active range, T=active
  bool ib_lo_limited_hi_;   // Battery low amp is pegged at positive limit of hardware, T=limited
  bool ib_lo_limited_lo_;   // Battery low amp is pegged at negative limit of hardware, T=limited
  bool ib_noa_hi_;          // ib noa above amp high limit, T=above hi
  bool ib_noa_invalid_;     // Battery noa is invalid (hard failed)
  bool ib_noa_lo_;          // ib noa below amp low limit, T=below hi
  float ib_quiet_;          // ib hardware noise, A/s
  bool ib_really_quiet_;    // ib hardware noise and low abs level
  float ib_rate_;           // ib rate, A/s
  int8_t ib_sel_stat_;      // Memory of Ib signal selection, -1=noa, 0=none, 1=a
  int8_t ib_sel_stat_last_; // past value
  bool latch_;              // There is a latched fail, T=latched fail
  bool latch_fake_;         // There would be a latched fail if not faking, T=latched fail
  bool rate_amp_;           // ib_amp rate fault, T=fault
  bool rate_noa_;           // ib_noa rate fault, T=fault
  bool reset_all_faults_;   // Reset all fault logic, gets reset before serial call
  bool reset_all_faults_print_;  // Reset all fault logic
  uint8_t *sp_preserving_;  // Saving fault buffer.   Stopped recording.  T=preserve
  bool splrr_amp_;          // ib_amp soft fault preceeded by local rate or range, T=preserve
  bool splrr_noa_;          // ib_noa soft fault preceeded by local rate or range, T=preserve
  int8_t tb_sel_stat_;      // Memory of Tb signal selection, 0=none, 1=sensor
  int8_t tb_sel_stat_last_; // past value
  int8_t vb_sel_stat_;      // Memory of Vb signal selection, 0=none, 1=sensor
  int8_t vb_sel_stat_last_; // past value
  float wrap_hi_volt_;      // Wrap high amplified, V
  float wrap_hi_noa_;       // Wrap high non-amplified, V
  float wrap_lo_volt_;      // Wrap low amplified, V
  float wrap_lo_noa_;       // Wrap low non-amplified, V
};

// Misc

float scale_select(const float in, const ScaleBrk *brk, const float lo, const float hi);
float scale_select(const float in, const ScaleBrk *brk, const float lo, const float hi, int8_t *sel_stat);
