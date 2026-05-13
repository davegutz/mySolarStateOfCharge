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

#include "debug.h"
#include "parameters.h"
#include "talk/chitchat.h"

extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle


// Check for heap fragmentation during String += operation
// Use pointer to be sure not to miss the final assignment effect
void add_verify(String *src, const String addend)
{
  int src_len = src->length();
  *src += addend;
  if ( src->length() != (src_len + addend.length()) )
  {
    Serial.printf("\n\n\n\n**FRAG**\n\n\n\n");
  }
}


// sp.debug()==12 EKF
void debug_12(BatteryMonitor *Mon, Sensors *Sen)
{
  Serial.printf("ib,ib_mod,   vb,vb_mod,  voc,voc_stat_mod,voc_mod,   K, y,    SOC_mod, SOC_ekf, SOC,   %7.3f,%7.3f,   %7.3f,%7.3f,   %7.3f,%7.3f,%7.3f,    %7.3f,%7.3f,   %7.3f,%7.3f,%7.3f,\n",
  Mon->ib(), Sen->Sim->ib(),
  Mon->vb(), Sen->Sim->vb(),
  Mon->voc(), Sen->Sim->voc_stat(), Sen->Sim->voc(),
  Mon->K_ekf(), Mon->y_ekf(),
  Sen->Sim->soc(), Mon->soc_ekf(), Mon->soc());
}

// sp.debug()==-13 ib_dscn for Arduino.
// Start Arduino serial plotter.  Toggle v like 'vv0;vv-13;' to produce legend
void debug_check_m13(Sensors *Sen)
{

  // Arduinio header
  static int8_t last_call = 0;
  if ( sp.debug()!=last_call && sp.debug()==-13 )
    Serial.printf("ib_sel_st:, ib_amph:, ib_noah:, ib_rate:, ib_quiet:,  dscn_flt:, dscn_fa:\n");
  last_call = sp.debug();

  // Plot
  if ( sp.debug()!=-13)
    return;
  else
      Serial.printf("%d, %7.3f,%7.3f,  %7.3f,%7.3f,   %d,%d\n",
  Sen->Flt->ib_sel_stat(),
  max(min(Sen->Ib_amp_hdwe(), 2), -2), max(min(Sen->Ib_noa_hdwe(), 2), -2),
  max(min(Sen->Flt->ib_rate(),2), -2), max(min(Sen->Flt->ib_quiet(), 2), -2),
  Sen->Flt->ib_dscn_fa(), Sen->Flt->ib_dscn_fa());
}

// sp.debug()==-23 vb for Arduino.
// Start Arduino serial plotter.  Toggle v like 'vv0;vv-23;' to produce legend
void debug_check_m23(Sensors *Sen)
{
  // Arduinio header
  static int8_t last_call = 0;
  if ( sp.debug()!=last_call && sp.debug()==-23 )
    Serial.printf("Vb_hdwe-Vb_hdwe_f:\n");
  last_call = sp.debug();

  // Plot
  if ( sp.debug()!=-23)
    return;
  else
      Serial.printf("%7.3f\n", Sen->Vb_hdwe() - Sen->Vb_hdwe_f());
}

// Start Arduino serial plotter.  Toggle v like 'vv0;vv-23;' to produce legend
void debug_check_m24(Sensors *Sen)
{
  // Arduinio header
  static int8_t last_call = 0;
  if ( sp.debug()!=last_call && sp.debug()==-23 )
    Serial.printf("Vb_hdwe-Vb_hdwe_f:, Ib_hdwe:\n");
  last_call = sp.debug();

  // Plot
  if ( sp.debug()!=-24)
    return;
  else
      Serial.printf("%7.3f, %7.3f\n", Sen->Vb_hdwe() - Sen->Vb_hdwe_f(), Sen->Ib_hdwe());
}

// Q quick print critical parameters
void debug_q(BatteryMonitor *Mon, Sensors *Sen)
{
  String txBuf;

  debug_qf(Mon, Sen);

  txBuf = String::format("ib_amp_fail %d\nib_noa_fail %d\nvb_fail %d\nTb%7.3f\nvb%7.3f\nvoc%7.3f\nvoc_filt%7.3f\nvoc_stat%7.3f\nvoc_stat_f%7.3f\nvoc_soc%7.3f\nvsat%7.3f\nVc%7.3f\nib%7.3f\nsoc_m%8.4f\n\
soc_ekf%8.4f\nsoc%8.4f\nsoc_min%8.4f\nsoc_inf%8.4f\nmodeling %d\n",
    Sen->Flt->ib_amp_fa(), Sen->Flt->ib_noa_fa(), Sen->Flt->vb_fail(),
    Sen->Tb_f(), Mon->vb(), Mon->voc(), Mon->voc_dead(), Mon->voc_stat(), Mon->voc_stat_f(), Mon->voc_soc(), Mon->vsat(), Sen->Vc(), Mon->ib(), Sen->Sim->soc(), Mon->soc_ekf(),
    Mon->soc(), Mon->soc_min(), Mon->soc_inf(), sp.modeling());
  sendTxBuf(txBuf, true, IN_SERVICE);

  txBuf = String::format("dq_inf/dq_abs%10.1f/%10.1f %8.4f coul_eff*=%9.6f, DAB+=%9.6f\nDQn%10.1f Tn%10.1f DQp%10.1f Tp%10.1f\n",
    Mon->delta_q_inf(), Mon->delta_q_abs(), Mon->delta_q_inf()/Mon->delta_q_abs(),
    -Mon->delta_q_neg()/Mon->delta_q_pos(),
    -(Mon->delta_q_neg() + Mon->delta_q_pos()) / nice_zero(Mon->time_neg() + Mon->time_pos(), 1e-6),
    Mon->delta_q_neg(), Mon->time_neg(), Mon->delta_q_pos(), Mon->time_pos());
  sendTxBuf(txBuf, true, IN_SERVICE);

  // if ( Sen->Flt->falw() || Sen->Flt->fltw() ) chit("Pf;", SOON);
  // time_long_2_str((time_t)sp.Time_now(), pr.buff);
  // txBuf = String::format(" time %ld hms:  %s\n", sp.Time_now(), pr.buff);
  // sendTxBuf(txBuf, true, IN_SERVICE);
  if ( Sen->Flt->fltw()>0UL || Sen->Flt->falw()>0UL )
  {
    if ( Sen->Flt->falw()>0UL )
      sendTxBuf("THERE ARE FAILURES:  ", true, IN_SERVICE);
    else if ( Sen->Flt->fltw()>0UL )
      sendTxBuf("there are faults:  ", true, IN_SERVICE);
    txBuf = String::format("fltw %lu falw %lu\n", Sen->Flt->fltw(), Sen->Flt->falw());
    sendTxBuf(txBuf, true, IN_SERVICE);
  }
  else
  {
    txBuf = String::format("no faults:  fltw %lu falw %lu\n", Sen->Flt->fltw(), Sen->Flt->falw());
    sendTxBuf(txBuf, true, IN_SERVICE);
  }
}

// Quick print critical selection parameters
void debug_qf(BatteryMonitor *Mon, Sensors *Sen)
{
  String txBuf;

  txBuf = String::format("  mod_tb      %d  mod_vb      %d mod_ib          %d\n  tb_s_st     %d  vb_s_st     %d ib_s_st         %d  ib_choice %d  ib_decision_ %d\n  mod_tb_dscn %d  mod_vb_dscn %d mod_ib_amp_dscn %d mod_ib_noa_dscn %d\n",
      sp.mod_tb(), sp.mod_vb(), sp.mod_ib(),
      sp.mod_tb_dscn(), sp.mod_vb_dscn(), sp.mod_ib_amp_dscn(), sp.mod_ib_noa_dscn(),
      Sen->Flt->tb_sel_status(), Sen->Flt->vb_sel_stat(), Sen->Flt->ib_sel_stat(), Sen->Flt->ib_choice(), Sen->Flt->ib_decision());
  sendTxBuf(txBuf, true, IN_SERVICE);

  txBuf = String::format(" fake_faults %d\n latched_fail %d\n latch_fake %d\n preserving %d\n\n",
      ap.fake_faults(), Sen->Flt->latched_fail(), Sen->Flt->latch_fake(), Sen->Flt->preserving()) +
    String::format(" wrap_hi_or_lo_fa %d wrap_hi_and_lo_fa %d\n\n", Sen->Flt->wrap_hi_or_lo_fa(), Sen->Flt->wrap_hi_and_lo_fa());
  sendTxBuf(txBuf, true, IN_SERVICE);

}

// Quick print critical selection parameters
void debug_qs(BatteryMonitor *Mon, Sensors *Sen)
{
  String txBuf;

  txBuf = String::format("Selection:\n %d  Amp->bare\n %d  NoAmp->bare\n %d  ib diff fail\n %d  wrap hi fail\n %d  wrap lo fail\n %d  wrap volt fail\n %d  cc diff fail\n %d  sp.ib_force()\n %d  vb fail\n %d  Tb fail\n",
    Sen->ShuntAmp->bare_shunt(), Sen->ShuntNoAmp->bare_shunt(), Sen->Flt->ib_diff_fa(), Sen->Flt->wrap_hi_fa(), Sen->Flt->wrap_lo_fa(), Sen->Flt->wrap_vb_fa(), Sen->Flt->cc_diff_fa(), sp.ib_force(), Sen->Flt->vb_fa_lt(), Sen->Flt->Tb_fa()) +
    String::format(" %d  fake\n %d  ib choice\n %d  ib choice past\n %d  ib decision\n %d  vb sel stat\n %d  vb sel stat past\n %d  latch\n %d  latch_fake\n",
    ap.fake_faults(), Sen->Flt->ib_choice(), Sen->Flt->ib_choice_past(), Sen->Flt->ib_decision(), Sen->Flt->vb_sel_stat(), Sen->Flt->vb_sel_stat_past(), Sen->Flt->latched_fail(), Sen->Flt->latch_fake()) +
    String::format(" %d  preserving faults\n", Sen->Flt->preserving());
  sendTxBuf(txBuf, true, IN_SERVICE);

}

// Calibration modes: Bluetooth serial output
void debug_check_98(BatteryMonitor *Mon, Sensors *Sen)
{
  if ( sp.debug()!=98 )
    return;
  String txBuf;

  txBuf = String::format("imh imhkf inh inkfh: %6.2fA %6.2fA,  %6.2fA,%6.2fA\n",
    Sen->Ib_amp_hdwe_f(), Sen->Ib_amp_hdwe_kf(), Sen->Ib_noa_hdwe_f(), Sen->Ib_noa_hdwe_kf());

  sendTxBuf(txBuf, true, IN_SERVICE);
}
void debug_check_99(BatteryMonitor *Mon, Sensors *Sen)
{
  static int8_t last_call = 0;
  if ( sp.debug()!=last_call )
  {
    if ( sp.debug()==99 )
    {
      sendTxBuf(String::format("\nSetting hardware 'Xm0,' and throughput 'Dr1,'\n"), true, IN_SERVICE);
      chit("Xm0,", QUEUE);      // Hardware mode
      chit("Dr1,", QUEUE);      // Max rate to measure throughput in zero script
    }
  }
  last_call = sp.debug();

  if ( sp.debug()!=99 )
    return;
  else
  {
    String txBuf = 
      String::format("\n Tb   |") +
      String::format("Vb      Vbrms  SV,     *Dc|") + 
      String::format("Vr    Vrrms|") +
      String::format(" Imh    Imhkf Imhrms  SA,    *DA|") +
      String::format(" Inh    Inhkf Inhrms  SB,    *DB|") +
      String::format(" Ibsel *SDasy|") +
      String::format("voc   voc_soc *DwTab|") +
      String::format("*Sran|") +
      String::format(" T  |\n") +
      String::format("%6.2f|%6.3f %6.3f %5.2f %6.2f|%5.3f %5.3f|%6.2f %6.2f %5.3f %5.2f %6.2f|%6.2f %6.2f %5.3f %5.2f %6.2f|%6.2f  %5.2f|%5.2f %5.2f  %7.2f| %4.2f|%6.4f|\n",
      Sen->Tb_hdwe_f(),
      Sen->Vb_hdwe_f(), Sen->Vb_rms(), ap.Vb_scale(), sp.Vb_bias_hdwe(),
      Sen->ShuntAmp->Vc(), Sen->Vc_rms(),
      Sen->Ib_amp_hdwe_f(), Sen->Ib_amp_hdwe_kf(), Sen->Ib_amp_rms(), ap.ib_scale_amp(), sp.ib_bias_amp(),
      Sen->Ib_noa_hdwe_f(), Sen->Ib_noa_hdwe_kf(), Sen->Ib_noa_rms(), ap.ib_scale_noa(), sp.ib_bias_noa(),
      Sen->Ib_hdwe_f_cal(), sp.ib_disch_slr(),
      Mon->voc(), Mon->voc_soc(), sp.Dw(),
      ap.slr_res(),
      Sen->T());

    sendTxBuf(txBuf, true, IN_SERVICE);
  }
}

#ifdef DEBUG_INIT
  // Various parameters to debug initialization stuff as needed
  void debug_m1(BatteryMonitor *Mon, Sensors *Sen)
  {
    Serial.printf("mod %d fake_f %d reset_temp %d soft_sim_hold %d Tb%7.3f Tb_f%7.3f Vb%7.3f Ib%7.3f\nib_s%7.3f soc_s%8.4f delta_q_s%10.1f\nib%7.3f soc  %8.4f dq  %10.1f soc_ekf%8.4f dq_ekf%10.1f\nvoc_filt %7.3f vsat %7.3f sat %d sat_s %d dq_z%10.1f lf %d llf %d\n",
        sp.modeling(), ap.fake_faults(), Sen->reset_temp(), cp.soft_sim_hold, Sen->Tb(), Sen->Tb_f(), Sen->Vb(), Sen->Ib(),
        Sen->Sim->ib(), Sen->Sim->soc(), Sen->Sim->delta_q(),
        Mon->ib(), Mon->soc(), Mon->delta_q(), Mon->soc_ekf(), Mon->delta_q_ekf(),
        Mon->voc_dead(), Mon->vsat(), Mon->sat(), Sen->Sim->sat(), sp.delta_q(), Sen->Flt->latched_fail(), Sen->Flt->latch_fake());
  }
#endif

#ifdef SOFT_DEBUG_QUEUE
void debug_queue(const String who)
{
  if ( cp.inp_str.length() || cp.ctl_str.length() || cp.asap_str.length() || cp.soon_str.length() || cp.queue_str.length() || cp.last_str.length() )
    Serial.printf("%s:  chitchat %d freeze %d inp_token %d CONTROL[%s] ASAP[%s] SOON[%s] QUEUE[%s] LAST[%s] CMD[%s]\n",
      who.c_str(), cp.chitchat, cp.freeze, cp.inp_token, cp.ctl_str.c_str(), cp.asap_str.c_str(), cp.soon_str.c_str(), cp.queue_str.c_str(), cp.last_str.c_str(), cp.cmd_str.c_str());
}
#endif
