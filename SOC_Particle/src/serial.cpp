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

#include "serial.h"
#include "command.h"
#include "constants.h"
#include "debug.h"
#include "ble.h"

extern CommandPars cp;  // Various parameters shared at system level
extern BleCharacteristic txCharacteristic;

// Strip cmd string from front of source string
String chat_cmd_from(String *source)
{
  String out_str = "";

  #ifdef SOFT_DEBUG_QUEUE
    // debug_queue("chat_cmd_from enter");
  #endif

  while ( source->length() )
  {
    // get the new byte, add to input and check for completion
    char in_char = source->charAt(0);
    source->remove(0, 1);
    out_str += in_char;
    if ( is_finished(in_char) )
    {
      out_str = finish_request(out_str);  // remove whitespace and ;, etc
      break;
    }
  }

  #ifdef SOFT_DEBUG_QUEUE
    // debug_queue("chat_cmd_from exit");
  #endif

  return out_str;
}

// Non-blocking delay
void delay_no_block(const unsigned long long interval)
{
  unsigned long long previousMillis = System.millis();
  unsigned long long currentMillis = previousMillis;
  while( currentMillis - previousMillis < interval )
  {
    currentMillis = System.millis();
  }
}

// Cleanup string for final processing by chitchat
String finish_request(const String in_str)
{
  String out_str = in_str;
  // Remove whitespace
  out_str.trim();
  out_str.replace("\n","");
  out_str.replace("\0","");
  out_str.replace("","");
  out_str.replace(",","");
  out_str.replace(" ","");
  out_str.replace("=","");
  out_str.replace(";","");
  return out_str;
}

// Test for string completion character
boolean is_finished(const char in_char)
{
    return  in_char == '\n' ||
            in_char == '\0' ||
            in_char == ';'  ||
            in_char == ',';    
}

// Print consolidation
void print_all_header(Sensors *Sen)
{
  print_battery_header();
  print_battery_serial();
  print_rapid_header();
  print_temp_header();
  if ( sp.debug()==2  )
  {
    print_sim_header();
    print_signal_sel_header();
    print_shunt_header(Sen);
  }
  if ( sp.debug()==3  )
  {
    print_sim_header();
    print_ekf_header();
  }
  if ( sp.debug()==4  )
  {
    print_sim_header();
    print_signal_sel_header();
    print_ekf_header();
  }
}

// print ekf for data collection
void print_battery_header()
{
  Serial.printf("Battery_hdr, HDWE_IB_HI_LO, HDWE_IB_HI_LO_NOA_LO, HDWE_IB_HI_LO_AMP_LO, HDWE_IB_HI_LO_AMP_HI, HDWE_IB_HI_LO_NOA_HI, IB_ABS_MAX_NOA, IB_ABS_MAX_AMP, KF_Q_STD, KF_R_STD,");
  Serial.printf("SHUNT_AMP_GAIN, CURR_BIAS_AMP, SHUNT_NOA_GAIN, CURR_BIAS_NOA, NS, NP, CURR_SCALE_DISCH, HYS_SCALE, dc_dc_on,");
  Serial.printf("EWLO_TRM_SLR, EWHI_TRM_SLR, WRAP_HI_AMP, WRAP_LO_AMP, WRAP_HI_NOA, WRAP_LO_NOA, EWHI_SLR, EWLO_SLR,");
  Serial.printf("IBATT_DISAGREE_THRESH, IB_DIFF_SLR,");
  Serial.printf("\n");
}

void print_battery_serial()
 {
  #ifdef HDWE_IB_HI_LO
    boolean hdwe_ib_hi_lo = true;
  #else
    boolean hdwe_ib_hi_lo = false;
  #endif
  Serial.printf("Battery_val,%d,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    hdwe_ib_hi_lo, HDWE_IB_HI_LO_NOA_LO, HDWE_IB_HI_LO_AMP_LO, HDWE_IB_HI_LO_AMP_HI, HDWE_IB_HI_LO_NOA_HI, IB_ABS_MAX_NOA, IB_ABS_MAX_AMP, KF_Q_STD, KF_R_STD);

  Serial.printf("%10.7f,%10.7f,%10.7f,%10.7f,%4.2f,%4.2f,%10.7f,%10.7f,%d,",
    SHUNT_AMP_GAIN, sp.ib_bias_amp_z, SHUNT_NOA_GAIN, sp.ib_bias_noa_z, NS, NP, sp.ib_disch_slr_z, ap.hys_scale, ap.dc_dc_on);
  
  Serial.printf("%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    EWLO_TRM_SLR, EWHI_TRM_SLR, WRAP_HI_AMP, WRAP_LO_AMP, WRAP_HI_NOA, WRAP_LO_NOA, ap.ewhi_slr, ap.ewlo_slr);

  Serial.printf("%10.7f,%10.7f,",
    IBATT_DISAGREE_THRESH, ap.ib_diff_slr);

  Serial.printf("\n");
}

// Print primary data
void print_rapid_header(void)
{
  Serial.printf ("unit_rap, hm, cTime, dt,       ");
  Serial.printf("reset, reset_temp, soft_reset, soft_reset_sim, reset_all_faults, ekf_reset, kf_reset, init_mon, init_sim,   ");
  Serial.printf("chm, qcrs, qcap, sat, sel, mod, bmso,  ");
  Serial.printf("Tb_rap, Tb_f_rap, Tb_f_rate_rap,  ");
  Serial.printf("vb, ib, ib_dyn, dv_hys,   ");
  Serial.printf("ib_charge, voc_soc, ib_dyn_rstate, ib_dyn_lstate,    ");
  Serial.printf("vsat, dv_dyn, voc_stat, voc_ekf,     ");
  Serial.printf("y_ekf,    ");
  Serial.printf("soc_s, soc_ekf, soc, soc_min, d_delta_q, delta_q,");
  Serial.printf("\n");
}
void print_rapid_serial(const boolean reset, Publish *pubList, Sensors *Sen, BatteryMonitor *Mon)
{
  // if ( Sen->T == 0.) return;
  double cTime = double(Sen->now)/1000;
  
  sprintf(pr.buff,  "%s,%s,%13.4f,%8.4f,   %2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,   ", \
    pubList->unit.c_str(), pubList->hm_string.c_str(), cTime, Sen->T,
    reset, Sen->reset_temp(), cp.soft_reset_print, cp.soft_reset_sim_print, Sen->Flt->reset_all_faults_print(), cp.ekf_reset_print,
    cp.kf_reset_print, Mon->initializing(), Sen->Sim->initializing());
    Serial.printf("%s", pr.buff);

  sprintf(pr.buff,  "%d,%10.4f,%10.4f,%2d,%2d,%2d,%2d,   %11.8f,%11.8f,%11.8f,  ", \
    CHEM, Mon->q_cap_rated_scaled(), Mon->q_capacity(), pubList->sat, sp.ib_force(), sp.modeling(), Mon->bms_off(),
    Sen->Tb, Sen->Tb_f, Sen->Tb_f_rate);
    Serial.printf("%s", pr.buff);

  sprintf(pr.buff,  "%11.7f,%11.7f,%11.7f,%11.7f,   %11.7f,%11.7f,%11.7f,%11.7f,", \
    Mon->vb(), Mon->ib(), Mon->ib_dyn(), Mon->dv_hys(),
    Mon->ib_charge(), Mon->voc_soc(), Mon->ib_dyn_rstate(), Mon->ib_dyn_lstate());
    Serial.printf("%s", pr.buff);

  sprintf(pr.buff,  "%11.7f,%11.7f,%11.7f,%11.7f,  %11.7f,  %11.8f,%11.8f,%11.8f,%5.3f,%12.7f,%12.7f,", \
    Mon->vsat(), Mon->dv_dyn(), Mon->voc_stat(), Mon->hx(),
    Mon->y_ekf(),
    Sen->Sim->soc(), Mon->soc_ekf(), Mon->soc(), Mon->soc_min(), Mon->d_delta_q(), Mon->delta_q());
    Serial.printf("%s", pr.buff);

    Serial.printf("\n");

    Log.info("    print_rapid_create_string cTime,%9.3f,", cTime);
}
void print_rapid_data(const boolean reset, Sensors *Sen, BatteryMonitor *Mon, const boolean reset_temp)
{
  static uint8_t last_read_debug = 0;     // Remember first time with new debug to print headers
  if ( ( sp.debug()==1 || sp.debug()==2 || sp.debug()==3 || sp.debug()==4 ) )
  {
    if ( reset || (last_read_debug != sp.debug()) )
    {
      cp.num_v_print = 0UL;
      print_all_header(Sen);
    }
    if ( sp.tweak_test() )
    {
      // no print, done by sub-functions
      cp.num_v_print++;
    }
    if ( cp.publishS )
    {
      Log.info("  print_rapid_data:  print_rapid_serial");
      print_rapid_serial(reset, &pp.pubList, Sen, Mon);
      cp.num_v_print++;
    }
  }
  last_read_debug = sp.debug();
}

// print ekf for data collection
void print_ekf_header(void)
{
  Serial.printf("unit_e,c_time,dt,Fx_, Bu_, Q_, R_, P_, S_, K_, u_, x_, y_, z_,");
  Serial.printf("x_prior_, P_prior_, x_post_, P_post_, hx_, H_, frz_, tb_f_hx_, x_for_hx_,");
  Serial.printf("  voc_stat_T, voc_stat_tau, voc_stat_rstate, voc_stat_lstate,");
  Serial.printf("\n");
}
 void EKF_1x1::print_ekf_serial(BatteryMonitor *Mon)
 {
  // if ( dt_ekf_ == 0. ) return;
  double eTime = double(now_ekf_)/1000.;

  Serial.printf("unit_ekf,%13.4f,%8.4f,%13.10f,%13.10f,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%11.8g,%10.7g,%10.7g,",
    eTime, dt_ekf_, Fx_, Bu_, Q_, R_, P_, S_, K_, u_, x_, y_, z_);

  Serial.printf("%11.8g,%10.7g,%11.8g,%10.7g,%10.7g,%10.7g,%d,%11.8f,%11.8f,",
    x_prior_, P_prior_, x_post_, P_post_, hx_, H_, freeze_, Tb_f_for_hx_, x_for_hx_);

  Serial.printf("%9.6f,%9.6f,%9.6f,%9.6f,",
    Mon->vocStatFilt_T(), Mon->vocStatFilt_tau(), 
    Mon->vocStatFilt_rstate(), Mon->vocStatFilt_lstate());

  Serial.printf("\n");
}

// Print shunt logic data
void print_shunt_header(Sensors *Sen)
{
  Serial.printf("unit_shunt,c_time,reset,kfres,vovcm,vovcmkf,vovcn,vovcnkf,iscm,ibmkf,iscn,ibnkf,  ");

  Sen->ShuntAmp->print_serial_header('m');
  Sen->ShuntNoAmp->print_serial_header('n');

  Serial.printf("\n");
}
void print_shunt_serial(const boolean reset, Sensors *Sen)
{
  if ( ( sp.debug()==2  ) && cp.publishS )
  {
    double cTime = double(Sen->now)/1000.;

    sprintf(pr.buff, "shunt_unit,%13.4f, %d, %d,  %11.6f,%11.6f,%11.6f,%11.6f,%11.6f,%11.6f,%11.6f,%11.6f,  ",
      cTime, reset, cp.kf_reset_print,
      Sen->ib_amp_vo_vc(), Sen->ib_amp_vo_vc_f(), Sen->ib_noa_vo_vc(), Sen->ib_noa_vo_vc_f(),
      Sen->ShuntAmp->ishunt_cal(), Sen->ib_amp_hdwe_kf(), Sen->ShuntNoAmp->ishunt_cal(), Sen->ib_noa_hdwe_kf());
    Serial.printf("%s", pr.buff);

    Sen->ShuntAmp->print_serial();
    Sen->ShuntNoAmp->print_serial();
    Serial.printf("\n");
  }
}
void Shunt::print_serial_header(const char suffix)
{
  KF_->print_serial_header(suffix);
}

void Shunt::print_serial()
{
  KF_->print_serial();
}
// Data stream
void KalmanFilter::print_serial_header(const char s)
{
  Serial.printf("dt%c, xp0%c,xp1%c, x0%c,x1%c,  Fx00%c,Fx01%c,Fx10%c,Fx11%c,  Pp00%c,Pp01%c,Pp10%c,Pp11%c,  P00%c,P01%c,P10%c,P11%c,  Q00%c,Q01%c,Q10%c,Q11%c,  ",
    s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s);

  Serial.printf("G0%c,G1%c,  H0%c,H1%c,  K0%c,K1%c,  S%c, u%c, y%c,  ",
    s, s, s, s, s, s, s, s, s);

}
void KalmanFilter::print_serial()
{
    sprintf(pr.buff, "%6.4f,  %10.6f,%10.6f,%10.6f,%10.6f,  %4.1f,%6.4f,%4.1f,%4.1f,  %13.6e,%13.6e,%13.6e,%13.6e, %13.6e,%13.6e,%13.6e,%13.6e,  ",
        dt_,   x_prior_[0],x_prior_[1], x_[0],x_[1],   Fx_[0][0],Fx_[0][1],Fx_[1][0],Fx_[1][1],   P_prior_[0][0],P_prior_[0][1],P_prior_[1][0],P_prior_[1][1],   P_[0][0],P_[0][1],P_[1][0],P_[1][1]);
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%13.6e,%13.6e,%13.6e,%13.6e,  ",
        Q_[0][0],Q_[0][1],Q_[1][0],Q_[1][1]);
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%9.6f,%9.6f,  %4.1f,%4.1f,  %10.6f,%10.6f,  %10.6f,  %10.6f, %10.6f,  ",
        G_[0],G_[1],  H_[0],H_[1],  K_[0],K_[1],  S_, u_, y_);
    Serial.printf("%s", pr.buff);
}


// print_signal_select for data collection
// TODO:  delete the _T, _tau, _rstate, _lstate stuff
void print_signal_sel_header(void)
{
  Serial.printf("unit_s,c_time,reset,resaf,user_sel,   cc_dif,  ibmh,ibnh,ibmm,ibnm,ibm,  kfres,vovcm,vovcn,ibmkf,ibnkf,  ib_diff, ib_diff_f,");
  Serial.printf("  vr,voc_soc,e_w,e_w_f,ib_dm,dv_dm,e_wm,e_wm_r,e_wm_f,ib_dn,dv_dn,e_wn,e_wn_f,e_wm_t,");
  Serial.printf("  ib_sel_stat,vc_h,ib_h,ib_s,mib,ib, vb_sel,vb_h,vb_s,mvb,vb,  mtb,Tb_fa, ");
  Serial.printf("  ib_rate, ib_quiet, ib_really_quiet, tb_sel, ccd_thr, ewmhi_thr, ewmlo_thr, ewnhi_thr, ewnlo_thr, ibd_thr, ibq_thr, preserving,ff,y_ekf_f,ib_dec,");
  Serial.printf("  ib_dyn_T_m, ib_dyn_tau_m, ib_dyn_rstate_m, ib_dyn_lstate_m,");
  Serial.printf("  ib_dyn_T_n, ib_dyn_tau_n, ib_dyn_rstate_n, ib_dyn_lstate_n,");
  Serial.printf("  ib_wrp_T_n, ib_wrp_tau_n, ib_wrp_rate_n, ib_wrp_state_n, disable_amp_fault,");
  Serial.printf("  ib_wrp_reset_m, ib_wrp_T_m, ib_wrp_tau_m, ib_wrp_rate_m, ib_wrp_state_m,ib_amp,");
  Serial.printf("  ib_amp_lo, ib_amp_hi, ib_noa_lo, ib_noa_hi, ib_noa_kf, kfres, x1n, ib_wrp_tr_m, ib_wrp_tr_n,");
  Serial.printf("  vb_m, voc_m, voc_soc_m, wrap_m_and_n_fa, ib_is_functional,v_low,");
  Serial.printf("  vb_h_f,");
  Serial.printf("  fltw, falw, dispw,");
  Serial.printf("\n");
}
void print_signal_sel_serial(const boolean reset, Sensors *Sen, BatteryMonitor *Mon, BatterySim *Sim)
{
  if ( (sp.debug()==2 || sp.debug()==4 || sp.debug()==61 )  && cp.publishS )
  {
      double cTime = double(Sen->now)/1000.;

      sprintf(pr.buff, "unit_sel,%13.4f, %d, %d, %d, %10.7f, %8.6f,%8.6f,%8.6f,%8.6f,%8.6f,   %d,%8.6f,%8.6f,%8.6f,%8.6f,   %8.6f,%8.6f, ",
          cTime, reset, Sen->Flt->reset_all_faults_print(), sp.ib_force(),
          Sen->Flt->cc_diff(),
          Sen->ib_amp_hdwe(), Sen->ib_noa_hdwe(), Sen->ib_amp_model(), Sen->ib_noa_model(), Sen->ib_model(), 
          cp.kf_reset_print, Sen->ib_amp_vo_vc(), Sen->ib_noa_vo_vc(), Sen->ib_amp_hdwe_kf(), Sen->ib_noa_hdwe_kf(),
          Sen->Flt->ib_diff(), Sen->Flt->ib_diff_f());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "   %8.6f,%7.6f,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,%2d,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,",
          Sen->vc_hdwe(), Mon->voc_soc(), Sen->Flt->e_wrap(), Sen->Flt->e_wrap_filt(), Sen->Flt->ib_dyn_m(), Sen->Flt->dv_dyn_m(), Sen->Flt->e_wrap_m(), Sen->Flt->e_wrap_m_r(), Sen->Flt->e_wrap_m_filt(),
          Sen->Flt->ib_dyn_n(), Sen->Flt->dv_dyn_n(), Sen->Flt->e_wrap_n(), Sen->Flt->e_wrap_n_filt(),
          Sen->Flt->LoopIbAmp->e_wrap_trim());
      Serial.printf("%s", pr.buff);

        sprintf(pr.buff, "  %d,%8.6f,%8.6f,%8.6f, %d,%8.6f,  %d,%8.6f,%8.6f, %d,%8.6f,  %d, %d, ",
            Sen->Flt->ib_sel_stat(), Sen->vc_hdwe(), Sen->ib_hdwe(), Sim->ib_s(), sp.mod_ib(), Sen->ib(),
            Sen->Flt->vb_sel_stat(), Sen->vb_hdwe(), Sen->vb_model(), sp.mod_vb(), Sen->vb(),
            sp.mod_tb(), Sen->Flt->tb_fa());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%7.3f, %7.3f, %d, %d, %9.6f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%d,%d,%7.3f,%d,",
          Sen->Flt->ib_rate(), Sen->Flt->ib_quiet(),  Sen->Flt->ib_really_quiet(), Sen->Flt->tb_sel_status(),
          Sen->Flt->cc_diff_thr(), Sen->Flt->LoopIbAmp->ewhi_thr(),Sen->Flt->LoopIbAmp->ewlo_thr(), Sen->Flt->LoopIbNoa->ewhi_thr(),
          Sen->Flt->LoopIbNoa->ewlo_thr(), Sen->Flt->ib_diff_thr(), Sen->Flt->ib_quiet_thr(), Sen->Flt->preserving(),
          ap.fake_faults, Mon->y_ekf_filt(), Sen->Flt->ib_decision());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,",
          Sen->Flt->LoopIbAmp->ib_dyn_T(), Sen->Flt->LoopIbAmp->ib_dyn_tau(),
          Sen->Flt->LoopIbAmp->ib_dyn_rstate(), Sen->Flt->LoopIbAmp->ib_dyn_lstate());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,",
          Sen->Flt->LoopIbNoa->ib_dyn_T(), Sen->Flt->LoopIbNoa->ib_dyn_tau(),
          Sen->Flt->LoopIbNoa->ib_dyn_rstate(), Sen->Flt->LoopIbNoa->ib_dyn_lstate());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,%d,",
          Sen->Flt->LoopIbNoa->ib_wrp_T(), Sen->Flt->LoopIbNoa->ib_wrp_tau(),
          Sen->Flt->LoopIbNoa->ib_wrp_rate(), Sen->Flt->LoopIbNoa->ib_wrp_state(),
          Sen->Flt->disable_amp_fault());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%d,%9.6f,%9.6f,%9.6f,%9.6f,%9.6f,",
          Sen->Flt->LoopIbAmp->reset(), Sen->Flt->LoopIbAmp->ib_wrp_T(), Sen->Flt->LoopIbAmp->ib_wrp_tau(),
          Sen->Flt->LoopIbAmp->ib_wrp_rate(), Sen->Flt->LoopIbAmp->ib_wrp_state(), Sen->ib_amp());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%d,%d,%d,%d,%9.6f,%d,%9.6f,%9.6f,%9.6f,",
          Sen->Flt->ib_amp_lo(), Sen->Flt->ib_amp_hi(), Sen->Flt->ib_noa_lo(), Sen->Flt->ib_noa_hi(),
          Sen->ShuntNoAmp->ishunt_cal_kf(), cp.kf_reset_print,
          Sen->ShuntNoAmp->get_v(),  Sen->Flt->LoopIbAmp->e_wrap_trimmed(),  Sen->Flt->LoopIbNoa->e_wrap_trimmed());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%d,%d,%d,",
        Sen->Flt->LoopIbAmp->vb(), Sen->Flt->LoopIbAmp->voc(), Sen->Flt->LoopIbAmp->voc_soc(),
        Sen->Flt->wrap_m_and_n_fa(), Sen->Flt->ib_is_functional(),
        Mon->voltage_low());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%8.6f,", Sen->vb_hdwe_f());
      Serial.printf("%s", pr.buff);

      sprintf(pr.buff, "%ld, %ld, %ld,", Sen->Flt->fltw(), Sen->Flt->falw(), cp.disp_word);
      Serial.printf("%s", pr.buff);

      Serial.printf("\n");
    }
}

// print sim for data collection
void print_sim_header(void)
{
  Serial.printf("unit_m,  c_time,      dt_s, chm_s, qcrs_s, bmso_s, Tb_f_s, vsat_s, voc_stat_s, ");
  Serial.printf("dv_dyn_s, vb_s, ib_s, ib_dyn_s, dv_hys_s, ib_in_s, ib_charge_s, ioc_s, ");
  Serial.printf("sat_s, dq_s, q_cap_s, soc_s, reset_s, ddq_s, ");
  Serial.printf("ib_dyn_s_T, ib_dyn_s_tau, ib_dyn_s_rstate, ib_dyn_s_lstate, ");
  Serial.printf("bmso_s, vlow_s,");
  Serial.printf("\n");
}
void print_sim_serial(const boolean initializing_all, const boolean reset_temp, Sensors *Sen, BatterySim *Sim)
{
    if ( (sp.debug()==2 || sp.debug()==3 || sp.debug()==4 )  && cp.publishS && !initializing_all)
    {
        // if ( Sim->dt() == 0. ) return;
        double cTime = double(Sen->now)/1000.;

        sprintf(pr.buff, "unit_sim, %13.4f, %8.4f, %d, %10.4f, %d, %11.8f, %7.6f,%7.6f, ",
            cTime, Sim->dt(), CHEM, Sim->q_cap_rated_scaled(), Sim->bms_off(), Sim->tb_f(), Sim->vsat(), Sim->voc_stat());
        Serial.printf("%s", pr.buff);

        sprintf(pr.buff, "%7.6f,%8.6f, %7.6f,%7.6f,%7.6f,%7.6f,%7.6f,%7.6f, ",
            Sim->dv_dyn(), Sim->vb(), Sim->ib_s(), Sim->ib_dyn(), Sim->dv_hys(), Sim->ib_in(), Sim->ib_charge(), Sim->ioc());
        Serial.printf("%s", pr.buff);

        sprintf(pr.buff, " %d,  %9.4f, %10.4f,  %11.8f, %d, %7.6f,",
            Sim->saturated(), Sim->delta_q(), Sim->q_capacity(), Sim->soc(), reset_temp, Sim->d_delta_q_s());
        Serial.printf("%s", pr.buff);

        sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,",
            Sim->chargeTransfer_T(), Sim->chargeTransfer_tau(),
            Sim->chargeTransfer_rstate(), Sim->chargeTransfer_lstate());
        Serial.printf("%s", pr.buff);

        sprintf(pr.buff, "%d, %d,",
            Sim->bms_off(), Sim->voltage_low());
        Serial.printf("%s", pr.buff);

        Serial.printf("\n");
    }
  }

// print temperatures for data collection
void print_temp_header(void)
{
 Serial.printf("unit_t, c_time, T_t, Tb_hdw, Tb_mod, Tb, reset_temp,  Tb_hdwe_filt, Tb_model_filt,Tb_f,  Tb_hdwe_filt_rate, Tb_model_filt_rate, Tb_f_rate,\n");
}
void print_temp_serial(const boolean reset, Sensors *Sen)
{
  if ( sp.debug()==1  || sp.debug()==2  || sp.debug()==3 || sp.debug()==4  || sp.debug()==16 )
  {
    // if ( Sen->T_temp == 0. ) return;
    double cTime = double(Sen->now_temp)/1000.;
    Serial.printf("temp_unit, %13.4f, %8.4f, %11.8f, %11.8f, %11.8f, %d, %11.8f, %11.8f, %11.8f, %11.8f, %11.8f,  %11.8f,\n",
      cTime, Sen->T_temp, Sen->Tb_hdwe, Sen->Tb_model, Sen->Tb, reset, Sen->Tb_hdwe_filt, Sen->Tb_model_filt, Sen->Tb_f, Sen->Tb_hdwe_filt_rate,
      Sen->Tb_model_filt_rate, Sen->Tb_f_rate);
    Log.info("    print_temp_serial cTime,%9.3f,", cTime);
  }
}

// General purpose transmitter
void sendTxBuf(const String& txBuf, const boolean sendSerial, const boolean sendBLE)
{
    // USB serial
    if ( sendSerial ) Serial.print(txBuf);

    // BLE notify (chunked)
    if ( sendBLE ) bleSendChunked(txCharacteristic, reinterpret_cast<const uint8_t*>(txBuf.c_str()), txBuf.length());
}
void sendTxBuf(const char* txBuf, const boolean sendSerial, const boolean sendBLE)
{
  // Calculate the length of the char array
  size_t bufLength = strlen(txBuf);

  // USB serial
  if ( sendSerial ) {
    Serial.print(txBuf);
  }

  // BLE notify (chunked)
  if ( sendBLE ) {
    // The char* is already compatible with const uint8_t* for this use case
    bleSendChunked(txCharacteristic, reinterpret_cast<const uint8_t*>(txBuf), bufLength);
  }
}
/*
  Special handler for UART usb that uses built-in callback. SerialEvent occurs whenever a new data comes in the
  hardware serial RX.  This routine is run between each time loop() runs, so using delay inside loop can delay
  response.  Multiple bytes of data may be available.

  Particle documentation says not to use something like the cp.inp_token in the while loop statement.
  They suggest handling all the data in one call.   But this works, so far.

  serialEvent handles Serial
 */
void serialEvent()
{
  static String serial_str = "";
  static boolean serial_ready = false;

  // Each pass try to complete input from avaiable
  while ( !serial_ready && Serial.available() )
  {
    char in_char = (char)Serial.read();  // get the new byte

    // Intake
    // if the incoming character to finish, add a ';' and set flags so the main loop can do something about it:
    if ( is_finished(in_char) )
    {
        serial_str += ';';
        serial_ready = true;
        break;
    }

    else if ( in_char == '\r' )
        Serial.printf("\n");  // scroll user terminal

    else if ( in_char == '\b' && serial_str.length() )
    {
        Serial.printf("\b \b");  // scroll user terminal
        serial_str.remove(serial_str.length() -1 );  // backspace
    }

    else
        serial_str += in_char;  // process new valid character
  }

  // Pass info to inp_str
  if ( serial_ready )
  {
      if ( !cp.inp_token )
      {
          cp.inp_token = true;
          add_verify(&cp.inp_str, serial_str);
          serial_ready = false;
          cp.inp_token = false;
          serial_str = "";
      }
  }

}


// Wait on user input to reset EERAM values
void wait_on_user_input()
{
  uint8_t count = 0;
  uint16_t answer = '\r';
  // Get user input but timeout at 120 seconds if no response
  while ( count<30 && answer!='Y' && answer!='y' && answer!='n' && answer!='N' )
  {
    if ( answer=='\r')
    {
      count++;
      if ( count>1 ) delay(4000);
    }
    else delay(100);

    if ( Serial.available() )
      answer=Serial.read();

    else if ( cp.ble_first_char!='\0' )
    {
      answer = cp.ble_first_char;
      cp.ble_first_char = '\0';
    }

    else
      Serial.printf("unavail\n");

    if ( answer=='\r')
    {
      sendTxBuf("\n\n", true, true);
      sp.pretty_print( false );
      sendTxBuf("Reset to defaults? [y/n]:", true, true);
    }
    else  // User is typing.  Ignore him until they answer 'Y', 'N', or 'n'.  But timeout at 30 seconds if they don't
    {
      while ( answer!='Y' && answer!='y' && answer!='N' && answer!='n' && count<30 )
      {
        if ( Serial.available() )
          answer = Serial.read();

        else if ( cp.ble_first_char!='\0' )
        {
          answer = cp.ble_first_char;
          cp.ble_first_char = '\0';
        }

        else
          {
            Serial.printf("?");
            count++;
            delay(1000);
          }
      }
    }

  }

  // Wrap it up
  if ( answer=='Y' || answer=='y' )
  {
    sendTxBuf("  Y reset\n\n", true, true);
    sp.set_nominal();
    sp.pretty_print( true );
    System.backupRamSync();
  }
  else if ( answer=='n' || answer=='N' || count==30 )
  {
    sendTxBuf(" N.  moving on...\n\n", true, true);
  }

}
