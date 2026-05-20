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
    char in_char = source->charAt(0);
    source->remove(0, 1);
    out_str += in_char;
    if ( is_finished(in_char) )
    {
      out_str = finish_request(out_str);
      break;
    }
  }

  #ifdef SOFT_DEBUG_QUEUE
    // debug_queue("chat_cmd_from exit");
  #endif

  return out_str;
}

// Non-blocking delay
void delay_no_block(const uint64_t delay_millis)
{
  uint64_t previousMillis = millis();
  uint64_t currentMillis = previousMillis;
  while( currentMillis - previousMillis < delay_millis )
  {
    currentMillis = millis();
  }
}

// Cleanup string for final processing by chitchat
String finish_request(const String in_str)
{
  String out_str = in_str;
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
bool is_finished(const char in_char)
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

// print battery parameter header
void print_battery_header()
{
  Serial.printf("Battery_hdr, hdwe_ib_hi_lo, AMP_WRAP_TRIM_GAIN, ap_cc_diff_slr, ap_dc_dc_on, ap_disab_ib_fa, ap_disab_tb_fa, ap_disab_vb_fa_lt,");
  Serial.printf("ap_ds_voc_soc, ap_dv_voc_soc, ap_eframe_mult, ap_ewhi_slr, ap_ewlo_slr, ap_hys_scale, ap_ib_diff_slr, ap_ib_quiet_slr,");
  Serial.printf("cp_ts, CHEM, DF2, EKF_CONV, EKF_NOM_DT, EKF_Q_SD_NORM, EKF_R_SD_NORM,");
  Serial.printf("EKF_T_CONV, EKF_T_RES, EWHI_TRM_SLR, EWLO_TRM_SLR, F_MAX_T_WRAP, HDB_VB, HDWE_IB_HI_LO_AMP_HI, HDWE_IB_HI_LO_AMP_LO,");
  Serial.printf("HDWE_IB_HI_LO_NOA_HI, HDWE_IB_HI_LO_NOA_LO, HYS_IB_THR, HYS_SOC_MIN_MARG, IB_ABS_MAX_AMP, IB_ABS_MAX_NOA, IB_LO_ACTIVE_SET,");
  Serial.printf("IB_LO_ACTIVE_RES, IB_MIN_UP, IBATT_DISAGREE_THRESH,");
  Serial.printf("IMAX_NUM, KF_Q_STD, KF_R_STD, MAX_TRIM_RATE, MAX_WRAP_ERR_FILT, MAX_Y_FILT, MIN_Y_FILT, MXEPS,");
  Serial.printf("NOA_WRAP_TRIM_GAIN, NOMINAL_TB, NOMINAL_VB, NOM_UNIT_CAP, NP, NS, RATED_TEMP, SHUNT_AMP_GAIN, SHUNT_NOA_GAIN,");
  Serial.printf("sp_cutback_gain_slr, sp_Dw, sp_ib_disch_slr, sp_s_cap_mon, sp_s_cap_sim, sp_vsat_add, TAU_Y_FILT, TB_FILT,");
  Serial.printf("TB_MAX, TB_MIN, TCHARGE_DISPLAY_DEADBAND, TMAX_FILT, T_RLIM, VB_DC_DC, VB_MAX, VB_MIN,");
  Serial.printf("VOC_STAT_FILT, WN_Y_FILT, WRAP_ERR_FILT, WRAP_HI_AMPV, WRAP_HI_NOAV, WRAP_HI_RES, WRAP_HI_SET, WRAP_HI_SETAT_MARG,");
  Serial.printf("WRAP_LO_AMPV, WRAP_LO_NOAV, WRAP_LO_RES, WRAP_LO_SET, WRAP_MOD_C_RATE, WRAP_SOC_HI_OFF, WRAP_SOC_HI_SLR,");
  Serial.printf("WRAP_SOC_LO_OFF_ABS, WRAP_SOC_LO_OFF_REL, WRAP_SOC_LO_SLR, WRAP_SOC_MOD_OFF, ZETA_Y_FILT,WRAP_HI_SETAT_SLR,");
  Serial.printf("\n");
}

void print_battery_serial()
{
  #ifdef HDWE_IB_HI_LO
    bool hdwe_ib_hi_lo = true;
  #else
    bool hdwe_ib_hi_lo = false;
  #endif
  sprintf(pr.buff, "Battery_val,%d,%10.7f,%10.7f,%d,%d,%d,%d,%10.7f,%10.7f,",
    hdwe_ib_hi_lo, AMP_WRAP_TRIM_GAIN, ap.cc_diff_slr(), ap.dc_dc_on(), ap.disab_ib_fa(), ap.disab_tb_fa(), ap.disab_vb_fa_lt(),
    ap.ds_voc_soc(), ap.dv_voc_soc());
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%d,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%d,%10.7f,",
    ap.eframe_mult(), ap.ewhi_slr(), ap.ewlo_slr(), ap.hys_scale(), ap.ib_diff_slr(), ap.ib_quiet_slr(), cp.ts, CHEM, DF2);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    EKF_CONV, EKF_NOM_DT, EKF_Q_SD_NORM, EKF_R_SD_NORM, EKF_T_CONV, EKF_T_RES);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    EWHI_TRM_SLR, EWLO_TRM_SLR, F_MAX_T_WRAP, HDB_VB, HDWE_IB_HI_LO_AMP_HI, HDWE_IB_HI_LO_AMP_LO, HDWE_IB_HI_LO_NOA_HI, HDWE_IB_HI_LO_NOA_LO);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    HYS_IB_THR, HYS_SOC_MIN_MARG, IB_ABS_MAX_AMP, IB_ABS_MAX_NOA, IB_LO_ACTIVE_SET, IB_LO_ACTIVE_RES, IB_MIN_UP, IBATT_DISAGREE_THRESH, IMAX_NUM, KF_Q_STD);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    KF_R_STD, MAX_TRIM_RATE, MAX_WRAP_ERR_FILT, MAX_Y_FILT, MIN_Y_FILT, MXEPS, NOA_WRAP_TRIM_GAIN, NOMINAL_TB, NOMINAL_VB);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%4.2f,%4.2f,%10.7f,%10.7f,%10.7f,%10.7f,",
    NOM_UNIT_CAP, NP, NS, RATED_TEMP, SHUNT_AMP_GAIN, SHUNT_NOA_GAIN, sp.cutback_gain_slr(), sp.Dw());
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    sp.ib_disch_slr(), ap.s_cap_mon(), ap.s_cap_sim(), sp.Vsat_add(), TAU_Y_FILT, TB_FILT, TB_MAX, TB_MIN);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    TCHARGE_DISPLAY_DEADBAND, TMAX_FILT, T_RLIM, VB_DC_DC, VB_MAX, VB_MIN, VOC_STAT_FILT, WN_Y_FILT);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    WRAP_ERR_FILT, WRAP_HI_AMPV, WRAP_HI_NOAV, WRAP_HI_RES, WRAP_HI_SET, WRAP_HI_SETAT_MARG, WRAP_LO_AMPV, WRAP_LO_NOAV);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,%10.7f,",
    WRAP_LO_RES, WRAP_LO_SET, WRAP_MOD_C_RATE, WRAP_SOC_HI_OFF, WRAP_SOC_HI_SLR, WRAP_SOC_LO_OFF_ABS, WRAP_SOC_LO_OFF_REL);
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff, "%10.7f,%10.7f,%10.7f,%10.7f, ",
    WRAP_SOC_LO_SLR, WRAP_SOC_MOD_OFF, ZETA_Y_FILT, WRAP_HI_SETAT_SLR);
  Serial.printf("%s", pr.buff);

  Serial.printf("\n");
}

// print ekf for data collection
void print_ekf_header(void)
{
  Serial.printf("unit_e,c_time_e,dt_ekf,Fx_, Bu_, Q_, R_, P_, S_, K_, u_, x_, y_, z_,");
  Serial.printf("x_prior_, P_prior_, x_post_, P_post_, hx_, H_, frz_, tb_f_hx_, x_for_hx_,");
  Serial.printf("  voc_stat_f_T, voc_stat_f_tau, voc_stat_f_rstate, voc_stat_f_lstate,");
  Serial.printf("y_ekf_f_T, y_ekf_f_tau, y_ekf_f_lstate,");
  Serial.printf("\n");
}

void EKF_1x1::print_ekf_serial(BatteryMonitor *Mon)
{
  static double last_eTime = 0.;
  double eTime = double(now_ekf_)/1000.;
  if ( eTime <= last_eTime + 0.00005 ) return;
  last_eTime = eTime;

  Serial.printf("unit_ekf,%13.4f,%8.4f,%13.10f,%13.10f,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%11.9g,%10.7g,%10.7g,",
    eTime, dt_ekf_, Fx_, Bu_, Q_, R_, P_, S_, K_, u_, x_, y_, z_);

  Serial.printf("%11.9g,%10.7g,%11.9g,%10.7g,%10.7g,%10.7g,%d,%11.8f,%11.9f,",
    x_prior_, P_prior_, x_post_, P_post_, hx_, H_, freeze_, Tb_f_for_hx_, x_for_hx_);

  Serial.printf("%9.6f,%9.6f,%9.6f,%9.6f,",
    Mon->vocStatFilt_T(), Mon->vocStatFilt_tau(),
    Mon->vocStatFilt_rstate(), Mon->vocStatFilt_lstate());

  Serial.printf("%9.6f,%9.6f,%9.6f,",
    Mon->y_ekf_f_T(), Mon->y_ekf_f_tau(), Mon->y_ekf_f_lstate());

  Serial.printf("\n");
}

void print_rapid_data(const bool reset, Sensors *Sen, BatteryMonitor *Mon, const bool reset_temp)
{
  static uint8_t last_read_debug = 0;
  static double last_cTime = 0.;
  if ( ( sp.debug()==1 || sp.debug()==2 || sp.debug()==3 || sp.debug()==4 ) )
  {
    if ( reset || (last_read_debug != sp.debug()) )
    {
      cp.num_v_print = 0UL;
      print_all_header(Sen);
    }
    if ( sp.tweak_test() )
    {
      cp.num_v_print++;
    }
    if ( cp.publishS && (reset || Mon->cTime() > last_cTime + 0.00005) )
    {
      print_rapid_serial(reset, &pp.pubList, Sen, Mon);
      cp.num_v_print++;
      last_cTime = Mon->cTime();
    }
  }
  last_read_debug = sp.debug();
}

// Print primary data
void print_rapid_header(void)
{
  Serial.printf ("unit_rap, cTime, dt, hm, reset, reset_temp, soft_reset, soft_reset_sim, reset_all_faults, ekf_reset, kf_reset, init_mon, init_sim,   ");
  Serial.printf("chm, qcrs, qcap, sat, saturated, sel, mod, bmso,  ");
  Serial.printf("Tb, Tb_f, ");
  Serial.printf("vb, ib, ib_dyn, dv_hys,   ");
  Serial.printf("ib_charge, voc_soc, ib_dyn_r, ib_dyn_T, ib_dyn_rstate, ib_dyn_lstate,    ");
  Serial.printf("vsat, dv_dyn, voc_stat, voc_ekf, y_ekf,    ");
  Serial.printf("soc_s, soc_ekf, soc, soc_min, d_delta_q, delta_q,");
  Serial.printf("\n");
}

void print_rapid_serial(const bool reset, Publish *pubList, Sensors *Sen, BatteryMonitor *Mon)
{
  sprintf(pr.buff,  "%s,%13.4f,%8.4f,%s,   %2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,   ", \
    pubList->unit.c_str(), Mon->cTime(), Mon->dt(), pubList->hm_string.c_str(), 
    reset, Sen->reset_temp(), cp.soft_reset_print, cp.soft_reset_sim_print, Sen->Flt->reset_all_faults_print(), cp.ekf_reset_print,
    cp.kf_reset_print, Mon->initializing(), Sen->Sim->initializing());
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff,  "%d,%15.9f,%15.9f,%2d,%2d,%2d,%2d,%2d, ", \
    CHEM, Mon->q_cap_rated_scaled(), Mon->q_capacity(), pubList->sat, pubList->saturated, sp.ib_force(), sp.modeling(), Mon->bms_off());
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff,  "%11.8f,%11.8f, ", \
    Sen->Tb(), Sen->Tb_f());
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff,  "%11.7f,%11.7f,%11.7f,%11.7f,   %11.7f,%11.7f,%11.7f,%11.7f,%11.7f,%11.7f,", \
    Mon->vb(), Mon->ib(), Mon->ib_dyn(), Mon->dv_hys(),
    Mon->ib_charge(), Mon->voc_soc(), Mon->ib_dyn_r(), Mon->ib_dyn_T(), Mon->ib_dyn_rstate(), Mon->ib_dyn_lstate());
  Serial.printf("%s", pr.buff);

  sprintf(pr.buff,  "%11.9f,%11.9f,%11.9f,%11.9f,%11.9f,  %11.9f,%11.9f,%11.9f,%9.7f,%15.9f,%15.9f,", \
    Mon->vsat(), Mon->dv_dyn(), Mon->voc_stat(), Mon->hx(), Mon->y_ekf(),
    Sen->Sim->soc(), Mon->soc_ekf(), Mon->soc(), Mon->soc_min(), Mon->d_delta_q(), Mon->delta_q());
  Serial.printf("%s", pr.buff);

  Serial.printf("\n");
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

void Shunt::print_serial_header(const char suffix)
{
  KF_->print_serial_header(suffix);
}

void Shunt::print_serial()
{
  KF_->print_serial();
}

// Print shunt logic data
void print_shunt_header(Sensors *Sen)
{
  Serial.printf("unit_shunt,c_time_shunt,reset,kfres,vovcm,vovcmkf,vovcn,vovcnkf,iscm,ib_amp_hdwe_kf,iscn,ib_noa_hdwe_kf,  ");
  Sen->ShuntAmp->print_serial_header('m');
  Sen->ShuntNoAmp->print_serial_header('n');
  Serial.printf("\n");
}

void print_shunt_serial(const bool reset, Sensors *Sen)
{
  static double last_cTime_sh = 0.;
  if ( ( sp.debug()==2  ) && cp.publishS )
  {
    double cTime = double(Sen->now())/1000.;
    if ( !reset && cTime <= last_cTime_sh + 0.00005 ) return;
    last_cTime_sh = cTime;

    sprintf(pr.buff, "shunt_unit,%13.4f, %d, %d,  %11.6f,%11.6f,%11.6f,%11.6f,%11.6f,%11.6f,%11.6f,%11.6f,  ",
      cTime, reset, cp.kf_reset_print,
      Sen->ib_amp_vo_vc(), Sen->ib_amp_vo_vc_kf(), Sen->ib_noa_vo_vc(), Sen->ib_noa_vo_vc_kf(),
      Sen->ShuntAmp->ishunt_cal(), Sen->ib_amp_hdwe_kf(), Sen->ShuntNoAmp->ishunt_cal(), Sen->ib_noa_hdwe_kf());
    Serial.printf("%s", pr.buff);

    Sen->ShuntAmp->print_serial();
    Sen->ShuntNoAmp->print_serial();
    Serial.printf("\n");
  }
}

// print_signal_select for data collection
void print_signal_sel_header(void)
{
  Serial.printf("unit_s, c_time_sel, dt_sel, reset, resaf, user_sel, cc_dif, ib_amp_hdwe, ib_noa_hdwe, ib_amp_model, ib_noa_model, ib_model, kfres, vovcm, vovcn, ib_amp_hdwe_kf, ib_noa_hdwe_kf, ib_diff, ib_diff_f, ");
  Serial.printf("  vc_sum, voc_soc, e_wrap, e_wrap_filt, ib_dyn_m, dv_dyn_m, e_wrap_m, e_wrap_m_reset, e_wrap_m_filt, e_wrap_m_trim, ib_dyn_n, dv_dyn_n, e_wrap_n, e_wrap_n_filt, e_wrap_n_trim, ");
  Serial.printf("  ib_sel_stat, ib_choice, vc_h,ib_h, ib_s, mib,ib, vb_sel, vb_hdwe, vb_s, mvb,vb,  mtb, ");
  Serial.printf("  ib_rate, ib_quiet, ib_really_quiet, tb_sel, ccd_thr, ewmhi_thr, ewmlo_thr, ewnhi_thr, ewnlo_thr, ibd_thr, ibq_thr, preserving,ff,y_ekf,y_ekf_f,ib_dec, ");
  Serial.printf("  ib_dyn_T_m, ib_dyn_tau_m, ib_dyn_rstate_m, ib_dyn_lstate_m, ib_lo_active, ");
  Serial.printf("  ib_dyn_T_n, ib_dyn_tau_n, ib_dyn_rstate_n, ib_dyn_lstate_n, ");
  Serial.printf("  ib_wrp_T_n, ib_wrp_tau_n, ib_wrp_rate_n, ib_wrp_state_n, disable_amp_fault, ");
  Serial.printf("  ib_wrp_reset_m, ib_wrp_reset_n, ib_wrp_T_m, ib_wrp_tau_m, ib_wrp_rate_m, ib_wrp_state_m, ib_amp, ib_noa, ");
  Serial.printf("  ib_amp_lo, ib_amp_hi, ib_noa_lo, ib_noa_hi, ib_noa_kf, kfres, kf_v_m, kf_v_n, e_wrap_m_trimmed, e_wrap_n_trimmed, ");
  Serial.printf("  vb_model, voc_m, voc_soc_m, voc_n, voc_soc_n, wrap_m_and_n_fa, ib_is_functional,voltage_low, ");
  Serial.printf("  Tb_f_rate, Tb_hdwe, Tb_hdwe_f, Tb_hdwe_f_rate, Tb_hdwe_f_rstate, Tb_hdwe_f_lstate, Tb_hdwe_f_dt, Tb_hdwe_f_tau, ");
  Serial.printf("  Tb_model, Tb_model_f, Tb_model_f_rate, Tb_model_f_rstate, Tb_model_f_lstate, Tb_model_f_dt, Tb_model_f_tau, ");
  Serial.printf("  fltw, falw, dispw,");
  Serial.printf("\n");
}

void print_signal_sel_serial(const bool reset, Sensors *Sen, BatteryMonitor *Mon, BatterySim *Sim)
{
  static double last_cTime_sel = 0.;
  if ( (sp.debug()==2 || sp.debug()==4 || sp.debug()==61 )  && cp.publishS )
  {
    if ( !reset && Sen->cTime() <= last_cTime_sel + 0.00005 ) return;
    last_cTime_sel = Sen->cTime();
    sprintf(pr.buff, "unit_sel,%13.4f, %8.6f, %d, %d, %d, %10.7f, %8.6f,%8.6f,%8.6f,%8.6f,%8.6f,   %d,%8.6f,%8.6f,%8.6f,%8.6f,   %8.6f,%8.6f, ",
        Sen->cTime(), Sen->T(), reset, Sen->Flt->reset_all_faults_print(), sp.ib_force(),
        Sen->Flt->cc_diff(),
        Sen->ib_amp_hdwe(), Sen->ib_noa_hdwe(), Sen->ib_amp_model(), Sen->ib_noa_model(), Sen->ib_model(),
        cp.kf_reset_print, Sen->ib_amp_vo_vc(), Sen->ib_noa_vo_vc(), Sen->ib_amp_hdwe_kf(), Sen->ib_noa_hdwe_kf(),
        Sen->Flt->ib_diff(), Sen->Flt->ib_diff_f());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "   %8.6f,%7.6f,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,%2d,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,",
        Sen->Vc_hdwe_sum(), Mon->voc_soc(), Sen->Flt->e_wrap(), Sen->Flt->e_wrap_filt(), Sen->Flt->ib_dyn_m(), Sen->Flt->dv_dyn_m(),
        Sen->Flt->e_wrap_m(), Sen->Flt->e_wrap_m_r(), Sen->Flt->e_wrap_m_filt(), Sen->Flt->WrapLoopAmp->e_wrap_trim(),
        Sen->Flt->ib_dyn_n(), Sen->Flt->dv_dyn_n(), Sen->Flt->e_wrap_n(), Sen->Flt->e_wrap_n_filt(),
        Sen->Flt->WrapLoopNoa->e_wrap_trim());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "  %d,%d,%8.6f,%8.6f,%8.6f, %d,%8.6f,  %d,%8.6f,%8.6f, %d,%8.6f,  %d, ",
        Sen->Flt->ib_sel_stat(), Sen->Flt->ib_choice(), Sen->Vc_hdwe(), Sen->ib_hdwe(), Sim->ib_s(), sp.mod_ib(), Sen->ib(),
        Sen->Flt->vb_sel_stat(), Sen->vb_hdwe(), Sen->Sim->vb(), sp.mod_vb(), Sen->vb(),
        sp.mod_tb());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%10.6f, %10.6f, %d, %d, %9.6f,%10.6f,%10.6f,%10.6f,%10.6f,%10.6f,%10.6f,%d,%d,%8.6f,%8.6f,%d,",
        Sen->Flt->ib_rate(), Sen->Flt->ib_quiet(), Sen->Flt->ib_really_quiet(), Sen->Flt->tb_sel_status(),
        Sen->Flt->cc_diff_thr(), Sen->Flt->WrapLoopAmp->ewhi_thr(), Sen->Flt->WrapLoopAmp->ewlo_thr(), Sen->Flt->WrapLoopNoa->ewhi_thr(),
        Sen->Flt->WrapLoopNoa->ewlo_thr(), Sen->Flt->ib_diff_thr(), Sen->Flt->ib_quiet_thr(), Sen->Flt->preserving(),
        ap.fake_faults(), Mon->y_ekf(), Mon->y_ekf_f(), Sen->Flt->ib_decision());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,%d,",
        Sen->Flt->WrapLoopAmp->ib_dyn_T(), Sen->Flt->WrapLoopAmp->ib_dyn_tau(),
        Sen->Flt->WrapLoopAmp->ib_dyn_rstate(), Sen->Flt->WrapLoopAmp->ib_dyn_lstate(), Sen->Flt->ib_lo_active());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,",
        Sen->Flt->WrapLoopNoa->ib_dyn_T(), Sen->Flt->WrapLoopNoa->ib_dyn_tau(),
        Sen->Flt->WrapLoopNoa->ib_dyn_rstate(), Sen->Flt->WrapLoopNoa->ib_dyn_lstate());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,%d,",
        Sen->Flt->WrapLoopNoa->ib_wrp_T(), Sen->Flt->WrapLoopNoa->ib_wrp_tau(),
        Sen->Flt->WrapLoopNoa->ib_wrp_rate(), Sen->Flt->WrapLoopNoa->ib_wrp_state(),
        Sen->Flt->disable_amp_fault());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%d,%d,%9.6f,%9.6f,%9.6f,%9.6f,%9.6f,%9.6f,",
        Sen->Flt->WrapLoopAmp->reset(), Sen->Flt->WrapLoopNoa->reset(), Sen->Flt->WrapLoopAmp->ib_wrp_T(),
        Sen->Flt->WrapLoopAmp->ib_wrp_tau(),
        Sen->Flt->WrapLoopAmp->ib_wrp_rate(), Sen->Flt->WrapLoopAmp->ib_wrp_state(), Sen->ib_amp(), Sen->ib_noa());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%d,%d,%d,%d,%9.6f,%d,%9.6f,%9.6f,%9.6f,%9.6f,",
        Sen->Flt->ib_amp_lo(), Sen->Flt->ib_amp_hi(), Sen->Flt->ib_noa_lo(), Sen->Flt->ib_noa_hi(),
        Sen->ShuntNoAmp->ishunt_cal_kf(), cp.kf_reset_print,
        Sen->ShuntAmp->kf_v(), Sen->ShuntNoAmp->kf_v(), Sen->Flt->WrapLoopAmp->e_wrap_trimmed(),
        Sen->Flt->WrapLoopNoa->e_wrap_trimmed());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,%9.6f,%d,%d,%d,",
      Sen->Flt->WrapLoopAmp->vb(), Sen->Flt->WrapLoopAmp->voc(), Sen->Flt->WrapLoopAmp->voc_soc(),
      Sen->Flt->WrapLoopNoa->voc(), Sen->Flt->WrapLoopNoa->voc_soc(),
      Sen->Flt->wrap_m_and_n_fa(), Sen->Flt->ib_is_functional(),
      Mon->voltage_low());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%11.8f,%11.8f,%11.8f,%11.8f,%11.8f,%11.8f,%11.8f,%11.8f, ",
      Sen->Tb_f_rate(), Sen->Tb_hdwe(), Sen->Tb_hdwe_f(), Sen->Tb_hdwe_f_rate(),
      Sen->Tb_hdwe_f_rstate(), Sen->Tb_hdwe_f_lstate(), Sen->Tb_hdwe_f_dt(), Sen->Tb_hdwe_f_tau());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%11.8f,%11.8f,%11.8f,%11.8f,%11.8f,%11.8f,%11.8f, ",
      Sen->Tb_model(), Sen->Tb_model_f(), Sen->Tb_model_f_rate(),
      Sen->Tb_model_f_rstate(), Sen->Tb_model_f_lstate(), Sen->Tb_model_f_dt(), Sen->Tb_model_f_tau());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%ld, %ld, %ld,", Sen->Flt->fltw(), Sen->Flt->falw(), cp.disp_word);
    Serial.printf("%s", pr.buff);

    Serial.printf("\n");
  }
}

// print sim for data collection
void print_sim_header(void)
{
  Serial.printf("unit_m,  c_time_sim,      dt_s, chm_s, qcrs_s, bms_off_s, Tb_s, Tb_f_s, vsat_s, voc_stat_s, ");
  Serial.printf("dv_dyn_s, vb_s, ib_s, ib_dyn_s, dv_hys_s, ib_in_s, ib_charge_s, ioc_s, ");
  Serial.printf("sat_s, delta_q_s, qcap_s, soc_s, reset_s, d_delta_q_s, ");
  Serial.printf("ib_dyn_T_s, ib_dyn_tau_s, ib_dyn_rstate_s, ib_dyn_lstate_s, ");
  Serial.printf("voltage_low_s,");
  Serial.printf("\n");
}

void print_sim_serial(const bool initializing_all, const bool reset_temp, Sensors *Sen, BatterySim *Sim)
{
  static double last_cTime_sim = 0.;
  if ( (sp.debug()==2 || sp.debug()==3 || sp.debug()==4 )  && cp.publishS && !initializing_all
       && (reset_temp || Sim->cTime() > last_cTime_sim + 0.00005) )
  {
    last_cTime_sim = Sim->cTime();
    sprintf(pr.buff, "unit_sim, %13.4f, %8.4f, %d, %10.4f, %d, %11.8f, %11.8f, %7.6f,%7.6f, ",
        Sim->cTime(), Sim->dt(), CHEM, Sim->q_cap_rated_scaled(), Sim->bms_off(), Sim->Tb(), Sim->Tb_f(), Sim->vsat(), Sim->voc_stat());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%7.6f,%8.6f, %7.6f,%7.6f,%7.6f,%7.6f,%7.6f,%7.6f, ",
        Sim->dv_dyn(), Sim->vb(), Sim->ib_s(), Sim->ib_dyn(), Sim->dv_hys(), Sim->ib_in(), Sim->ib_charge(), Sim->ioc());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, " %d,  %9.4f, %10.4f,  %11.9f, %d, %7.6f,",
        Sim->saturated(), Sim->delta_q(), Sim->q_capacity(), Sim->soc(), reset_temp, Sim->d_delta_q_s());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%9.6f,%9.6f,%9.6f,%9.6f,",
        Sim->chargeTransfer_T(), Sim->chargeTransfer_tau(),
        Sim->chargeTransfer_rstate(), Sim->chargeTransfer_lstate());
    Serial.printf("%s", pr.buff);

    sprintf(pr.buff, "%d,",
        Sim->voltage_low());
    Serial.printf("%s", pr.buff);

    Serial.printf("\n");
  }
}

// General purpose transmitter
void sendTxBuf(const String& txBuf, const bool sendSerial, const bool sendBLE)
{
  if ( sendSerial ) Serial.print(txBuf);
  if ( sendBLE ) bleSendChunked(txCharacteristic, reinterpret_cast<const uint8_t*>(txBuf.c_str()), txBuf.length());
}

void sendTxBuf(const char* txBuf, const bool sendSerial, const bool sendBLE)
{
  size_t bufLength = strlen(txBuf);
  if ( sendSerial ) Serial.print(txBuf);
  if ( sendBLE ) bleSendChunked(txCharacteristic, reinterpret_cast<const uint8_t*>(txBuf), bufLength);
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
  static bool serial_ready = false;

  while ( !serial_ready && Serial.available() )
  {
    char in_char = (char)Serial.read();

    if ( is_finished(in_char) )
    {
      serial_str += ';';
      serial_ready = true;
      break;
    }
    else if ( in_char == '\r' )
      Serial.printf("\n");
    else if ( in_char == '\b' && serial_str.length() )
    {
      Serial.printf("\b \b");
      serial_str.remove(serial_str.length() - 1);
    }
    else
      serial_str += in_char;
  }

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
  int answer = 0;

  sendTxBuf("\n\n", true, IN_SERVICE);
  sp.pretty_print(false);
  sendTxBuf("Reset to defaults? [y/N]:", true, IN_SERVICE);

  while ( count < 30 && answer!='Y' && answer!='y' && answer!='n' && answer!='N' )
  {
    if ( Serial.available() )
    {
      answer = Serial.read();
      if ( answer=='\r' || answer=='\n' )
      {
        answer = 'n';
        break;
      }
    }
    else if ( cp.ble_first_char != '\0' )
    {
      answer = cp.ble_first_char;
      cp.ble_first_char = '\0';
      if ( answer=='\r' || answer=='\n' )
      {
        answer = 'n';
        break;
      }
    }
    else
    {
      Serial.printf("?");
      count++;
      delay(1000);
    }
  }

  if ( answer=='Y' || answer=='y' )
  {
    sendTxBuf("  Y reset\n\n", true, IN_SERVICE);
    sp.set_nominal();
    sp.pretty_print(true);
    System.backupRamSync();
  }
  else
  {
    sendTxBuf(" N.  moving on...\n\n", true, IN_SERVICE);
  }
}
