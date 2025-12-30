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
#include "application.h"
#include "help.h"
#include "../subs.h"
#include "../command.h"
#include "../constants.h"
#include "../Summary.h"
#include "../parameters.h"
#include <math.h>
#include "../debug.h"

extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level
extern CommandPars cp;  // Various parameters shared at system level
extern Flt_st mySum[NSUM];  // Summaries for saving charge history

#undef HELPLESS

// Talk Help
void talkH(BatteryMonitor *Mon, Sensors *Sen)
{
  char buffer[32];
  sendTxBuf("No help photon for test. Look at code.\n", true, true, true);
  sendTxBuf("\n\nHelp menu.  Omit '=' and end entry with ';'\n", true, true, true);
  String txBuf;

  #ifndef HELPLESS
  sendTxBuf("\nb<?>   Manage fault buffer\n", true, true, true);
  sendTxBuf("\nb<?>   Manage fault buffer\n", true, true, true);
  sendTxBuf("  bd= dump fault buffer\n", true, true, true);
  sendTxBuf("  bh= reset history buffer\n", true, true, true);
  sendTxBuf("  br= reset fault buffer\n", true, true, true);
  sendTxBuf("  bR= reset all buffers\n", true, true, true);

  sendTxBuf("\nB<?> Battery e.g.:\n", true, true, true);
  sp.nP_p->print_help();  //* BP
  sp.nS_p->print_help();  //* BS

  sendTxBuf("\nBZ Benignly zero test settings\n", true, true, true);
  
  sendTxBuf("\ncc  clear talk queues end XQ\n", true, true, true);
  sendTxBuf("\ncf  freeze talk queues\n", true, true, true);
  sendTxBuf("\ncu  unfreeze talk queues\n", true, true, true);

  sendTxBuf("\nC<?> Chg SOC e.g.:\n", true, true, true);
  ap.init_all_soc_p->print_help();  // Ca
  sendTxBuf("  Cm=  model (& ekf if mod)- '(0-1.1)'\n", true, true, true);
  ap.ekf_x_p->print_help();  // Ce
  ap.ekf_p_p->print_help();  // Cp

  sendTxBuf("\nD/S<?> Adj e.g.:\n", true, true, true);
  sp.ib_bias_amp_p->print_help();  //* DA
  sp.ib_bias_noa_p->print_help();  //* DB
  sp.Vb_bias_hdwe_p->print_help();  //* Dc
  ap.sum_delay_p->print_help();  //  Dh
  sendTxBuf("    set 'Dh0;' for nominal\n", true, true, true);
  sp.ib_bias_all_p->print_help();  //* DI
  sp.ib_bias_amp_p->print_help();  //  Dm
  ap.eframe_mult_p->print_help();  //  ED
  ap.ib_max_amp_p->print_help();  // Mm
  ap.ib_min_amp_p->print_help();  // Mn
  ap.Ib_amp_noise_amp_p->print_help();  // DM
  sp.ib_bias_noa_p->print_help();  //  Dn
  ap.ib_max_noa_p->print_help();  // Nm
  ap.ib_min_noa_p->print_help();  // Nn
  ap.samp_points_p->print_help();  //  Cx
  ap.Ib_noa_noise_amp_p->print_help();  // DN
  ap.vc_add_p->print_help();  // D3
  ap.print_mult_p->print_help();  //  DP
  ap.temp_delay_p->print_help();  //  Dq
  ap.read_delay_p->print_help();  //  Dr
  ap.ds_voc_soc_p->print_help();  //  Ds
  sp.vsat_add_p->print_help();  //  DS
  sp.Tb_bias_hdwe_p->print_help();  //* Dt
  ap.Tb_noise_amp_p->print_help();  // DT
  ap.vb_add_p->print_help();  // Dv
  ap.Vb_noise_amp_p->print_help();  // DV
  sp.Dw_p->print_help();  //* Dw
  ap.dv_voc_soc_p->print_help();  //  Dy
  ap.Tb_bias_model_p->print_help();  // D^
  ap.talk_delay_p->print_help();  //  D>
  sp.ib_scale_amp_p->print_help();  //* SA
  sp.ib_scale_noa_p->print_help();  //* SB
  sp.ib_disch_slr_p->print_help();  //* SD
  ap.hys_scale_p->print_help();  //  Sh
  ap.hys_state_p->print_help();  //  SH
  sp.cutback_gain_slr_p->print_help();  //* Sk
  sp.s_cap_mon_p->print_help();  //* SQ
  sp.s_cap_sim_p->print_help();  //* Sq
  sp.Vb_scale_p->print_help();  //* SV
  ap.q_std_p->print_help();  // Kq
  ap.r_std_p->print_help();  // Kr

  sendTxBuf("\nF<?>   Faults\n", true, true, true);
  ap.cc_diff_slr_p->print_help();  // Fc
  ap.fake_faults_p->print_help();  // Ff
  ap.ewhi_slr_p->print_help();  // Fi
  ap.ewlo_slr_p->print_help();  // Fo
  ap.ib_quiet_slr_p->print_help();  // Fq
  ap.disab_ib_fa_p->print_help();  // FI
  ap.disab_tb_fa_p->print_help();  // FT
  ap.disab_vb_fa_p->print_help();  // FV

  sendTxBuf("\nH<?>   Manage history\n", true, true, true);
  sendTxBuf("  Hd= dump summ log\n", true, true, true);
  sendTxBuf("  HR= reset summ log\n", true, true, true);
  sendTxBuf("  Hs= save and print log\n", true, true, true);

  sendTxBuf("\nP<?>   Print values\n", true, true, true);
  sendTxBuf("  Pa= all\n", true, true, true);
  sendTxBuf("  Pb= vb details\n", true, true, true);
  sendTxBuf("  Pe= ekf\n", true, true, true);
  sendTxBuf("  Pf= faults\n", true, true, true);
  sendTxBuf("  Pm= Mon\n", true, true, true);
  sendTxBuf("  PM= amp shunt\n", true, true, true);
  sendTxBuf("  PN= noa shunt\n", true, true, true);
  sendTxBuf("  PR= all retained adj\n", true, true, true);
  sendTxBuf("  Pr= off-nom ret adj\n", true, true, true);
  sendTxBuf("  PS= Sensors\n", true, true, true);
  sendTxBuf("  Ps= Sim\n", true, true, true);
  sendTxBuf("  PV= all vol adj\n", true, true, true);
  sendTxBuf("  Pv= off-nom vol adj\n", true, true, true);
  sendTxBuf("  Px= ib select\n", true, true, true);

  sendTxBuf("\nQ      vital stats\n", true, true, true);

  sendTxBuf("\nR<?>   Reset\n", true, true, true);
  sendTxBuf("  Ca=<val> initialize_all to present inputs\n", true, true, true);
  sendTxBuf("  Rb= batteries to present inputs\n", true, true, true);
  sendTxBuf("  Rf= fault logic latches\n", true, true, true);
  sendTxBuf("  Ri= infinite counter\n", true, true, true);
  sendTxBuf("  Rk= kalman filters in shunt\n", true, true, true);
  sendTxBuf("  Rr= saturate Mon and equalize Sim & Mon\n", true, true, true);
  sendTxBuf("  RR= DEPLOY\n", true, true, true);
  sendTxBuf("  Rs= small.  Reinitialize filters\n", true, true, true);
  sendTxBuf("  RS= SavedPars: Renominalize saved\n", true, true, true);
  sendTxBuf("  RV= Renominalize volatile\n", true, true, true);

  sp.ib_force_p->print_help();  //* si
  sp.Time_now_p->print_help();  //* UT
  time_long_2_str((time_t)sp.Time_now_z, buffer);
  sendTxBuf(String::format(" time %ld hms:  %s\n", sp.Time_now_z, buffer), true, true, true);
  ap.ekf_conv_p->print_help();  // VC
  ap.ekf_q_p->print_help();  // VQ
  ap.ekf_r_p->print_help();  // VR
  ap.voc_stat_filt_p->print_help();  // VS
  ap.tb_filt_p->print_help();  // VT
  sp.debug_p->print_help();  // vv

  sendTxBuf("  -<>: Negative - Arduino plot compatible\n", true, true, true);
  sendTxBuf(" vv-2: ADS counts for throughput meas\n", true, true, true);
  #ifdef DEBUG_DETAIL
    sendTxBuf("  v-1: Debug\n", true, true, true);
  #endif
  sendTxBuf("  vv1: GP\n", true, true, true);
  sendTxBuf("  vv2: GP, Sim, Sel, & Shunt\n", true, true, true);
  sendTxBuf("  vv3: EKF\n", true, true, true);
  sendTxBuf("  vv4: GP, Sim, Sel, & EKF\n", true, true, true);
  sendTxBuf("  vv5: OLED display\n", true, true, true);
  sendTxBuf(" vv12: EKF\n", true, true, true);
  sendTxBuf("vv-13: ib_dscn\n", true, true, true);
  sendTxBuf(" vv14: vshunt and Ib raw\n", true, true, true);
  sendTxBuf(" vv15: vb raw\n", true, true, true);
  sendTxBuf(" vv16: Tb\n", true, true, true);
  sendTxBuf("vv-23: Vb_hdwe_ac\n", true, true, true);
  sendTxBuf("vv-24: Vb_hdwe_ac, Ib_hdwe\n", true, true, true);
  sendTxBuf(" vv34: EKF detail\n", true, true, true);
  sendTxBuf(" vv35: ChargeTransfer balance\n", true, true, true);
  sendTxBuf(" vv36: EKF short in EKF\n", true, true, true);
  sendTxBuf(" vv37: EKF short\n", true, true, true);
  sendTxBuf(" vv75: voc_low check mod\n", true, true, true);
  sendTxBuf(" vv76: vb model\n", true, true, true);
  sendTxBuf(" vv78: Batt model sat\n", true, true, true);
  sendTxBuf(" vv79: sat_ib model\n", true, true, true);
  sendTxBuf(" vv98: shunt filtering check\n", true, true, true);
  sendTxBuf(" vv99: calibration\n", true, true, true);

  sendTxBuf("\nW<?> - iters to wait\n", true, true, true);

  sendTxBuf("\nw - save * confirm adjustments to SRAM\n", true, true, true);

  sendTxBuf("\nX<?> - Test Mode.   For example:\n", true, true, true);
  ap.dc_dc_on_p->print_help();  // Xd
  ap.until_q_p->print_help();  // XQ
  sp.modeling_p->print_help();  //* Xm
  sp.pretty_print_modeling();

  #endif

  sp.amp_p->print_help();  //* Xa
  sp.freq_p->print_help();  //* Xf
  sp.Type_p->print_help();  //* Xt

  #ifndef HELPLESS
  sendTxBuf(" Xp= <?>, scripted tests...\n", true, true, true); 
  sendTxBuf("  Xp0: reset tests\n", true, true, true);
  sendTxBuf("  Xp6: +/-500 A pulse EKF\n", true, true, true);
  sendTxBuf("  Xp7: +/-500 A sw pulse SS\n", true, true, true);
  sendTxBuf("  Xp8: +/-500 A hw pulse SS\n", true, true, true);
  sendTxBuf("  Xp10:tweak sin\n", true, true, true);
  sendTxBuf("  Xp11:slow sin\n", true, true, true);
  sendTxBuf("  Xp12:slow half sin\n", true, true, true);
  sendTxBuf("  Xp13:tweak tri\n", true, true, true);
  sendTxBuf("  Xp20:collect fast\n", true, true, true);
  sendTxBuf("  Xp21:collect slow\n", true, true, true);
  ap.cycles_inj_p->print_help();  // XC
  sendTxBuf(" XR  RUN inj\n", true, true, true);
  sendTxBuf(" XS  STOP inj\n", true, true, true);
  ap.s_t_sat_p->print_help();  // Xs
  ap.tail_inj_p->print_help();  // XT
  ap.wait_inj_p->print_help();  // XW
  ap.fail_tb_p->print_help();  // Xu
  ap.tb_stale_time_slr_p->print_help();  // Xv
  // sp.testB_p->print_help();  // XB
  // sp.testD_p->print_help();  // XD
  // sp.testY_p->print_help();  // XY
  sendTxBuf("\nurgency of cmds: -=ASAP,*=SOON, '' or +=QUEUE, <=LAST\n", true, true, true);
  #endif
}
