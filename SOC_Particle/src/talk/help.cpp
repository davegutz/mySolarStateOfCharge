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
  sendTxBuf("No help photon for test. Look at code.\n", true, IN_SERVICE);
  sendTxBuf("\n\nHelp menu.  Omit '=' and end entry with ';'\n", true, IN_SERVICE);

  #ifndef HELPLESS
  sendTxBuf("\nb<?>   Manage fault buffer\n", true, IN_SERVICE);
  sendTxBuf("  bd= ", true, IN_SERVICE); sendTxBuf("dump fault buffer\n", true, IN_SERVICE);
  sendTxBuf("  bh= ", true, IN_SERVICE); sendTxBuf("reset history buffer\n", true, IN_SERVICE);
  sendTxBuf("  br= ", true, IN_SERVICE); sendTxBuf("reset fault buffer\n", true, IN_SERVICE);
  sendTxBuf("  bR= ", true, IN_SERVICE); sendTxBuf("reset all buffers\n", true, IN_SERVICE);

  sendTxBuf("\nB<?> Battery e.g.:\n", true, IN_SERVICE);
  ap.nP_p->print_help();  //* BP
  ap.nS_p->print_help();  //* BS

  sendTxBuf("\nBZ Benignly zero test settings\n", true, IN_SERVICE);
  
  sendTxBuf("\ncc  clear talk queues end XQ\n", true, IN_SERVICE);
  sendTxBuf("\ncf  freeze talk queues\n", true, IN_SERVICE);
  sendTxBuf("\ncu  unfreeze talk queues\n", true, IN_SERVICE);

  sendTxBuf("\nC<?> Chg SOC e.g.:\n", true, IN_SERVICE);
  ap.init_all_soc_p->print_help();  // Ca
  sendTxBuf("  Cm=  model (& ekf if mod)- '(0-1.1)'\n", true, IN_SERVICE); 
  ap.ekf_x_p->print_help();  // Ce
  ap.ekf_p_p->print_help();  // Cp

  sendTxBuf("\nD/S<?> Adj e.g.:\n", true, IN_SERVICE);
  sp.ib_bias_amp_p->print_help();  //* DA
  sp.ib_bias_noa_p->print_help();  //* DB
  sp.Vb_bias_hdwe_p->print_help();  //* Dc
  ap.sum_delay_p->print_help();  //  Dh
  sendTxBuf("    set 'Dh0;' for nominal\n", true, IN_SERVICE);
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
  ap.ib_scale_amp_p->print_help();  // SA
  ap.ib_scale_noa_p->print_help();  // SB
  sp.ib_disch_slr_p->print_help();  //* SD
  ap.hys_scale_p->print_help();  //  Sh
  ap.hys_state_p->print_help();  //  SH
  sp.cutback_gain_slr_p->print_help();  //* Sk
  ap.s_cap_mon_p->print_help();  // SQ
  ap.s_cap_sim_p->print_help();  // Sq
  ap.Vb_scale_p->print_help();  // SV
  ap.s_cap_mon_p->print_help();  // SQ
  ap.snap_wait_p->print_help();  // SW
  ap.q_std_p->print_help();  // Kq
  ap.r_std_p->print_help();  // Kr

  sendTxBuf("\nF<?>   Faults\n", true, IN_SERVICE);
  ap.cc_diff_slr_p->print_help();  // Fc
  ap.fake_faults_p->print_help();  // Ff
  ap.ewhi_slr_p->print_help();  // Fi
  ap.ewlo_slr_p->print_help();  // Fo
  ap.ib_quiet_slr_p->print_help();  // Fq
  ap.disab_ib_fa_p->print_help();  // FI
  ap.disab_tb_fa_p->print_help();  // FT
  ap.dis_vb_fa_lt_p->print_help();  // FV

  sendTxBuf("\nH<?>   Manage history\n", true, IN_SERVICE);
  sendTxBuf("  Hd= ", true, IN_SERVICE); sendTxBuf("dump summ log\n", true, IN_SERVICE);
  sendTxBuf("  HR= ", true, IN_SERVICE); sendTxBuf("reset summ log\n", true, IN_SERVICE);
  sendTxBuf("  Hs= ", true, IN_SERVICE); sendTxBuf("save and print log\n", true, IN_SERVICE);

  sendTxBuf("\nP<?>   Print values\n", true, IN_SERVICE);
  sendTxBuf("  Pa= ", true, IN_SERVICE); sendTxBuf("all\n", true, IN_SERVICE);
  sendTxBuf("  Pb= ", true, IN_SERVICE); sendTxBuf("vb details\n", true, IN_SERVICE);
  sendTxBuf("  Pe= ", true, IN_SERVICE); sendTxBuf("ekf\n", true, IN_SERVICE);
  sendTxBuf("  Pf= ", true, IN_SERVICE); sendTxBuf("faults\n", true, IN_SERVICE);
  sendTxBuf("  Pm= ", true, IN_SERVICE); sendTxBuf("Mon\n", true, IN_SERVICE);
  sendTxBuf("  PM= ", true, IN_SERVICE); sendTxBuf("amp shunt\n", true, IN_SERVICE);
  sendTxBuf("  PN= ", true, IN_SERVICE); sendTxBuf("noa shunt\n", true, IN_SERVICE);
  sendTxBuf("  PR= ", true, IN_SERVICE); sendTxBuf("all retained adj\n", true, IN_SERVICE);
  sendTxBuf("  Pr= ", true, IN_SERVICE); sendTxBuf("off-nom ret adj\n", true, IN_SERVICE);
  sendTxBuf("  PS= ", true, IN_SERVICE); sendTxBuf("Sensors\n", true, IN_SERVICE);
  sendTxBuf("  Ps= ", true, IN_SERVICE); sendTxBuf("Sim\n", true, IN_SERVICE);
  sendTxBuf("  PV= ", true, IN_SERVICE); sendTxBuf("all vol adj\n", true, IN_SERVICE);
  sendTxBuf("  Pv= ", true, IN_SERVICE); sendTxBuf("off-nom vol adj\n", true, IN_SERVICE);
  sendTxBuf("  Px= ", true, IN_SERVICE); sendTxBuf("ib select\n", true, IN_SERVICE);

  sendTxBuf("\nQ      vital stats\n", true, IN_SERVICE);

  sendTxBuf("\nR<?>   Reset\n", true, IN_SERVICE);
  sendTxBuf("  Ca=<val> ", true, IN_SERVICE); sendTxBuf("initialize_all to present inputs\n", true, IN_SERVICE);
  sendTxBuf("  Rb= ", true, IN_SERVICE); sendTxBuf("batteries to present inputs\n", true, IN_SERVICE);
  sendTxBuf("  Re= ", true, IN_SERVICE); sendTxBuf("Extended Kalman Filter in battery\n", true, IN_SERVICE);
  sendTxBuf("  Rf= ", true, IN_SERVICE); sendTxBuf("fault logic latches\n", true, IN_SERVICE);
  sendTxBuf("  Ri= ", true, IN_SERVICE); sendTxBuf("infinite counter\n", true, IN_SERVICE);
  sendTxBuf("  Rk= ", true, IN_SERVICE); sendTxBuf("kalman Filters in shunt\n", true, IN_SERVICE);
  sendTxBuf("  Rr= ", true, IN_SERVICE); sendTxBuf("saturate Mon and equalize Sim & Mon\n", true, IN_SERVICE);
  sendTxBuf("  RR= ", true, IN_SERVICE); sendTxBuf("DEPLOY\n", true, IN_SERVICE);
  sendTxBuf("  Rs= ", true, IN_SERVICE); sendTxBuf("small.  Reinitialize filters\n", true, IN_SERVICE);
  sendTxBuf("  RS= ", true, IN_SERVICE); sendTxBuf("SavedPars: Renominalize saved\n", true, IN_SERVICE);
  sendTxBuf("  RV= ", true, IN_SERVICE); sendTxBuf("Renominalize volatile\n", true, IN_SERVICE);

  sp.ib_force_p->print_help();  //* si
  sp.Time_now_p->print_help();  //* UT
  time_long_2_str((time_t)sp.Time_now(), buffer);
  sendTxBuf(String::format(" time %ld hms:  %s\n", sp.Time_now(), buffer), true, IN_SERVICE);
  ap.ekf_conv_p->print_help();  // VC
  ap.ekf_q_p->print_help();  // VQ
  ap.ekf_r_p->print_help();  // VR
  ap.voc_stat_filt_p->print_help();  // VS
  ap.Tb_filt_p->print_help();  // VT
  sp.debug_p->print_help();  // vv

  sendTxBuf("  -<>: Negative - Arduino plot compatible\n", true, IN_SERVICE);
  #ifdef DEBUG_INIT
    sendTxBuf("  v-1: Debug\n", true, IN_SERVICE);
  #endif
  sendTxBuf("  vv1: GP\n", true, IN_SERVICE);
  sendTxBuf("  vv2: GP, Sim, Sel, & Shunt\n", true, IN_SERVICE);
  sendTxBuf("  vv3: EKF\n", true, IN_SERVICE);
  sendTxBuf("  vv4: GP, Sim, Sel, & EKF\n", true, IN_SERVICE);
  sendTxBuf("  vv5: OLED display\n", true, IN_SERVICE);
  sendTxBuf(" vv12: EKF\n", true, IN_SERVICE);
  sendTxBuf("vv-13: ib_dscn\n", true, IN_SERVICE);
  sendTxBuf(" vv14: vshunt and Ib raw\n", true, IN_SERVICE);
  sendTxBuf(" vv15: vb raw\n", true, IN_SERVICE);
  sendTxBuf(" vv16: Tb\n", true, IN_SERVICE);
  sendTxBuf(" vv21: ib_quiet\n", true, IN_SERVICE);
  sendTxBuf("vv-23: Vb_hdwe_ac\n", true, IN_SERVICE);
  sendTxBuf("vv-24: Vb_hdwe_ac, Ib_hdwe\n", true, IN_SERVICE);
  sendTxBuf(" vv34: EKF detail\n", true, IN_SERVICE);
  sendTxBuf(" vv35: ChargeTransfer balance\n", true, IN_SERVICE);
  sendTxBuf(" vv36: EKF short in EKF\n", true, IN_SERVICE);
  sendTxBuf(" vv37: EKF short\n", true, IN_SERVICE);
  sendTxBuf(" vv99: calibration\n", true, IN_SERVICE);

  sendTxBuf("\nW<?> - iters to wait\n", true, IN_SERVICE);

  sendTxBuf("\nw - save * confirm adjustments to SRAM\n", true, IN_SERVICE);

  sendTxBuf("\nX<?> - Test Mode.   For example:\n", true, IN_SERVICE);
  ap.dc_dc_on_p->print_help();  // Xd
  ap.until_q_p->print_help();  // XQ
  sp.modeling_p->print_help();  //* Xm
  sp.pretty_print_modeling();

  #endif

  sp.amp_p->print_help();  //* Xa
  sp.freq_p->print_help();  //* Xf
  sp.Type_p->print_help();  //* Xt

  #ifndef HELPLESS
  sendTxBuf(" Xp= <?>, scripted tests...\n", true, IN_SERVICE); 
  sendTxBuf("  Xp0: reset tests\n", true, IN_SERVICE);
  sendTxBuf("  Xp6: +/-500 A pulse EKF\n", true, IN_SERVICE);
  sendTxBuf("  Xp7: +/-500 A sw pulse SS\n", true, IN_SERVICE);
  sendTxBuf("  Xp8: +/-500 A hw pulse SS\n", true, IN_SERVICE);
  sendTxBuf("  Xp10:tweak sin\n", true, IN_SERVICE);
  sendTxBuf("  Xp11:slow sin\n", true, IN_SERVICE);
  sendTxBuf("  Xp12:slow half sin\n", true, IN_SERVICE);
  sendTxBuf("  Xp13:tweak tri\n", true, IN_SERVICE);
  sendTxBuf("  Xp20:collect fast\n", true, IN_SERVICE);
  sendTxBuf("  Xp21:collect slow\n", true, IN_SERVICE);
  ap.cycles_inj_p->print_help();  // XC
  sendTxBuf(" XD  ", true, IN_SERVICE); sendTxBuf("DONE message\n", true, IN_SERVICE);
  sendTxBuf(" XK  ", true, IN_SERVICE); sendTxBuf("READY message\n", true, IN_SERVICE);
  sendTxBuf(" XR  ", true, IN_SERVICE); sendTxBuf("RUN inj\n", true, IN_SERVICE);
  sendTxBuf(" XS  ", true, IN_SERVICE); sendTxBuf("STOP inj\n", true, IN_SERVICE);
  sendTxBuf(" XY  ", true, IN_SERVICE); sendTxBuf("SYNC message\n", true, IN_SERVICE);
  ap.s_t_sat_p->print_help();  // Xs
  ap.tail_inj_p->print_help();  // XT
  ap.wait_inj_p->print_help();  // XW
  ap.fail_tb_p->print_help();  // Xu
  ap.tb_stale_time_slr_p->print_help();  // Xv
  sendTxBuf("\nurgency of cmds: -=ASAP,*=SOON, '' or +=QUEUE, <=LAST\n", true, IN_SERVICE);
  #endif
}
