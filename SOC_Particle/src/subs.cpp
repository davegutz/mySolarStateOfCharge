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

#include "subs.h"
#include "command.h"
#include "constants.h"
#include <math.h>
#include "debug.h"
#include "Summary.h"
#include "talk/chitchat.h"
#include "ble.h"
#include "Sensors.h"

extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level
extern CommandPars cp;  // Various parameters shared at system level
extern PrinterPars pr;  // Print buffer
extern PublishPars pp;  // For publishing
extern Flt_st mySum[NSUM];
extern Pins *myPins;
extern BleCharacteristic txCharacteristic;
extern BleCharacteristic rxCharacteristic;

void sample_burst(Pins *myPins, Sensors *Sen)
{
  uint32_t local_micros_init = micros();
  if ( ap.samp_points() > 0 )
  {
    Serial.printf("u_cx, time, vovcn, vovcnkf,\n");
    for (uint32_t i=0; i<ap.samp_points(); i++ )  // Cx
    {
      uint32_t local_micros = micros();
      if ( i== 0 ) local_micros_init = local_micros;
      Sen->ShuntNoAmp->sample_Vo();
      Sen->ShuntNoAmp->sample_Vc();
      Sen->ShuntNoAmp->sample_combine();
      Sen->ShuntNoAmp->sample_filter_kf(false);
      Serial.printf("cx_u, %8.6f, %8.6f, %8.6f,\n",
        (local_micros - local_micros_init)*1e-6, Sen->ShuntNoAmp->Vo_Vc(), Sen->ShuntNoAmp->Vo_Vc_kf());
    }
    ap.samp_points(0);
  }
}

// Harvest charge caused temperature change.   More charge becomes available as battery warms
void harvest_temp_change(const double Tb_f, BatteryMonitor *Mon, BatterySim *Sim, const float tb_rate, const float dt)
{
#ifdef DEBUG_INIT
if ( sp.debug()==-1 ) Serial.printf("entry harvest_temp_change:  Delta_q %10.1f Tb_f %5.1f delta_q_model %10.1f tb_s %5.1f\n",
  sp.delta_q(), Tb_f, sp.delta_q_model(), Tb_f);
#endif
  sp.put_Delta_q(sp.delta_q() - Mon->dqdt() * Mon->q_capacity() * tb_rate * dt);
  sp.put_delta_q_model(sp.delta_q_model() - Sim->dqdt() * Sim->q_capacity() * tb_rate * dt);
#ifdef DEBUG_INIT
if ( sp.debug()==-1 ) Serial.printf("exit harvest_temp_change:  Delta_q %10.1f Tb_f %5.1f delta_q_model %10.1f tb_s %5.1f\n",
  sp.delta_q(), Tb_f, sp.delta_q_model(), Tb_f);
#endif
}

// Complete initialization of all parameters in Mon and Sim including EKF
// Force current to be zero because initial condition undefined otherwise with charge integration
void initialize_all(BatteryMonitor *Mon, Sensors *Sen, const float soc_in, const bool use_soc_in)
{
  // Sample debug statements
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 )
    {
      Serial.printf("\n\n");
      sp.pretty_print(true);
      cp.pretty_print();
      Serial.printf("Entering initialize_all: use_soc_in %d soc_in %5.3f, falw %ld Tb_fa %d\n", use_soc_in, soc_in, Sen->Flt->falw(), Sen->Flt->Tb_fa());
      debug_m1(Mon, Sen);
    }
  #endif

  // Gather and apply inputs
  if ( sp.mod_ib() )
    Sen->Ib_model_in(sp.inj_bias() + sp.ib_bias_all());
  else
    Sen->Ib_model_in(Sen->Ib_hdwe());
  if ( sp.mod_tb() )
  {
    Sen->Tb(Sen->Tb_model());
    Sen->Tb_f(Sen->Tb_model_f());
    Sen->Tb_f_rate(Sen->Tb_model_f_rate());
  }
  else
  {
    Sen->Tb(Sen->Tb_hdwe());
    Sen->Tb_f(Sen->Tb_hdwe_f());
    Sen->Tb_f_rate(Sen->Tb_hdwe_f_rate());
  }
  if ( use_soc_in )
  {
    Mon->apply_soc(soc_in, Sen->Tb_f());  // saves sp.delta_q and sp.T_state
    Sen->Sim->apply_soc(soc_in, Sen->Tb_f());  // saves sp.delta_q and sp.T_state
  }
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 )
    { 
      Serial.printf("before harvest_temp, use_soc_in %d falw %ld Tb_fa %d:  ", use_soc_in, Sen->Flt->falw(), Sen->Flt->Tb_fa()); debug_m1(Mon, Sen);
    }
  #endif
  if ( !Sen->Flt->Tb_fa() ) harvest_temp_change(Sen->Tb_f(), Mon, Sen->Sim, Sen->Tb_f_rate(), 0.);
  
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("after harvest_temp:  cp.soft_sim_hold %d:  ", cp.soft_sim_hold); debug_m1(Mon, Sen);}
  #endif

  if ( cp.soft_sim_hold )  
    Sen->Sim->apply_delta_q_t(Sen->Sim->delta_q(), Sen->Sim->Tb_f());  // applies sp.delta_q and sp.T_state
  else
    Sen->Sim->apply_delta_q_t(Mon->delta_q(), Mon->Tb_f());  // applies sp.delta_q and sp.T_state

  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("S.a_d_q_t: cp.soft_sim_hold %d:  ", cp.soft_sim_hold); debug_m1(Mon, Sen);}
  #endif

  // Make Sim accurate even if not used
  Sen->Sim->init_battery_sim(true, Sen);
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("S.i_b:"); debug_m1(Mon, Sen);}
  #endif
  if ( !sp.mod_vb() )
  {
    Sen->Sim->apply_soc(Sen->Sim->soc(), Sen->Tb_f());
  }
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("S.a_b: sp.mod_vb() %d:  ", sp.mod_vb()); debug_m1(Mon, Sen);}
  #endif
  // Call calculate twice because sat_ is a used-before-calculated (UBC)
  // Simple 'call twice' method because sat_ is discrete no analog which would require iteration
  Sen->Vb_model(Sen->Sim->calculate(Sen, ap.dc_dc_on(), true) * ap.nS());
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("S.a_b1:  "); debug_m1(Mon, Sen);}
  #endif
  Sen->Vb_model(Sen->Sim->calculate(Sen, ap.dc_dc_on(), true) * ap.nS());  // Call again because sat is a UBC
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("S.a_b2:  "); debug_m1(Mon, Sen);}
  #endif
  Sen->Ib_model(Sen->Sim->ib_fut() * ap.nP());
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("S.a_b3:  "); debug_m1(Mon, Sen);}
  #endif

  // Call to count_coulombs not strictly needed for init.  Calculates some things not otherwise calculated for 'all'
  // Need sat initialized before here
  Sen->Sim->count_coulombs(Sen, true, Mon, true);
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("S.a_b4:  "); debug_m1(Mon, Sen);}
  #endif

  // Signal preparations
  if ( sp.mod_vb() )
  {
    Sen->Vb(Sen->Vb_model());
  }
  else
  {
    Sen->Vb(Sen->Vb_hdwe());
  }
  if ( sp.mod_ib() )
  {
    Sen->Ib(Sen->Ib_model());
  }
  else
  {
    Sen->Ib(Sen->Ib_hdwe());
  }
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("SENIB:  "); debug_m1(Mon, Sen);}
  #endif
  if ( sp.mod_vb() && !cp.soft_sim_hold )
  {
    Mon->apply_soc(Sen->Sim->soc(), Sen->Tb_f());
  }
  Mon->init_battery_mon(true, Sen);
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ) { Serial.printf("M.i_b:  "); debug_m1(Mon, Sen);}
  #endif

  // Call calculate/count_coulombs twice because sat_ is a used-before-calculated (UBC)
  // Simple 'call twice' method because sat_ is discrete not analog which would require iteration
  Mon->calculate(Sen, true, true);
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("M.calc1:  "); debug_m1(Mon, Sen);}
  #endif
  Mon->count_coulombs(Sen, true, 0., Mon->is_sat(true), Mon->is_sat(true));
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("M.c_c1:  "); debug_m1(Mon, Sen);}
  #endif
  Mon->calculate(Sen, true, true);  // Call again because sat is a UBC
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("M.calc2:  "); debug_m1(Mon, Sen);}
  #endif
  Mon->count_coulombs(Sen, true, 0., Mon->is_sat(true), Mon->is_sat(true));  // Call again because sat is a UBC
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("M.c_c2:  "); debug_m1(Mon, Sen);}
  #endif
  
  // // Solve EKF
  #ifdef DEBUG_INIT
    if ( sp.debug()==-1 ){ Serial.printf("end:  "); debug_m1(Mon, Sen);}
  #endif

  // Finally....clear all faults
  Sen->Flt->reset_all_faults();
}

// Load high fidelity signals; filtered in hardware the same bandwidth, sampled the same
// Outputs:   Sen->Ib_model_in, Sen->Ib_hdwe, 
void load_ib_vb_tb(const bool reset, const bool reset_temp, const bool reset_kf, Sensors *Sen, Pins *myPins, BatteryMonitor *Mon)
{
  // Load shunts Ib
  // Outputs:  Sen->Ib_model_in, Sen->Ib_hdwe, Sen->Vb, Sen->Wb
  Sen->ShuntAmp->convert(sp.mod_ib_amp_dscn(), reset_kf, Sen);
  Sen->ShuntNoAmp->convert(sp.mod_ib_noa_dscn(), reset_kf, Sen);
  Sen->Flt->vc_check(Sen, Mon, VC_MIN, VC_MAX, reset);
  Sen->shunt_select_initial(reset);
  if ( sp.debug()==14 ) Sen->shunt_print();
  #ifdef DEBUG_INIT
    if ( sp.debug()==62 ) Sen->select_print(Sen, Mon);
  #endif

  // Load voltage Vb
  // Outputs:  Sen->Vb
  Sen->vb_load(myPins->Vb_pin, reset);
  if ( !sp.mod_vb_dscn() )  Sen->Flt->vb_check(Sen, Mon, VB_MIN, VB_MAX, reset);
  else                      Sen->Flt->vb_check(Sen, Mon, -1.0, 1.0, reset);
  if ( sp.debug()==15 ) Sen->vb_print();

  // Load temperature Tb
  Sen->Tb_load(myPins->VTb_pin, reset);
  Sen->Flt->Tb_check(Sen, TB_MIN, TB_MAX,  reset);  // Sets Tb_fa()
  if ( sp.debug()==18 ) Sen->Tb_print();

  // Power calculation
  Sen->Wb(Sen->Vb()*Sen->Ib());
}

// Calculate Ah remaining for display to user
// Inputs:  sp.mon_chm, Sen->Ib, Sen->Vb, Sen->Tb_f
// States:  Mon.soc, Mon.soc_ekf
// Outputs: tcharge_wt, tcharge_ekf, Voc, Voc_filt
void  monitor(const bool reset, const bool reset_temp, const bool reset_ekf, const uint64_t now,
  TFDelay *Is_sat_delay, BatteryMonitor *Mon, Sensors *Sen)
{
  // EKF - calculates tb_f_, voc_stat_, voc_ as functions of sensed parameters vb & ib (not soc)
  Mon->calculate(Sen, reset_temp, reset_ekf);

  // Debounce saturation calculation done in ekf using voc model
  Sen->sat(Mon->is_sat(reset));
  Sen->saturated(Is_sat_delay->calculate(Sen->sat(), T_SAT*ap.s_t_sat(), T_DESAT*ap.s_t_sat(), min(Sen->T(), T_SAT/2.), reset));

  // Memory store
  // Both ib sensors hard-failed: force 0 into the Coulomb counter (the EKF was
  // already fed 0 inside Mon->calculate above). Belt-and-suspenders behind the
  // UsingNone selection upstream and the zeroing inside BatteryMonitor::calculate.
  float cc_ib_in = Mon->ib_charge();
  if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() && !ap.fake_faults() )
    cc_ib_in = 0.;
  Mon->count_coulombs(Sen, reset_temp, cc_ib_in, Sen->sat(), Sen->saturated());

  // Charge charge time for display
  Mon->calc_charge_time(Mon->q(), Mon->q_capacity(), Sen->ib(), Mon->soc());
}

// Read sensors, model signals, select between them.
// Sim used for any missing signals (Tb, Vb, Ib)
//    Needed here in this location to have available a value for
//    Sen->Tb_f when called.   Recalculates Sen->Ib accounting for
//    saturation.  Sen->Ib is a feedback (used-before-calculated).
// Inputs:  sp.config, sp.sim_chm, Sen->Tb, Sen->Ib_model_in
// States:  Sim.soc
// Outputs: Sim.Tb_f, Sen->Ib, Sen->Ib_model,
//   Sen->Vb_model, Sen->Tb_f, sp.inj_bias
void sense_synth_select(const bool reset, const bool reset_temp, const bool reset_kf,
   const uint64_t now, const uint64_t elapsed,  Pins *myPins, BatteryMonitor *Mon, Sensors *Sen)
{
  static uint64_t last_snap = now;
  bool storing_fault_data = ( now - last_snap )>SNAP_WAIT;
  if ( storing_fault_data || reset ) last_snap = now;

  // Load Ib and Vb
  // Outputs: Sen->Ib_model_in, Sen->Ib, Sen->Vb
  load_ib_vb_tb(reset, reset_temp, reset_kf, Sen, myPins, Mon);

  // Sim initialize as needed from memory
  if ( reset_temp )
  {
    initialize_all(Mon, Sen, 0., false);
  }
  Sen->Sim->apply_delta_q_t(reset);
  Sen->Sim->init_battery_sim(reset, Sen);
  Mon->init_battery_mon(reset, Sen);

  // Sim calculation
  //  Inputs:  Sen->Tb_f(past), Sen->Ib_model_in
  //  States: Sim->soc(past)
  //  Outputs:  Tb_hdwe, Ib_model, Vb_model, sp.inj_bias, Sim.model_saturated
  Sen->Vb_model(Sen->Sim->calculate(Sen, ap.dc_dc_on(), reset) * ap.nS() + Sen->Vb_add());
  Sen->Ib_model(Sen->Sim->ib_fut() * ap.nP());
  cp.model_cutback = Sen->Sim->cutback();
  cp.model_saturated = Sen->Sim->saturated();

  // Apply noise to model values BEFORE fault logic so ib_diff sees the same values that get logged.
  // Inputs:  Sim->Ib
  Sen->Ib_amp_model(max(min(Sen->Ib_model() + Sen->Ib_amp_add() + Sen->Ib_amp_noise(), Sen->Ib_amp_max()), Sen->Ib_amp_min()));  // Dm
  Sen->Ib_noa_model(max(min(Sen->Ib_model() + Sen->Ib_noa_add() + Sen->Ib_noa_noise(), Sen->Ib_noa_max()), Sen->Ib_noa_min()));  // Dn

  Sen->Flt->ib_range(reset, Sen, Mon);
  Sen->Flt->ib_logic(reset, Sen, Mon);
  Sen->Flt->ib_wrap(reset, Sen, Mon);
  Sen->Flt->ib_quiet(reset, Sen);
  Sen->Flt->cc_diff(reset, Sen, Mon);
  Sen->Flt->ib_diff(reset, Sen, Mon);

  // Select
  //  Inputs:                                       --->   Outputs:
  //  Ib_model, Ib_hdwe, Vc_hdwe                    --->   Ib
  //  Vb_model, Vb_hdwe,                            --->   Vb
  //  constant,         Tb_hdwe, Tb_hdwe_f        --->   Tb, Tb_f
  // Log.info("  sense_synth_select:  select_all_logic");
  Sen->Flt->select_all_logic(Sen, Mon, reset);
  // Log.info("  sense_synth_select:  select_volt_and_current_and_temp");
  Sen->select_volt_and_current_and_temp(Mon);

  // Fault snap buffer management
  static uint8_t fails_repeated = 0;
  if ( Sen->Flt->reset_all_faults() )
  {
    fails_repeated = 0;
    Sen->Flt->preserving(false);
  }
  static bool record_past = Sen->Flt->record();
  bool instant_of_failure = record_past && !Sen->Flt->record();
  if ( storing_fault_data || instant_of_failure )
  {
    if ( Sen->Flt->record() ) fails_repeated = 0;
    else fails_repeated = min(fails_repeated + 1, 99);
    if ( fails_repeated < 3 )
    {
      sp.put_Iflt(sp.iflt() + 1);
      if ( sp.iflt()>sp.nflt() - 1 ) sp.put_Iflt(0);  // wrap buffer
      Flt_st fault_snap;
      fault_snap.assign_unfilt(Sen->now(), Mon, Sen);
      sp.put_fault(fault_snap, sp.iflt());
    }
    else if ( fails_repeated < 4 )
    {
      Serial.printf("preserving fault buffer\n");
      Sen->Flt->preserving(true);
    }
    if ( instant_of_failure ) last_snap = now;
  }
  record_past = Sen->Flt->record();

  // Charge calculation and memory store
  // Inputs: Sim.model_saturated, Sen->Tb, Sen->Ib
  // States: Sim.soc
  // Log.info("  sense_synth_select:  Sen->Sim->count_coulombs");
  Sen->Sim->count_coulombs(Sen, reset_temp, Mon, false);

  // Injection test
  if ( (Sen->start_inj() <= Sen->now()) && (Sen->now() <= Sen->end_inj()) && (Sen->now() > 0ULL) ) // in range, test in progress
  {
    // Shift times because sampling is asynchronous: improve repeatibility
    if ( Sen->elapsed_inj()==0ULL )
    {
      Sen->end_inj(Sen->end_inj() + Sen->now() - Sen->start_inj());
      Sen->stop_inj(Sen->stop_inj() + Sen->now() - Sen->start_inj());
      Sen->start_inj(Sen->now());
      Serial.printf("SYNC,%7.3f\n", double(Sen->now())/1000.);
    }

    Sen->elapsed_inj(Sen->now() - Sen->start_inj() + 1UL); // Shift by 1 because using ==0 as reset button

    // Put a stop to this but retain sp.amp_ to scale fault and history printouts properly
    if (Sen->now() > Sen->stop_inj())
    {
      sp.put_Inj_bias(0);
      sp.put_Type(0);
    }
  }

  else if ( Sen->elapsed_inj() && sp.tweak_test() )  // Done.  elapsed_inj set to 0 is the reset button
  {
    Serial.printf("STOP echo\n");
    Sen->elapsed_inj(0ULL);
    chit("vv0;", ASAP);    // Turn off echo
    chit("Xp0;", SOON);    // Reset
  }
  Sen->Sim->calc_inj(Sen->elapsed_inj(), sp.type(), sp.Amp(ap.nP()), sp.freq());

  // Quiet logic.   Reset to ready state at soc=0.5; do not change Modeling.  Passes at least once before running chit.
  static uint64_t millis_past = millis();
  static uint32_t until_q_past = ap.until_q();
  if ( ap.until_q()>0UL && until_q_past==0UL ) until_q_past = ap.until_q();
  ap.until_q( (uint32_t) max(0, (long) ap.until_q()  - (long)(millis() - millis_past)) );
  if ( ap.until_q()==0UL && until_q_past>0UL )
  {
    chit("BZ;", SOON);
    cp.freeze = false;  // unfreeze the queues
  }
  until_q_past = ap.until_q();
  millis_past = millis();

}


// Serial display function
void serial_display(Sensors *Sen, BatteryMonitor *Mon)
{
  static uint8_t blink = 0;
  String disp_0, disp_1, disp_2;
  cp.clear_disp_word();

  // ---------- Top Line of Display -------------------------------------------
  // Tb
  sprintf(pr.buff, "%3.0f", pp.pubList.Tb);
  disp_0 = pr.buff;  // Default
  if ( Sen->Flt->Tb_fa() && (blink==0 || blink==1) )
  {
    disp_0 = "***";
    dispAssign(true, flt_tb);
  }

  // Voc
  sprintf(pr.buff, "%5.2f", pp.pubList.Voc);
  disp_1 = pr.buff;  // Default
  if ( Sen->Flt->vb_sel_stat()==0 && (blink==1 || blink==2) )
  {
    disp_1 = "*fail";
    dispAssign(true, fail_vb);
  }
  else if ( Sen->bms_off() )
  {
    disp_1 = " off ";
    dispAssign(true, off);
  }

  // Ib
  sprintf(pr.buff, "%6.1f", pp.pubList.Ib);
  disp_2 = pr.buff;  // Default
  #ifdef HDWE_IB_HI_LO
    if ( blink==2 )
    {
      if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() )
      {
        disp_2 = "*fail";
        dispAssign(true, fail_ib);
      }
      else if ( Sen->Flt->ib_choice()==1 )
      {
        disp_2 = "*fail";
        dispAssign(true, fail_ib);
      }
      else if ( Sen->Flt->ib_choice()==-1 )
      {
        disp_2 = "*amp";
        dispAssign(true, fail_ibm);
      }
      // auto section
      else if ( Sen->Flt->ib_diff_fa() )
      {
        disp_2 = " diff ";
        dispAssign(true, diff_ib);
      }
      // another default
      else if ( Sen->Flt->ib_choice()!=0 )
      {
        disp_2 = " redl ";
        dispAssign(true, red_loss);
      }
    }
    else if ( blink==3 )
    {
      if ( Sen->Flt->dscn_fa() && !sp.mod_ib() )
      {
        disp_2 = " conn ";
        dispAssign(true, conn);
      }
    }
  #else
    if ( blink==2 )
    {
      if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() && !sp.mod_ib() )
      {
        disp_2 = "*fail";
        dispAssign(true, fail_vb);
      }
      else if ( Sen->Flt->dscn_fa() && !sp.mod_ib() )
      {
        disp_2 = " conn ";
        dispAssign(true, conn);
      }
      else if ( Sen->Flt->ib_diff_fa() )
      {
        disp_2 = " diff ";
        dispAssign(true, diff_ib);
      }
      else if ( Sen->Flt->red_loss() )
      {
        disp_2 = " redl ";
        dispAssign(true, red_loss);
      }
    }
    else if ( blink==3 )
    {
      if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() && !sp.mod_ib() )
      {
        disp_2 = "*fail";
        dispAssign(true, fail_vb);
      }
      else if ( Sen->Flt->dscn_fa() && !sp.mod_ib() )
      {
        disp_2 = " conn ";
        dispAssign(true, conn);}
  #endif
  String disp_Tbop = disp_0.substring(0, 4) + " " + disp_1.substring(0, 6) + " " + disp_2.substring(0, 7);

  // --------------------- Bottom line of Display ------------------------------
  // A-Hrs EKF
  sprintf(pr.buff, "%3.0f", pp.pubList.Amp_hrs_remaining_ekf);
  disp_0 = pr.buff;  // Default

  #ifdef HDWE_IB_HI_LO
    if ( blink==0 || blink==1 || blink==2 )
    {
      if ( Sen->Flt->cc_diff_fa() && !Sen->Flt->ib_diff_fa() )
      {
        disp_0 = "---";
        dispAssign(true, flt_ekf);
      }
    }
  #else
    if ( blink==0 || blink==1 || blink==2 )
    {
      if ( Sen->Flt->cc_diff_fa() )
        disp_0 = "---";
        dispAssign(true, flt_ekf);
    }
  #endif

  // t charge
  if ( abs(pp.pubList.tcharge) < 24. )
  {
    sprintf(pr.buff, "%5.1f", pp.pubList.tcharge);
  }
  else
  {
    sprintf(pr.buff, " --- ");
    dispAssign(true, time_long);
  }
  disp_1 = pr.buff;

  // A-Hrs Coulomb counter remaining
  #ifdef HDWE_IB_HI_LO
    sprintf(pr.buff, "%3.0f", pp.pubList.Amp_hrs_remaining_soc);
    if ( Sen->saturated() && blink==0 )
    {
      disp_2 = "SAT";
      dispAssign(true, SAT);
    }
    else if ( blink==2 )
    {
      if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() )
      {
        disp_2 = "fail";
        dispAssign(true, fail_ib);
    }
      else if ( Sen->Flt->ib_choice()!=0 )
      {
        disp_2 = "accy";
        dispAssign(true, accy);
      }
      else
      {
        disp_2 = pr.buff;
      }
    }
    else
    {
      disp_2 = pr.buff;
    }
  #else
    sprintf(pr.buff, "%3.0f", min(pp.pubList.Amp_hrs_remaining_soc, 999.));
    if (Sen->saturated() && blink==0)
    {
      disp_2 = "SAT";
      dispAssign(true, SAT);
    }
    else if ( blink==2 )
    {
      if ( Sen->Flt->ib_amp_fa() && Sen->Flt->ib_noa_fa() && !sp.mod_ib() )
      {
        disp_2 = "fail";
        dispAssign(true, fail_ib);
      }
      else if ( ( Sen->Flt->ib_amp_fa() || Sen->Flt->ib_noa_fa() ) && !sp.mod_ib() )
      {
        disp_2 = "accy";
        dispAssign(true, accy);
      }
      else
      {
        disp_2 = pr.buff;
      }
    }
    else
    {
      disp_2 = pr.buff;
    }
  #endif
  String dispBot = disp_0 + disp_1 + " " + disp_2;

  // Text basic Bluetooth (use serial bluetooth app)
  debug_check_99(Mon, Sen);
  debug_check_98(Mon, Sen);
  if ( sp.debug()!=-2 && sp.debug()==5 )  // Normal display as long as 'vv5'
  {
    String txBuf;
    txBuf = String::format("%s   Tb,C  VOC,V  Ib,A \n%s   EKF,Ah  chg,hrs  CC, Ah\nPf; for fails.  prints=%ld\n\n",
        disp_Tbop.c_str(), dispBot.c_str(), cp.num_v_print);
    sendTxBuf(txBuf, true, IN_SERVICE);
  }
  else if ( 1<=sp.debug() && sp.debug()<=4 )  // Normal BLE display as long as 'vv4' so can watch GUI_test in progress
  {
    String txBuf;
    txBuf = String::format("%s   Tb,C  VOC,V  Ib,A \n%s   EKF,Ah  chg,hrs  CC, Ah\nPf; for fails.  prints=%ld\n\n",
        disp_Tbop.c_str(), dispBot.c_str(), cp.num_v_print);
    sendTxBuf(txBuf, false, true);
  }

  blink += 1;
  if (blink>3) blink = 0;

  #ifdef DEBUG_INIT
    if ( sp.debug()==63 )
    {
      #ifdef HDWE_IB_HI_LO
        Serial.printf("\nmodib %d ibchc %d vbsst %d tbfa %d ibmfa %d ibnafa %d ibdiffa %d dscnfa %d redloss %d\n",
            sp.mod_ib(), Sen->Flt->ib_choice(), Sen->Flt->vb_sel_stat(), Sen->Flt->Tb_fa(), Sen->Flt->ib_amp_fa(), Sen->Flt->ib_noa_fa(),  Sen->Flt->ib_diff_fa(), Sen->Flt->dscn_fa(), Sen->Flt->red_loss());
        Serial.printf("%s   Tb,C  VOC,V  Ib,A \n%s   EKF,Ah  chg,hrs  CC, Ah\nPf; for fails.  prints=%ld\n\n",
            disp_Tbop.c_str(), dispBot.c_str(), cp.num_v_print);
      #else
        Serial.printf("\nmodib %d ibsst %d vbsst %d tbfa %d ibmfa %d ibnafa %d ibdiffa %d dscnfa %d redloss %d\n",
            sp.mod_ib(), Sen->Flt->ib_sel_stat(), Sen->Flt->vb_sel_stat(), Sen->Flt->Tb_fa(), Sen->Flt->ib_amp_fa(), Sen->Flt->ib_noa_fa(),  Sen->Flt->ib_diff_fa(), Sen->Flt->dscn_fa(), Sen->Flt->red_loss());
        Serial.printf("%s   Tb,C  VOC,V  Ib,A \n%s   EKF,Ah  chg,hrs  CC, Ah\nPf; for fails.  prints=%ld\n\n",
            disp_Tbop.c_str(), dispBot.c_str(), cp.num_v_print);
      #endif
    }
  #endif
}


// Time synchro for web information
void sync_time(uint64_t now, uint64_t *last_sync, uint64_t *millis_flip)
{
  *last_sync = millis();

  // Request time synchronization from the Particle Cloud
  if ( Particle.connected() ) Particle.syncTime();

  // Refresh millis() at turn of Time.now
  int count = 0;
  long time_begin = Time.now();  // Seconds since start of epoch
  while ( Time.now()==time_begin && ++count<1100 )  // Time.now() truncates to seconds
  {
    delay(1);
    *millis_flip = millis()%1000;
  }
}


// Initialize serial and BLE in setup()
void setup_serial_ble()
{
  Serial.begin(SOFT_SBAUD);
  Serial.flush();
  delay(1000);
  sendTxBuf("Hi!\n", true, IN_SERVICE);

  BLE.on();
  BLE.addCharacteristic(txCharacteristic);
  BLE.addCharacteristic(rxCharacteristic);
  BleAdvertisingData data;
  data.appendServiceUUID(serviceUuid);
  BLE.advertise(&data);
}

// Configure hardware pins and report temperature sensor type in setup()
void setup_pins()
{
  myPins = new Pins(D7, D12, D11, D13, D14, D0, true);
  pinMode(myPins->status_led, OUTPUT);
  digitalWrite(myPins->status_led, LOW);

  #if defined(HDWE_BARE)
    sendTxBuf("Going naked\n", true, IN_SERVICE);
  #elif defined(HDWE_2WIRE)
    sendTxBuf("Using 2Wire Temperature sensor\n", true, IN_SERVICE);
  #else
    #error "Temperature sensor undefined"
  #endif
}

// Check retained memory for corruption; reset to nominal if found
void check_and_fix_corruption()
{
  sendTxBuf("Check corruption......", true, IN_SERVICE);
  bool corrupt = sp.is_corrupt();
  if ( corrupt )
  {
    sendTxBuf("\n\n", true, IN_SERVICE);
    sp.pretty_print(false);
    sendTxBuf("\n\n", true, IN_SERVICE);
    sp.set_nominal();
    sendTxBuf("Fixed corruption\n", true, IN_SERVICE);
    sp.pretty_print(true);
  }
  else sendTxBuf("\nclean\n", true, IN_SERVICE);
}

// Handle booted flag and optional renominalize prompt in setup()
void handle_boot_sequence()
{
  sp.get_booted();
  sendTxBuf(String::format("booted = %d\n", sp.booted()), true, IN_SERVICE);
  if ( ASK_DURING_BOOT == 0 && !sp.booted() )
  {
    sp.set_nominal();
    sp.put_booted(true);
    sendTxBuf("\n\nSet booted true and stored...", true, IN_SERVICE);
    System.backupRamSync();
    delay(1000);
    sendTxBuf("backup Ram synced *\n", true, IN_SERVICE);
    sp.get_booted();
    sendTxBuf(String::format("booted = %d\n", sp.booted()), true, IN_SERVICE);
    sendTxBuf("booted should be true\n\n", true, IN_SERVICE);
    delay(1000);
  }
  if ( ASK_DURING_BOOT == 1 )
  {
    if ( sp.num_diffs() )
      wait_on_user_input();
  }
}

// Rate-divide the read frame to set cp.publishS true every print_mult reads
void update_publish_frame()
{
  static uint8_t print_count = 0;
  if ( print_count >= ap.print_mult()-1 || print_count == UINT8_MAX )
  {
    print_count = 0;
    cp.publishS = true;
  }
  else
  {
    print_count++;
    cp.publishS = false;
  }
}

// Rotate history and summary ring buffers on schedule or manual request
void manage_summaries(const bool boot_wait, const bool summarizing, BatteryMonitor *Mon, Sensors *Sen)
{
  if ( (!boot_wait && summarizing) || cp.write_summary )
  {
    sp.put_Ihis(sp.ihis() + 1);
    if ( sp.ihis() > (sp.nhis() - 1) ) sp.put_Ihis(0);
    Flt_st hist_snap, hist_bounced;
    hist_snap.assign(Sen->now(), Mon, Sen);
    hist_bounced = sp.put_history(hist_snap, sp.ihis());

    sp.put_Isum(sp.isum() + 1);
    if ( sp.isum() > (uint16_t)(sp.nsum()-1) ) sp.put_Isum(0);
    mySum[sp.isum()].copy_to_Flt_ram_from(hist_bounced);
    sendTxBuf("Summ...\n", true, IN_SERVICE);
    cp.write_summary = false;
  }
}

// Apply pending soft/ekf/kf reset command-par flags to loop state variables
void handle_soft_reset(bool *reset, bool *reset_temp, bool *reset_kf, bool *reset_ekf, uint64_t *start_reset, const bool read)
{
  if ( read ) cp.soft_sim_hold = false;
  cp.soft_reset_print = cp.soft_reset;
  cp.soft_reset_sim_print = cp.soft_reset_sim;
  if ( cp.soft_reset || cp.soft_reset_sim )
  {
    *reset = *reset_temp = *reset_kf = true;
    *start_reset = millis();
    if ( cp.soft_reset_sim ) cp.cmd_soft_sim_hold();
  }
  if ( cp.ekf_reset ) cp.ekf_reset_print = *reset_ekf = true;
  if ( cp.kf_reset ) cp.kf_reset_print = *reset_kf = true;
  cp.soft_reset = cp.soft_reset_sim = cp.ekf_reset = cp.kf_reset = false;
}

// For summary prints
String time_long_2_str(const time_t time, char *tempStr)
{
    // Serial.printf("Time.year:  time_t %d ul %d as-is %d\n", 
    //   Time.year((time_t) 1703267248), Time.year((uint32_t )1703267248), Time.year(time));
    uint32_t year = Time.year(time);
    uint8_t month = Time.month(time);
    uint8_t day = Time.day(time);
    uint8_t hours = Time.hour(time);
    uint8_t minutes   = Time.minute(time);
    uint8_t seconds   = Time.second(time);
    sprintf(tempStr, "%4u-%02u-%02uT%02u:%02u:%02u", int(year), month, day, hours, minutes, seconds);
    // Serial.printf("time_long_2_str: %lld %ld %d %d %d %d %d\n", time, year, month, day, hours, minutes, seconds);
    // sprintf(tempStr, "time_long_2_str");
    return ( String(tempStr) );
}
