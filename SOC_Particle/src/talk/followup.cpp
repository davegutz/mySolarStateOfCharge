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
#include "../subs.h"
#include "../command.h"
#include "../parameters.h"
#include "../debug.h"

extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level
extern CommandPars cp;  // Various parameters shared at system level
extern Flt_st mySum[NSUM];  // Summaries for saving charge history

bool followup(const char letter_0, const char letter_1, BatteryMonitor *Mon, Sensors *Sen, uint16_t modeling_past)
{
    bool reset = false;
    bool found = true;
    char buffer[32];
    switch ( letter_0 )
    {

        case ( 'C' ):  // C:
            switch ( letter_1 )
            {

                case ( 'a' ):  // Ca<>:  assign charge state in fraction to all versions including model
                    if ( ap.init_all_soc_p->success() )
                    {
                        initialize_all(Mon, Sen, ap.init_all_soc(), true);
                        #ifdef DEBUG_INIT
                        if ( sp.debug()==-1 ){ Serial.printf("after initialize_all:"); debug_m1(Mon, Sen);}
                        #endif
                        if ( sp.modeling() )
                        {
                        cp.cmd_reset();
                        chit("W3;", SOON);  // Wait 10 passes of Control
                        }
                        else
                        {
                        cp.cmd_reset();
                        chit("W3;", SOON);  // Wait 10 passes of Control
                        }
                    }
                    else
                        Serial.printf("skipping %s\n", cp.cmd_str.c_str());
                    break;

                case ( 'm' ):  // Cm<>:  assign curve charge state in fraction to model only (ekf if modeling)
                    if ( ap.init_sim_soc_p->success() )  // Apply crude limit to prevent user error
                    {
                        Sen->Sim->apply_soc(ap.init_sim_soc(), Sen->Tb_f());
                        Serial.printf("soc%8.4f, dq%7.3f, soc_mod%8.4f, dq mod%7.3f,\n",
                            Mon->soc(), Mon->delta_q(), Sen->Sim->soc(), Sen->Sim->delta_q());
                        if ( sp.modeling() ) cp.cmd_reset_sim(); // Does not block.  Commands a reset
                    }
                    else
                        Serial.printf("soc%8.4f; must be 0-1.1\n", ap.init_sim_soc());
                    break;
            }
            break;

        case ( 'D' ):
            switch ( letter_1 )
            {
                case ( 'h' ):  //   Dh<>:  Summary sample time input  Not sure this section needed since nominalizing capability added 11/2025
                    if ( ap.sum_delay_p->success() )
                        Sen->Summarize->delay(max(ap.read_delay(), ap.sum_delay()), Sen->now());  // validated
                    else if (ap.value_str()=="0" || ap.value_str()=="")
                    {
                        Serial.printf("setting NOMINAL instead\n");
                        ap.sum_delay_p->set_nominal();
                        Sen->Summarize->delay(max(ap.read_delay(), ap.sum_delay()), Sen->now());
                    }
                    break;

                case ( 'r' ):  //   Dr<>:  READ sample time input
                    if ( ap.read_delay_p->success() )
                    {
                        Sen->ReadSensors->delay(ap.read_delay());  // validated
                        Sen->Summarize->delay(max(ap.read_delay(), ap.sum_delay()));  // validated
                        cp.ts = ap.read_delay() / NOM_READ_DELAY_S / 1000.;
                    }
                    break;

                case ( 's' ):  //   Ds<>:  Battery Sim dx_voc bias
                    if ( ap.ds_voc_soc_p->success() )
                    {
                        Sen->Sim->put_dx_voc(ap.ds_voc_soc());  // validated
                    }
                    break;

                case ( 't' ):  //*  Dt<>:  Temp bias change hardware
                    if ( sp.Tb_bias_hdwe_p->success() &&  sp.modeling()!=230 )  // modeling==230 is special regression test
                        cp.cmd_reset();
                    break;

                case ( 'v' ):  //     Dv<>:  voltage signal adder for faults
                    if ( ap.vb_add_p->success() )
                        ap.vb_add_p->print();
                    break;

                case ( 'w' ):  //   Dw<>:  Battery Monitor dz_voc bias
                    if ( sp.Dw_p->success() )
                    {
                        Mon->put_dz_voc(sp.Dw());  // validated
                    }
                    break;

                case ( '>' ):  //   D><>:  TALK sample time input
                    if ( ap.talk_delay_p->success() )
                        Sen->Talk->delay(ap.talk_delay());  // validated
                    break;
            }
            break;

        case ( 'F' ):  // Fault stuff
            switch ( letter_1 )
            {

                case ( 'f' ):  //* si, Ff<>:  fake faults
                if ( ap.fake_faults_p->success() )
                {
                    Sen->Flt->reset_all_faults();
                    sp.put_ib_force(cp.cmd_str.substring(2).toInt());
                }
                break;
            }
            break;

        case ( 'K' ):  // Kalman filter stuff
            switch ( letter_1 )
            {

                case ( 'q' ):  // Kq
                if ( ap.q_std_p->success() )
                {
                    Sen->ShuntAmp->kf_q_std(ap.q_std());
                    Sen->ShuntAmp->kf_r_std(ap.r_std());
                    Sen->ShuntNoAmp->kf_q_std(ap.q_std());
                    Sen->ShuntNoAmp->kf_r_std(ap.r_std());
                }
                break;
            }
            break;

        case ( 'l' ):
            if ( sp.debug_p->success() )
            switch ( sp.debug() )
            {
                case ( -1 ):  // l-1:
                // Serial.printf("SOCu_s-90  ,SOCu_fa-90  ,Ishunt_amp  ,Ishunt_noa  ,Vb_fo*10-110  ,voc_s*10-110  ,dv_dyn_s*10  ,v_s*10-110  , voc_dyn*10-110,,,,,,,,,,,\n");
                break;
                case ( 1 ):  // l1:
                print_rapid_header();
                break;
                case ( 2 ):  // l2:
                print_signal_sel_header();
                print_sim_header();
                print_shunt_header(Sen);
                print_rapid_header();
                break;
                case ( 3 ):  // l3:
                print_ekf_header();
                print_sim_header();
                print_rapid_header();
                break;
                default:
                print_rapid_header();
            }
            break;

        case ( 'S' ):
            switch ( letter_1 )
            {

                case ( 'H' ):  //   SH<>: state of all hysteresis
                    if ( ap.hys_state_p->success() )
                    {
                        Sen->Sim->hys_state(ap.hys_state());
                        Sen->Flt->wrap_err_filt_state(0.);
                    }
                    break;

                case ( 'q' ):  //   Sq<>: scale capacity sim
                    if ( ap.s_cap_sim_p->success() )
                    {
                        Sen->Sim->apply_cap_scale(ap.s_cap_sim());
                        if ( sp.modeling() ) Mon->init_soc_ekf(Sen->Sim->soc());
                    }
                    break;
            
                case ( 'Q' ):  //   SQ<>: scale capacity mon
                    if ( ap.s_cap_mon_p->success() )
                    {
                        Mon->apply_cap_scale(ap.s_cap_mon());
                    }
                    break;      
            }
            break;

        case ( 's' ):
            switch ( letter_1 )
            {

                case ( 'i' ):  //* si, force ib selection
                    Sen->Flt->reset_all_faults();
                    sp.put_ib_force(cp.cmd_str.substring(2).toInt());
                    break;

            }
            break;

        case ( 'U' ):
            if ( sp.Time_now_p->success() )
            switch ( letter_1 )
            {

                case ( 'T' ):  //*  UT<>:  Unix time since epoch
                  Time.setTime( (time_t) (sp.Time_now()) );
                  time_long_2_str((time_t)sp.Time_now(), buffer);
                  sendTxBuf(String::format(" time %ld hms:  %s\n", sp.Time_now(), buffer), true, IN_SERVICE);
                break;

            }
            break;

        case ( 'X' ):  // X
            switch ( letter_1 )
            {

                case ( 'm' ):  // Xm<>:  code for modeling level
                    if ( sp.modeling_p->success() )
                    {
                        reset = (sp.modeling() != modeling_past) && sp.debug()!=99;
                        if ( reset )
                        {
                            Serial.printf("Xm reset\n");
                            cp.cmd_reset();
                        }
                    }
                    break;

                case ( 'a' ): // Xa<>:  injection amplitude
                    if ( sp.amp_p->success() )
                        Serial.printf("Inj amp, %s, %s set%7.3f & inj_bias set%7.3f\n", sp.amp_p->units(), sp.amp_p->description(), sp.Amp(ap.nP()), sp.inj_bias());
                    break;

                case ( 'f' ): //*  Xf<>:  injection frequency
                    if ( sp.freq_p->success() ) sp.freq(sp.freq()*(2. * PI));
                    break;

                case ( 'b' ): //*  Xb<>:  injection bias
                    if ( sp.inj_bias_p->success() )
                        Serial.printf("Inj amp, %s, %s set%7.3f & inj_bias set%7.3f\n", sp.amp_p->units(), sp.amp_p->description(), sp.Amp(ap.nP()), sp.inj_bias());
                    break;

                case ( 'Q' ): //  XQ<>: time until quiet
                    if ( ap.until_q_p->success() )
                        Serial.printf("Going black for %7.1f seconds\n", float(ap.until_q()) / 1000.);
                        Serial.printf("Freezing queues.  When using 'XQ' unfreeze with 'cc'\n");
                        cp.freeze = true;
                    break;
            }
            break;

            default:
                // Serial.print( letter_0 ); Serial.print(" ? 'h'\n");
                break;
    }
    return found;
}
