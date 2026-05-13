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

#include "Cloud.h"
#include "constants.h"
#include "Variable.h"


// Definition of structure for external control coordination
struct PublishPars
{
  Publish pubList;          // Publish object
  PublishPars(void)
  {
    pubList = Publish();
  }
};


class CommandPars
{
public:
  ~CommandPars();

  // Small static value area for 'retained'
  String ctl_str;           // Hold control queue
  String cmd_str;           // Hold final cmd data queue
  String inp_str;           // Hold incoming data queue
  String last_str;           // Hold chit_chat end data - after everything else, 1 per Control pass
  String queue_str;         // Hold chit_chat queue data - queue with Control pass, 1 per Control pass
  String soon_str;          // Hold chit_chat soon data - priority with next Control pass, 1 per Control pass
  String asap_str;          // Hold chit_chat asap data - no waiting, ASAP all of now_str processed before Control pass
  bool freeze;           // Stop applying (describe()) the queues
  bool inp_token;        // Whether inp_str is complete
  bool cmd_token;        // Whether cmd_str has been applied
  bool chitchat;         // Outer frame call, used in chitchat functions
  bool inf_reset;        // Use talk to reset infinite counter
  bool model_cutback;    // On model cutback
  bool model_saturated;  // Sim on cutback and saturated
  uint32_t num_v_print;// Number of print echos made, for checking on BLE
  bool publishS;         // Print serial monitor data
  bool soft_reset;       // Use talk to reset all
  bool soft_reset_print;       // Use talk to reset all
  bool soft_reset_sim;   // Use talk to reset sim only
  bool soft_reset_sim_print;   // Use talk to reset sim only
  bool soft_sim_hold;    // Use talk to reset sim only
  double ts;     // Used to scale time delays in fault logic when running very slow verification testing, slr
  bool write_summary;    // Use talk to issue a write command to summary
  bool ekf_reset;        // Reset Extended Kalman Filter
  bool ekf_reset_print;  // Reset Extended Kalman Filter status saved for printing
  bool kf_reset;         // Reset kalman filters
  bool kf_reset_print;   // Reset kalman filters status saved for printing
  uint8_t ble_first_char;   // Control boot communication, psuedo token
  uint32_t disp_word;       // Display status bitmapped word (see dispw:: enum)

  CommandPars()
  {
    inf_reset = false;
    model_cutback = false;
    model_saturated = false;
    num_v_print = 0UL;
    publishS = false;
    soft_reset = false;
    soft_reset_print = false;
    soft_reset_sim = false;
    soft_reset_sim_print = false;
    soft_sim_hold = false;
    write_summary = false;
    ts = 1.;
    chitchat = false;
    inp_token = false;
    freeze = false;
    ctl_str = "";
    inp_str = "";
    cmd_str = "";
    last_str = "";
    queue_str = "";
    soon_str = "";
    asap_str = "";
    ekf_reset = false;
    ekf_reset_print = false;
    kf_reset = false;
    kf_reset_print = false;
    ble_first_char = '\0';
    disp_word = 0UL;
  }

  void clear_disp_word() { disp_word = 0; };

  void cmd_reset(void) { soft_reset = true; ekf_reset = true; kf_reset = true; }

  void cmd_reset_ekf(void) { ekf_reset = true; }

  void cmd_reset_kf(void) { kf_reset = true; }

  void cmd_reset_sim(void) { soft_reset_sim = true; }

  void cmd_soft_sim_hold(void) { soft_sim_hold = true; }

  void cmd_summarize(void) { write_summary = true; }

  void large_reset(void)
  {
    model_cutback = true;
    model_saturated = true;
    soft_reset = true;
    ekf_reset = true;
    kf_reset = true;
    num_v_print = 0UL;
    ts = 1.;
  }

  void pretty_print(void)
  {
    #ifndef SOFT_DEPLOY_PHOTON

      //   text    data     bss     dec     hex filename
      // 290338  119852   13698  423888   677d0 c:/Users/daveg/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/target/6.
      sendTxBuf(String::format("command parameters(cp):\n"), true, IN_SERVICE);
      sendTxBuf(String::format(" inf_reset %d\n", inf_reset), true, IN_SERVICE);
      sendTxBuf(String::format(" model_cutback %d\n", model_cutback), true, IN_SERVICE);
      sendTxBuf(String::format(" model_saturated %d\n", model_saturated), true, IN_SERVICE);
      sendTxBuf(String::format(" publishS %d\n", publishS), true, IN_SERVICE);
      sendTxBuf(String::format(" soft_reset %d\n", soft_reset), true, IN_SERVICE);
      sendTxBuf(String::format(" soft_reset_sim %d\n", soft_reset_sim), true, IN_SERVICE);
      sendTxBuf(String::format(" ts %7.3f\n", ts), true, IN_SERVICE);
      sendTxBuf(String::format(" write_summary %d\n", write_summary), true, IN_SERVICE);
      sendTxBuf(String::format(" kf_reset %d\n", kf_reset), true, IN_SERVICE);
      sendTxBuf(String::format(" ekf_reset %d\n", ekf_reset), true, IN_SERVICE);
      sendTxBuf(String::format(" disp_word %d\n\n", disp_word), true, IN_SERVICE);

    #endif
  }

};
