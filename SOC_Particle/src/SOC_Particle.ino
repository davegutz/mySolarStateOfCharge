/*
 * Project SOC_Photon

 * Description:
  * Monitor battery State of Charge (SOC) using Coulomb Counting (CC).  An experimental EKF is
  * used to estimate the SOC from voltage and temperature and to detect faults in the current
  * sensor.  The EKF is also used to reset the CC when it drifts too far from the EKF estimate.
  * The EKF is based on a simple battery model with a voltage source (VOC) and series resistance (Rss) that are functions of SOC and temperature.  The model parameters are stored in tables that can be generated from data or from a more complex model.  The EKF also estimates the hysteresis charge storage and diffusion effects that cause VOC to lag behind SOC changes.  The hysteresis model is used to improve the EKF performance and to detect faults in the current sensor by comparing the expected hysteresis voltage with the measured voltage.
  * To improve the accuracy of the current measurement, two current sensors are used: an amplified sensor for low currents and a non-amplified sensor for high currents.  The system automatically switches between the two sensors based on the current level and the sensor errors.  The system also includes various parameters to control the modeling and fault handling behavior, as well as support for BLE communication and a GUI for configuration and data visualization.

  * By:  Dave Gutz September 2021

  * 09-Aug-2021   Initial Git commit.   Unamplified ASD1013 12-bit shunt voltage sensor
  * 26-Dec-2021   Added 1 Hz anti-alias filters (AAF) in hardware to cleanup the 60 Hz
  * inverter noise on Vb and Ib.
  *               Add amplified (OPA333) current sensor ASD1013 with Texas Instruments (TI)
  * amplifier design in hardware
  *               First working prototype with iterative solver SOC-->Vb from polynomial
  * that have coefficients in tables
  *               Mark last good working version before class code.  EKF functional
  *               Put in class code for Monitor and Model
  * 31-Jan-2022   Vb model in tables.  Add battery heater in hardware
  * 18-May-2022   Manually tune for current sensor errors.   Vb model in tables
  *               Add Tweak methods to dynamically determine current sensor erros
  *               Bunch of cleanup and reorganization
  * 21-Sep-2022   Alpha release v20220917.  Branch GitHub repository.  Added signal redundancy checks and fault handling.
  * 22-Dec-2022   First Beta release v20221028.   Branch GitHub repository.  Various debugging fixes hysteresis.
  *               RetainedPars-->SavedPars to support Argon with 47L16 EERAM device
  *               Dual amplifier replaces dual ADS.  Beta release v20221220.  ADS still used on Photon.
  * 01-Dec-2023   g20231111 Photon 2, DS2482
  * 17-Apr-2024   g20230331 ib_charge = ib_ / ap.nS() while Randles uses ib_.  Tune Tb initialization
  *               Undo previous ib_/ap.nS() change
  * 02-Feb-2026   BLE and HI_LO ib selection
  * 13-Mar-2026   Add modeling and preserving parameters to control how much of the system is modeled
  * and how much is preserved in faults.  Use claude code to clean up and simplify code
  * 20-Apr-2026   Add support for GUI_Plink fully automated testing and configuration.  Add more parameters to support testing and configuration.  Add fake_faults parameter to allow testing of fault handling without actually causing faults.
  * ....see git log for more details
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
//
// See README.md
*/

#include "constants.h"
#undef ARDUINO
#if (PLATFORM_ID != PLATFORM_P2)
  #error "copy local_config.xxxx.h to constants.h"
#endif

// Dependent includes.   Easier to sp.debug code if remove unused include files
#include "Sync.h"
#include "subs.h"
#include "Summary.h"
#include "Cloud.h"
#include "debug.h"
#include "parameters.h"
#include "serial.h"
#include "ble.h"

//#define BOOT_CLEAN      // Use this to clear 'lockup' problems introduced during testing using Talk

// Turn on Log
#ifdef LOGHANDLE
  SerialLogHandler logHandler;
#endif

// ── extern declarations (defined in other .cpp files) ────────────────
extern SavedPars sp;              // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap;           // Various adjustment parameters shared at system level
extern CommandPars cp;            // Various parameters shared at system level
extern PrinterPars pr;            // Print buffer structure
extern PublishPars pp;            // For publishing
extern Flt_st mySum[NSUM];        // Summaries for saving charge history
extern BleCharacteristic txCharacteristic;  // Transmit to BLE
extern BleCharacteristic rxCharacteristic;  // Receive from BLE

// ── retained SRAM (survives power cycle) ─────────────────────────────
retained Flt_st saved_hist[NHIS];    // For displaying history
retained Flt_st saved_faults[NFLT];  // For displaying faults
retained SavedPars sp = SavedPars(saved_hist, uint16_t(NHIS), saved_faults, uint16_t(NFLT));  // Various parameters to be common at system level

// ── regular globals ───────────────────────────────────────────────────
Flt_st mySum[NSUM];                   // Summaries
PrinterPars pr = PrinterPars();       // Print buffer
VolatilePars ap = VolatilePars();     // Various adjustment parameters commanding at system level.  Initialized on start up.  Not retained.
CommandPars cp = CommandPars();       // Various control parameters commanding at system level.  Initialized on start up.  Not retained.
PublishPars pp = PublishPars();       // Common parameters for publishing.  Future-proof cloud monitoring
BleCharacteristic rxCharacteristic("rx", BleCharacteristicProperty::WRITE_WO_RSP, rxUuid, serviceUuid, onBLE_DataReceived, NULL);
BleCharacteristic txCharacteristic("tx", BleCharacteristicProperty::NOTIFY, txUuid, serviceUuid);
uint64_t millis_flip = millis(); // Timekeeping
uint64_t last_sync = millis();   // Timekeeping

int num_timeouts = 0;           // Number of Particle.connect() needed to unfreeze
String hm_string = "00:00";     // time, hh:mm
Pins *myPins;                   // Photon hardware pin mapping used

// Setup
void setup()
{
  setup_serial_ble();

  // Time
  sp.put_Time_now(max(sp.Time_now(), (uint32_t)Time.now()));  // Synch with web when possible
  Time.setTime( (time_t) (sp.Time_now()) );

  setup_pins();

  // Synchronize clock
  // Device needs to be configured for wifi (hold setup 3 sec run Particle app) and in range of wifi
  // Phone hotspot is very convenient
  delay(2000);
  WiFi.off();
  delay(1000);
  sendTxBuf("Done WiFi\n", true, IN_SERVICE);
  sendTxBuf("done CLOUD\n", true, IN_SERVICE);

  check_and_fix_corruption();

  // Determine millis() at turn of Time.now   Used to improve accuracy of timing.
  sync_time(millis(), &last_sync, &millis_flip);

  // Enable and print stored history
  System.enableFeature(FEATURE_RETAINED_MEMORY);
  if ( sp.debug()==1 || sp.debug()==2 || sp.debug()==3 || sp.debug()==4 )
  {
    sp.print_history_array();
    sp.print_fault_header(&pp.pubList);
  }
  sp.nsum(NSUM);  // Store

  // Ask to renominalize or force nominal.  Set in config file (see local_config.h for presently used config file)
  handle_boot_sequence();

  sendTxBuf("End setup()\n\n", true, IN_SERVICE);
} // setup


// Loop
void loop()
{
  // Synchronization
  static uint64_t now = (uint64_t) millis();
  now = (uint64_t) millis();
  bool chitchat = false;
  static Sync *Talk = new Sync(TALK_DELAY);
  #if IN_SERVICE
    static Sync *NoSaveWarn = new Sync(NO_SAVE_WARN);
  #endif
  bool read = false;
  static Sync *ReadSensors = new Sync(READ_DELAY);
  bool read_temp = false;
  static Sync *ReadTemp = new Sync(TEMP_DELAY);
  bool display_and_remember;
  static Sync *DisplayUserSync = new Sync(DISPLAY_USER_DELAY);
  bool summarizing;
  static bool boot_wait = true;  // waiting for a while before summarizing
  static Sync *Summarize = new Sync(SUMMARY_DELAY);
  uint64_t elapsed = 0;
  uint64_t elapsed_reset = 0;
  static bool reset = true;
  static bool reset_ekf = true;
  static bool reset_kf = true;
  static bool reset_temp = true;
  static uint64_t start = millis();
  static uint64_t start_reset = millis();

  // Monitor to count Coulombs and run EKF
  static BatteryMonitor *Mon = new BatteryMonitor(0., 0., sp.Dw());

  // Sensor conversions.  The embedded model 'Sim' is contained in Sensors
  uint64_t time_now = (uint64_t) Time.now();
  static Sensors *Sen = new Sensors(EKF_NOM_DT, 0, myPins, ReadSensors, ReadTemp, Talk, Summarize, time_now, start, Mon);

  // Battery saturation debounce
  static TFDelay *Is_sat_delay = new TFDelay(false, T_SAT, T_DESAT, EKF_NOM_DT);


  ///////////////////////////////////////////////////////////// Top of loop////////////////////////////////////////


  // Synchronize
  if ( now - last_sync > ONE_DAY_MILLIS || reset )  sync_time(now, &last_sync, &millis_flip);
  Sen->control_time(double(Sen->now())/1000.);
  char buffer[32];
  time_long_2_str(time_now, buffer);
  hm_string = String(buffer);
  read_temp = ReadTemp->update(now, reset);
  read = ReadSensors->update(now, reset);
  chitchat = Talk->update(now, reset);
  elapsed = ReadSensors->now() - start;
  elapsed_reset = ReadSensors->now() - start_reset;
  display_and_remember = DisplayUserSync->update(now, reset);
  bool boot_summ = boot_wait && ( elapsed >= SUMMARY_WAIT / (SUMMARY_DELAY / ap.sum_delay()) ) && !sp.modeling();
  if ( elapsed >= SUMMARY_WAIT / (SUMMARY_DELAY / ap.sum_delay()) ) boot_wait = false;
  summarizing = Summarize->update(now, false) || boot_summ;


  if ( read )
  {
    // Warn if parameters have been changed but not saved
    #if IN_SERVICE
      if ( NoSaveWarn->update(now, reset) && sp.dirty() )
      {
        sendTxBuf("WARNING: unsaved Retained parameter.  Enter 'w' to save.\n", true, IN_SERVICE);
      }
    #endif

    // Sample Ib
    if ( reset_kf ) sendTxBuf(" SOC_Particle:  reseting kfs\n", true, IN_SERVICE);
    Sen->ShuntAmp->sample(reset_kf);
    Sen->ShuntNoAmp->sample(reset_kf);

    Sen->reset(reset);

    // Check for really slow data capture and run EKF each read frame
    // ap.eframe_mult() = max(int(float(READ_DELAY)*float(EKF_EFRAME_MULT)/float(ReadSensors->delay())+0.9999), 1);

    update_publish_frame();

    // Read sensors, model signals, select between them, synthesize injection signals on current
    // Inputs:  sp.config, sp.sim_chm
    // Outputs: Sen->Ib, Sen->Vb, sp.inj_bias
    sense_synth_select(reset, reset_temp, reset_kf, ReadSensors->now(), elapsed, myPins, Mon, Sen);

    // Calculate Ah remaining
    // Inputs:  sp.mon_chm, Sen->Ib, Sen->Vb, Sen->Tb_f
    // States:  Mon.soc
    // Outputs: tcharge_wt, tcharge_ekf
    monitor(reset, reset_temp, reset_ekf, now, Is_sat_delay, Mon, Sen);

    // Re-init Coulomb Counter to EKF if it is different than EKF or if never saturated
    Mon->regauge(Sen->Tb_f(), Sen);

    // Empty battery
    if ( sp.modeling() && reset && Sen->Sim->q()<=0. ) Sen->Ib(0.);

    // Debug for read
    if ( sp.debug()==12 ) debug_12(Mon, Sen);

    // Publish for variable print rate
    if ( cp.publishS )
    {
      assign_publist(&pp.pubList, ReadSensors->now(), unit, hm_string, Sen, num_timeouts, Mon);
      static bool wrote_last_time = false;
      if ( wrote_last_time )
        digitalWrite(myPins->status_led, LOW);
      else
        digitalWrite(myPins->status_led, HIGH);
      wrote_last_time = !wrote_last_time;
    }

    // Print
    print_shunt_serial(reset, Sen);
    print_signal_sel_serial(reset, Sen, Mon, Sen->Sim);
    print_rapid_data(reset, Sen, Mon, reset_temp);

  }  // end read (high speed frame)


  // Bluetooth display drivers.   Also convenient update time for saving parameters (remember)
  if ( display_and_remember )
  {
    serial_display(Sen, Mon);
    sp.put_Time_now(max( sp.Time_now(), (uint32_t)Time.now()));  // If happen to connect to wifi (assume updated automatically), save new time
  }

  
  // Discuss things with the user
  // When open interactive serial monitor such as puTTY
  // then can enter commands by sending strings.   End the strings with a real carriage return
  // right in the "Send String" box then press "Send."
  // String definitions are below.

  // Chit-chat requires 'read' timing so 'DP' and 'Dr' can manage sequencing
  // Running chitter unframed allows queues of different priorities to be built from long
  // runs of Serial inputs
  if ( chitter(chitchat && !reset_temp, Mon, Sen) )  // Parse inputs to queues; returns true if any queue has work
  {
    chatter();  // Prioritize commands to describe.  ctl_str and asap_str queues always run.  Others only with chitchat
    describe(Mon, Sen);  // Run the commands
  }

  
  // Summary management.   Every boot after a wait an initial summary is saved in rotating buffer
  // Then every half-hour unless modeling.   Can also request manually via cp.write_summary (Talk)
  manage_summaries(boot_wait, summarizing, Mon, Sen);

  
  // Data capture
  sample_burst(myPins, Sen);

  
  // Initialize complete once sensors and models started and summary written
  if ( read )
  {
    reset = reset_ekf = reset_kf = cp.ekf_reset_print = cp.kf_reset_print = false;
    if ( reset_temp ) sendTxBuf("*", true, IN_SERVICE);
  }
  if ( read_temp && elapsed_reset>TEMP_DELAY && reset_temp )
  {
    sendTxBuf("...temp init complete\n", true, IN_SERVICE);
    reset_temp = false;
  }


  // Soft reset
  handle_soft_reset(&reset, &reset_temp, &reset_kf, &reset_ekf, &start_reset, read);

} // loop
