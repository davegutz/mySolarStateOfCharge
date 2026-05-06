/*
 * Project SOC_Photon
  * Description:
  * Monitor battery State of Charge (SOC) using Coulomb Counting (CC).  An experimental EKF is 
  * used to estimate the SOC from voltage and temperature and to detect faults in the current
  * sensor.  The EKF is also used to reset the CC when it drifts too far from the EKF estimate.
  * The EKF is based on a simple battery model with a voltage source (VOC) and series resistance (Rss) that are functions of SOC and temperature.  The model parameters are stored in tables that can be generated from data or from a more complex model.  The EKF also estimates the hysteresis charge storage and diffusion effects that cause VOC to lag behind SOC changes.  The hysteresis model is used to improve the EKF performance and to detect faults in the current sensor by comparing the expected hysteresis voltage with the measured voltage.
  * By:  Dave Gutz September 2021
  * 09-Aug-2021   Initial Git commit.   Unamplified ASD1013 12-bit shunt voltage sensor
  * ??-Sep-2021   Added 1 Hz anti-alias filters (AAF) in hardware to cleanup the 60 Hz
  * inverter noise on Vb and Ib.
  * 27-Oct-2021   Add amplified (OPA333) current sensor ASD1013 with Texas Instruments (TI)
  * amplifier design in hardware
  * 27-Aug-2021   First working prototype with iterative solver SOC-->Vb from polynomial
  * that have coefficients in tables
  * 22-Dec-2021   Mark last good working version before class code.  EKF functional
  * 26-Dec-2021   Put in class code for Monitor and Model
  * ??-Jan-2021   Vb model in tables.  Add battery heater in hardware
  * 03-Mar-2022   Manually tune for current sensor errors.   Vb model in tables
  * 21-Apr-2022   Add Tweak methods to dynamically determine current sensor erros
  * 18-May-2022   Bunch of cleanup and reorganization
  * 21-Sep-2022   Alpha release v20220917.  Branch GitHub repository.  Added signal redundancy checks and fault handling.
  * 26-Nov-2022   First Beta release v20221028.   Branch GitHub repository.  Various debugging fixes hysteresis.
  * 12-Dec-2022   RetainedPars-->SavedPars to support Argon with 47L16 EERAM device
  * 22-Dec-2022   Dual amplifier replaces dual ADS.  Beta release v20221220.  ADS still used on Photon.
  * 01-Dec-2023   g20231111 Photon 2, DS2482
  * 01-Apr-2024   g20230331 ib_charge = ib_ / ap.nS() while Randles uses ib_.  Tune Tb initialization
  * 17-Apr-2024   Undo previous ib_/ap.nS() change
  * ....see git log for more details
  * 02-Feb-2026   BLE and HI_LO ib selection
  * 13-Mar-2026   Add modeling and preserving parameters to control how much of the system is modeled
  * and how much is preserved in faults.  Use claude code to clean up and simplify code
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
// Prevent mixing up local_config files (still could sneak soc0p through as pro0p)
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

// Globals
extern SavedPars sp;              // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap;           // Various adjustment parameters shared at system level
extern CommandPars cp;            // Various parameters shared at system level
extern PrinterPars pr;            // Print buffer structure
extern PublishPars pp;            // For publishing
extern Flt_st mySum[NSUM];        // Summaries for saving charge history
extern BleCharacteristic txCharacteristic;  // Transmit to BLE
extern BleCharacteristic rxCharacteristic;  // Receive from BLE

retained Flt_st saved_hist[NHIS];    // For displaying history
retained Flt_st saved_faults[NFLT];  // For displaying faults
retained SavedPars sp = SavedPars(saved_hist, uint16_t(NHIS), saved_faults, uint16_t(NFLT));  // Various parameters to be common at system level

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
  // Log.info("begin setup");
  // Serial
  // Serial.blockOnOverrun(false);  doesn't work
  Serial.begin(SOFT_SBAUD);
  Serial.flush();
  delay(1000);          // Ensures a clean display
  sendTxBuf("Hi!\n", true, true);

  // BLE
	BLE.on();
  BLE.addCharacteristic(txCharacteristic);
  BLE.addCharacteristic(rxCharacteristic);
  BleAdvertisingData data;
  data.appendServiceUUID(serviceUuid);
  BLE.advertise(&data);

  // Time
  sp.put_Time_now(max(sp.Time_now(), (uint32_t)Time.now()));  // Synch with web when possible
  Time.setTime( (time_t) (sp.Time_now()) );

  // Peripherals (non-Photon2)
  // D6 - one-wire temp sensor
  // D7 - status led heartbeat
  // A1 - Vb
  // A2 - Primary Ib amp (called by old ADS name Amplified, amp)
  // A3 - Backup Ib amp (called by old ADS name Non Amplified, noa)
  // A4 - Vr or Vc

  // Peripherals (Photon2)
  // D3 - one-wire temp sensor ******** to be replaced by I2C device
  // D7 - status led heartbeat
  // A0 (pin 'D11') - Primary Ib amp (called by old ADS name Amplified, amp)
  // A1 (pin 'D12') - Vb
  // A2 (pin 'D13') - Backup Ib amp (called by old ADS name Non Amplified, noa)
  // A3 (pin 'D0') - alternate to SDA.  Sometimes used for 2wire temperature
  // A4 (pin 'D1') - alternate to SCL.
  // A5 (pin 'D14') - Vr or Vc

  // Log.info("setup Pins");
  myPins = new Pins(D3, D7, D12, D11, D13, D14, D0, true);
  pinMode(myPins->status_led, OUTPUT);
  digitalWrite(myPins->status_led, LOW);

  // 1-Wire chip card for I2C (after start Wire)
  #if defined(HDWE_BARE)
    sendTxBuf("Going naked\n", true, true);
  #elif defined(HDWE_2WIRE)
    sendTxBuf("Using 2Wire Temperature sensor\n", true, true);
  #else
    #error "Temperature sensor undefined"
  #endif

  // Synchronize clock
  // Device needs to be configured for wifi (hold setup 3 sec run Particle app) and in range of wifi
  // Phone hotspot is very convenientwait_on_user_input
  delay(2000);
  WiFi.off();
  delay(1000);
  sendTxBuf("Done WiFi\n", true, true);
  sendTxBuf("done CLOUD\n", true, true);

  // Clean boot logic.  This occurs only when doing a structural rebuild clean make on initial flash, because
  // the SRAM is not explicitly initialized.   This is by design, as SRAM must be remembered between boots
  // Time is never changed by this operation.  It could be corrupt.  Change using "UT" talk feature.
  sendTxBuf("Check corruption......", true, true);
  bool corrupt = sp.is_corrupt();
  if ( corrupt )
  {
    sendTxBuf("\n\n", true, true);
    sp.pretty_print( false );
    sendTxBuf("\n\n", true, true);
    sp.set_nominal();
    sendTxBuf("Fixed corruption\n", true, true);
    sp.pretty_print(true);
  }
  else sendTxBuf("\nclean\n", true, true);

  // Determine millis() at turn of Time.now   Used to improve accuracy of timing.
  long time_begin = Time.now();
  uint16_t count = 0;
  while ( Time.now()==time_begin && count++<1000 )
  {
    delay(1);
    millis_flip = millis()%1000;
  }

  // Enable and print stored history
  System.enableFeature(FEATURE_RETAINED_MEMORY);
  if ( sp.debug()==1 || sp.debug()==2 || sp.debug()==3 || sp.debug()==4 )
  {
    sp.print_history_array();
    sp.print_fault_header(&pp.pubList);
  }
  sp.nsum(NSUM);  // Store

  // Ask to renominalize or force nominal.  Set in config file (see local_config.h for presesntly used config file)
  sp.get_booted();  // get the stored booted state.  This is a hack to ensure that we don't have to wait for the normal backup on reset to occur.
  sendTxBuf(String::format("booted = %d\n", sp.booted()), true, true);
  if ( ASK_DURING_BOOT == 0 && !sp.booted() )  // automatically renominalize and reboot after a dirty boot.
  {
    sp.set_nominal();  // sets booted to false by the way
    sp.put_booted(true);  // sets booted to true so on next startups we don't have to renominalize to clean a dirty boot.
    sendTxBuf("\n\nSet booted true and stored...", true, true);
    System.backupRamSync();  // Force backup of RAM to ensure booted = true is saved.  This is important because the system reset below is a no-wait reset that doesn't wait for the normal backup on reset to occur.
    delay(1000);
    sendTxBuf("backup Ram synced *\n", true, true);
    sp.get_booted();  // get the stored booted state.  This is a hack to ensure that we don't have to wait for the normal backup on reset to occur.
    sendTxBuf(String::format("booted = %d\n", sp.booted()), true, true);
    sendTxBuf("booted should be true\n\n", true, true);
    delay(1000);          // Ensures true saves before rebooting
  }
  
  if ( ASK_DURING_BOOT == 1 )
  {
    // Log.info("setup renominalize");
    if ( sp.num_diffs() )
    {
      wait_on_user_input();
    }
  }

  // Log.info("setup end");
  sendTxBuf("End setup()\n\n", true, true);
} // setup


// Loop
void loop()
{
  // Synchronization
  static uint64_t now = (uint64_t) millis();
  now = (uint64_t) millis();
  bool chitchat = false;
  static Sync *Talk = new Sync(TALK_DELAY);
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
  static bool reset_publish = true;
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

  // Sample temperature
  // Outputs:   Sen->Tb,  Sen->Tb_f
  if ( read_temp )
  {
    Sen->T_temp(ReadTemp->updateTime());
    if ( reset_temp )
    {
      if ( sp.mod_tb() )
      {
        Sen->Tb_model(NOMINAL_TB + ap.Tb_bias_model());
        Sen->Tb_model_filt(NOMINAL_TB + ap.Tb_bias_model());
        Sen->Tbx_model(NOMINAL_TB + ap.Tb_bias_model());
        Sen->Tbx_model_f(NOMINAL_TB + ap.Tb_bias_model());
      }
      else
      {
        Sen->Tb_model(Sen->Tb());
        Sen->Tb_model_filt(Sen->Tb_f());
        Sen->Tbx_model(Sen->Tbx());
        Sen->Tbx_model_f(Sen->Tbx_f());
      }

      if ( sp.debug()==16 ) sendTxBuf(String::format("SOC_Particle.ino ln 336 reset_temp:  Sen->Tb_model, Sen->Tb_model_filt,  %11.8f %11.8f\n",
        Sen->Tb_model(), Sen->Tb_model_filt()), true, true);
      if ( sp.debug()==16 ) sendTxBuf(String::format("SOC_Particle.ino ln 336 reset_temp:  Sen->Tbx_model, Sen->Tbx_model_f, %11.8f %11.8f \n",
        Sen->Tbx_model(), Sen->Tbx_model_f()), true, true);
    }
    // Log.info("ino:  temp_load_and_filter");
    
    Sen->temp_load_and_filter_and_select(Mon, reset_temp);

    if ( sp.debug()==16 ) sendTxBuf(String::format("SOC_Particle.ino ln 324 final: reset_temp Sen->Sim->tb_f Sen->Tb_model, Sen->Tb_model_filt, Sen->Tb_hdwe_filt_rate, %d %11.8f %11.8f %11.8f  %11.8f\n",
        reset_temp, Sen->Sim->tb_f(), Sen->Tb_model(), Sen->Tb_model_filt(), Sen->Tb_hdwe_filt_rate()), true, true);
    if ( sp.debug()==16 ) sendTxBuf(String::format("SOC_Particle.ino ln 326 final: reset_temp Sen->Sim->tbx_f Sen->Tbx_model, Sen->Tbx_model_f, Sen->Tbx_model, Sen->Tbx_model_f, %d %11.8f %11.8f %11.8f  %11.8f\n",
        reset_temp, Sen->Sim->Tbx_f(), Sen->Tbx_model(), Sen->Tbx_model_f(), Sen->Tbx_model(), Sen->Tbx_model_f()), true, true);
    // Log.info("ino:  print_temp_serial");
    print_temp_serial(reset_temp, Mon, Sen);
  }

  // Sample Ib
  if ( read )
  {
    // Log.info("Read shunt");
    if ( reset_kf )sendTxBuf(" SOC_Particle:  reseting kfs\n", true, true);
    Sen->ShuntAmp->sample(reset_kf);
    // Log.info("ino:  Shunt::sample_time,%lld,cTime,%7.3f,", Sen->ShuntAmp->sample_time(), double(Sen->ShuntAmp->sample_time() - Sen->inst_millis() + Sen->inst_time()*1000)/1000.f);
    Sen->ShuntNoAmp->sample(reset_kf);
  }

  // Input all other sensors and do high rate calculations
  if ( read )
  {
    // Log.info("ino:  read");
    Sen->reset(reset);

    // Check for really slow data capture and run EKF each read frame
    // ap.eframe_mult() = max(int(float(READ_DELAY)*float(EKF_EFRAME_MULT)/float(ReadSensors->delay())+0.9999), 1);

    // Set print frame
    static uint8_t print_count = 0;
    if ( print_count>=ap.print_mult()-1 || print_count==UINT8_MAX )  // > avoids lockup on change by user
    {
      print_count = 0;
      cp.publishS = true;
    }
    else
    {
      print_count++;
      cp.publishS = false;
    }

    // Read sensors, model signals, select between them, synthesize injection signals on current
    // Inputs:  sp.config, sp.sim_chm
    // Outputs: Sen->Ib, Sen->Vb, sp.inj_bias
    // Log.info("ino:  sense_synth_select");
    sense_synth_select(reset, reset_temp, reset_kf, ReadSensors->now(), elapsed, myPins, Mon, Sen);

    // Calculate Ah remaining`
    // Inputs:  sp.mon_chm, Sen->Ib, Sen->Vb, Sen->Tb_f
    // States:  Mon.soc
    // Outputs: tcharge_wt, tcharge_ekf
    monitor(reset, reset_temp, reset_ekf, now, Is_sat_delay, Mon, Sen);

    // Re-init Coulomb Counter to EKF if it is different than EKF or if never saturated
    Mon->regauge(Sen->Tb_f());

    // Empty battery
    if ( sp.modeling() && reset && Sen->Sim->q()<=0. ) Sen->Ib(0.);

    // Debug for read
    if ( sp.debug()==12 ) debug_12(Mon, Sen);

    // Publish for variable print rate
    if ( cp.publishS )
    {
      // Log.info("ino:  assign_publist ReadSensors->now()=%lld", ReadSensors->now());
      assign_publist(&pp.pubList, ReadSensors->now(), unit, hm_string, Sen, num_timeouts, Mon);
      static bool wrote_last_time = false;
      if ( wrote_last_time )
        digitalWrite(myPins->status_led, LOW);
      else
        digitalWrite(myPins->status_led, HIGH);
      wrote_last_time = !wrote_last_time;
    }

    // Print
    // Log.info("ino:  print_rapid_data");
    print_shunt_serial(reset, Sen);
    print_signal_sel_serial(reset, Sen, Mon, Sen->Sim);
    print_rapid_data(reset, Sen, Mon, reset_temp);

    // Log.info("end read");
  }  // end read (high speed frame)

  // Bluetooth display drivers.   Also convenient update time for saving parameters (remember)
  if ( display_and_remember )
  {
    // Log.info("display and remember");
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
  if ( (!boot_wait && summarizing) || cp.write_summary )
  {
    sp.put_Ihis(sp.ihis() + 1);
    if ( sp.ihis() > (sp.nhis() - 1) ) sp.put_Ihis(0);  // wrap buffer
    Flt_st hist_snap, hist_bounced;
    hist_snap.assign(Sen->now(), Mon, Sen);
    hist_bounced = sp.put_history(hist_snap, sp.ihis());

    sp.put_Isum(sp.isum() + 1);
    if ( sp.isum() > (uint16_t)(sp.nsum()-1) ) sp.put_Isum(0);  // wrap buffer
    mySum[sp.isum()].copy_to_Flt_ram_from(hist_bounced);
    sendTxBuf("Summ...\n", true, true);
    cp.write_summary = false;
  }

  // Data capture
  sample_burst(myPins, Sen);

  // Initialize complete once sensors and models started and summary written
  if ( read )
  {
    reset = reset_ekf = reset_kf = cp.ekf_reset_print = cp.kf_reset_print = false;
    if ( reset_temp && !Sen->tb_fa_one_shot() ) sendTxBuf("*", true, true);
  }
  if ( read_temp && elapsed_reset>ap.temp_delay() && reset_temp )
  {
    sendTxBuf("...temp init complete\n", true, true);
    reset_temp = false;
  }

  if ( cp.publishS ) reset_publish = false;

  // Soft reset
  if ( read ) cp.soft_sim_hold = false;
  cp.soft_reset_print = cp.soft_reset;
  cp.soft_reset_sim_print = cp.soft_reset_sim;
  if ( cp.soft_reset || cp.soft_reset_sim )
  {
    reset = reset_temp = reset_kf = reset_publish = true;
    start_reset = millis();
    if ( cp.soft_reset_sim ) cp.cmd_soft_sim_hold();
  }
  if ( cp.ekf_reset ) cp.ekf_reset_print = reset_ekf = true;
  if ( cp.kf_reset ) cp.kf_reset_print = reset_kf = true;
  cp.soft_reset = cp.soft_reset_sim = cp.ekf_reset = cp.kf_reset = false;
  // Log.info("ino:  end loop\n\n\n");

} // loop

