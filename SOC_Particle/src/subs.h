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

#include "application.h"
#include "myLibrary/myFilters.h"
#include "Battery.h"
#include "constants.h"
#include "Cloud.h"
#include "talk/chitchat.h"
#include "parameters.h"
#include "command.h"
#include "Sync.h"
#include "FaultStore.h"

// Sensors
#include "Sensors.h"
#include "serial.h"

extern SavedPars sp;         // Various parameters to be static at system level and saved through power cycle
extern PublishPars pp;       // For publishing
extern CommandPars cp;       // Various parameters to be static at system level
extern Pins *myPins;         // Hardware pin mapping
extern Flt_st mySum[NSUM];  // Summaries


// Pins
struct Pins
{
  uint16_t status_led;  // On-board led
  uint16_t Vb_pin;      // Battery voltage, e.g. Battleborn, CHINS
  uint16_t Vcn_pin;     // No Amp (n) common voltage
  uint16_t Von_pin;     // No Amp (n) output voltage
  uint16_t Vcm_pin;     // Amp (m) common voltage
  uint16_t Vom_pin;     // Amp (m) output voltage
  uint16_t Vh3v3_pin;   // 3.3V voltage
  uint16_t VTb_pin;     // Tb 2wire measurement voltage
  bool using_opAmp;     // Using differential hardware amp
  bool using_hv3v3;  // Using differential hardware amp
  bool using_VTb;    // Using I2C port for 2wire temperature measurement (RTD)
  Pins(void) {}
  Pins(uint16_t status_led, uint16_t Vb_pin, uint16_t Vcn_pin, uint16_t Von_pin, uint16_t Vcm_pin, uint16_t Vom_pin)
  {
    this->status_led = status_led;
    this->Vb_pin = Vb_pin;
    this->Vcn_pin = Vcn_pin;
    this->Von_pin = Von_pin;
    this->Vcm_pin = Vcm_pin;
    this->Vom_pin = Vom_pin;
    this->using_opAmp = false;
    this->using_hv3v3 = false;
  }
  Pins(uint16_t status_led, uint16_t Vb_pin, uint16_t Von_pin, uint16_t Vom_pin)
  {
    this->status_led = status_led;
    this->Vb_pin = Vb_pin;
    this->Von_pin = Von_pin;
    this->Vom_pin = Vom_pin;
    this->using_opAmp = true;
    this->using_hv3v3 = false;
  }
  Pins(uint16_t status_led, uint16_t Vb_pin, uint16_t Von_pin, uint16_t Vom_pin, uint16_t Vh3v3_pin)
  {
    this->status_led = status_led;
    this->Vb_pin = Vb_pin;
    this->Von_pin = Von_pin;
    this->Vom_pin = Vom_pin;
    this->Vh3v3_pin = Vh3v3_pin;
    this->using_opAmp = true;
    this->using_hv3v3 = true;
  }
  Pins(uint16_t status_led, uint16_t Vb_pin, uint16_t Von_pin, uint16_t Vom_pin, uint16_t Vh3v3_pin, uint16_t VTb_pin, bool using_2wire)
  {
    this->status_led = status_led;
    this->Vb_pin = Vb_pin;
    this->Von_pin = Von_pin;
    this->Vom_pin = Vom_pin;
    this->Vh3v3_pin = Vh3v3_pin;
    this->using_opAmp = true;
    this->using_hv3v3 = true;
    this->VTb_pin = VTb_pin;
    this->using_VTb = using_2wire;
  }
};


// Headers
void check_and_fix_corruption();
void handle_boot_sequence();
void handle_soft_reset(bool *reset, bool *reset_temp, bool *reset_kf, bool *reset_ekf, uint64_t *start_reset, const bool read);
void harvest_temp_change(const double tb_f, BatteryMonitor *Mon, BatterySim *Sim, const float rate, const float dt);
void initialize_all(BatteryMonitor *Mon, Sensors *Sen, const float soc_in, const bool use_soc_in);
void load_ib_vb_tb(const bool reset, const bool reset_temp, const bool reset_kf, Sensors *Sen, Pins *myPins, BatteryMonitor *Mon);
void manage_summaries(const bool boot_wait, const bool summarizing, BatteryMonitor *Mon, Sensors *Sen);
void monitor(const bool reset, const bool reset_temp, const bool reset_ekf, const uint64_t now,
  TFDelay *Is_sat_delay, BatteryMonitor *Mon, Sensors *Sen);
void sample_burst(Pins *myPins, Sensors *SenS);
void sense_synth_select(const bool reset, const bool reset_temp, const bool reset_kf, const uint64_t now,
  const uint64_t elapsed, Pins *myPins, BatteryMonitor *Mon, Sensors *Sen);
void serial_display(Sensors *Sen, BatteryMonitor *Mon);
void setup_pins();
void setup_serial_ble();
void sync_time(uint64_t now, uint64_t *last_sync, uint64_t *millis_flip);
String time_long_2_str(const time_t current_time, char *tempStr);
void update_publish_frame();
