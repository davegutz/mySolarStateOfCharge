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

#include "myLibrary/myFilters.h"
#include "Battery.h"
#include "constants.h"
#include "Cloud.h"
#include "talk/chitchat.h"
#include "parameters.h"
#include "command.h"
#include "Sync.h"

// Sensors
#include "Sensors.h"

const size_t UART_TX_BUF_SIZE = 20;

extern SavedPars sp;    // Various parameters to be static at system level and saved through power cycle
extern PublishPars pp;  // For publishing
extern CommandPars cp;  // Various parameters to be static at system level

// Headers
String chat_cmd_from(String *source);
void delay_no_block(const uint64_t delay_millis);
String finish_request(const String in_str);
bool is_finished(const char in_char);
void print_battery_header();
void print_battery_serial();
void print_all_header(Sensors *Sen);
void print_rapid_data(const bool reset, Sensors *Sen, BatteryMonitor *Mon, const bool reset_temp);
void print_rapid_header(void);
void print_rapid_serial(const bool reset, Publish *pubList, Sensors *Sen, BatteryMonitor *Mon);
void print_sim_serial(const bool initializing_all, const bool reset_temp, Sensors *Sen, BatterySim *Sim);
void print_sim_header(void);
void print_shunt_header(Sensors *Sen);
void print_shunt_serial(const bool reset, Sensors *Sen);
void print_signal_sel_serial(const bool reset, Sensors *Sen, BatteryMonitor *Mon, BatterySim *Sim);
void print_signal_sel_header(void);
void print_ekf_header(void);
void sendTxBuf(const String& txBuf, const bool sendSerial, const bool sendBLE);
void sendTxBuf(const char* txBuf, const bool sendSerial, const bool sendBLE);
void wait_on_user_input();
