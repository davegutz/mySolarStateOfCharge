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

#include "Battery.h"
#include "./hardware/SerialRAM.h"

String time_long_2_str(const time_t current_time, char *tempStr);

// SRAM retention summary
struct Flt_st
{
  uint64_t t_flt = 1ULL; // Timestamp ms since start of epoch
  int16_t Tb_hdwe_filt = 0;  // Battery bank temperature, hardware, C
  int16_t vb_hdwe_filt = 0;  // Battery single unit measured potential, hardware, V
  int16_t Vc_hdwe_sum = 0;  // Common voltage used Ib sensing, hardware, V
  int16_t ib_amp_hdwe_filt = 0;  // Battery single unit measured input current, amp, A
  int16_t ib_noa_hdwe_filt = 0;  // Battery single unit measured input current, no amp, A
  int16_t Tb_filt = 0;       // Battery bank temperature, filtered, C
  int16_t vb_filt = 0;       // Battery single unit measured potential, filtered, V
  int16_t ib_filt = 0;       // Battery single unit measured input current, filtered, A
  int16_t soc = 0;      // Battery bank state of charge, free Coulomb counting algorithm, frac
  int16_t soc_min = 0;  // Battery bank min state of charge, frac
  int16_t soc_ekf = 0;  // Battery bank state of charge, ekf, frac
  int16_t voc_filt = 0;      // Battery single unit open circuit voltage measured vb-ib*Z, V
  int16_t voc_stat_filt = 0; // Stored single unit charge voltage from measurement, V
  int16_t e_wrap_filt = 0;  // Wrap model error, filtered, V
  int16_t e_wrap_m_filt = 0;// Wrap amp model error, filtered, V
  int16_t e_wrap_m_trim = 0;// Wrap amp model trim, filtered, V
  int16_t e_wrap_n_filt = 0;// Wrap noa model error, filtered, V
  uint32_t fltw = 0;    // Fault word
  uint32_t falw = 0;    // Fail word
  uint32_t dummy = 0;  // padding to absorb Wire.write corruption
  void assign(const uint64_t now, BatteryMonitor *Mon, Sensors *Sen);
  void assign_unfilt(const uint64_t now, BatteryMonitor *Mon, Sensors *Sen);
  void copy_to_Flt_ram_from(Flt_st input);
  void get() {};
  void nominal();
  void pretty_print(const String code);
  void print_flt(const String code);
  void put(Flt_st source);
  void put_nominal();
};

class Flt_ram : public Flt_st
{
public:
  Flt_ram();
  ~Flt_ram();
  void get();
  void put(const Flt_st input);
  void put_nominal();

  void put_t_flt(const uint64_t value) { t_flt = value; };
  void put_Tb_hdwe_filt(const int16_t value)         { Tb_hdwe_filt = value; };
  void put_vb_hdwe_filt(const int16_t value)         { vb_hdwe_filt = value; };
  void put_Vc_hdwe_sum(const int16_t value)              { Vc_hdwe_sum = value; };
  void put_ib_amp_hdwe_filt(const int16_t value)     { ib_amp_hdwe_filt = value; };
  void put_ib_noa_hdwe_filt(const int16_t value)     { ib_noa_hdwe_filt = value; };
  void put_Tb_filt(const int16_t value)              { Tb_filt = value; };
  void put_vb_filt(const int16_t value)              { vb_filt = value; };
  void put_ib_filt(const int16_t value)              { ib_filt = value; };
  void put_soc(const int16_t value)             { soc = value; };
  void put_soc_min(const int16_t value)         { soc_min = value; };
  void put_soc_ekf(const int16_t value)         { soc_ekf = value; };
  void put_voc_filt(const int16_t value)             { voc_filt = value; };
  void put_voc_stat_filt(const int16_t value)        { voc_stat_filt = value; };
  void put_e_wrap_filt(const int16_t value)     { e_wrap_filt = value; };
  void put_fltw(const uint32_t value)           { fltw = value; };
  void put_falw(const uint32_t value)           { falw = value; };
protected:
  SerialRAM *rP_;
};