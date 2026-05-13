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
#include "FaultStore.h"
#include "Sensors.h"

extern SavedPars sp;       // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level

// struct Flt_st.  This file needed to avoid circular reference to sp in header files
void Flt_st::assign(const uint64_t now, BatteryMonitor *Mon, Sensors *Sen)
{
  this->t_flt = now;
  this->Tb_hdwe_filt = int16_t(Sen->Tb_hdwe_f()*SCL_600);
  this->vb_hdwe_filt = int16_t(Sen->Vb_hdwe_f()/ap.nS()*sp.vb_hist_slr());
  this->Vc_hdwe_sum = int16_t((Sen->ShuntAmp->Vc() + Sen->ShuntNoAmp->Vc())*SCL_3000);
  this->ib_amp_hdwe_filt = int16_t(Sen->Ib_amp_hdwe_f()/ap.nP()*sp.ib_hist_m_slr());
  this->ib_noa_hdwe_filt = int16_t(Sen->Ib_noa_hdwe_f()/ap.nP()*sp.ib_hist_n_slr());
  this->Tb_filt = int16_t(Sen->Tb_f()*SCL_600);
  this->vb_filt = int16_t(Sen->Vb_f()/ap.nS()*sp.vb_hist_slr());
  this->ib_filt = int16_t(Sen->Ib_f()/ap.nP()*sp.ib_hist_n_slr());
  this->soc = int16_t(Mon->soc()*SCL_16000);
  this->soc_min = int16_t(Mon->soc_min()*SCL_16000);
  this->soc_ekf = int16_t(Mon->soc_ekf()*SCL_16000);
  this->voc_filt = int16_t((Mon->voc_stat_f() + Mon->dv_hys())*sp.vb_hist_slr());
  this->voc_stat_filt = int16_t(Mon->voc_stat_f()*sp.vb_hist_slr());
  this->e_wrap_filt = int16_t(Sen->Flt->e_wrap_filt()*sp.vb_hist_slr());
  this->e_wrap_m_filt = int16_t(Sen->Flt->e_wrap_m_filt()*sp.vb_hist_slr());
  this->e_wrap_m_trim = int16_t(Sen->Flt->e_wrap_m_trim()*sp.vb_hist_slr());
  this->e_wrap_n_filt = int16_t(Sen->Flt->e_wrap_n_filt()*sp.vb_hist_slr());
  this->fltw = Sen->Flt->fltw();
  this->falw = Sen->Flt->falw();
}

// struct Flt_st.  This file needed to avoid circular reference to sp in header files
void Flt_st::assign_unfilt(const uint64_t now, BatteryMonitor *Mon, Sensors *Sen)
{
  this->t_flt = now;
  this->Tb_hdwe_filt = int16_t(Sen->Tb_hdwe()*SCL_600);
  this->vb_hdwe_filt = int16_t(Sen->Vb_hdwe()/ap.nS()*sp.vb_hist_slr());
  this->Vc_hdwe_sum = int16_t((Sen->ShuntAmp->Vc() + Sen->ShuntNoAmp->Vc())*SCL_3000);
  this->ib_amp_hdwe_filt = int16_t(Sen->Ib_amp_hdwe()/ap.nP()*sp.ib_hist_m_slr());
  this->ib_noa_hdwe_filt = int16_t(Sen->Ib_noa_hdwe()/ap.nP()*sp.ib_hist_n_slr());
  this->Tb_filt = int16_t(Sen->Tb_f()*SCL_600);
  this->vb_filt = int16_t(Sen->Vb_f()/ap.nS()*sp.vb_hist_slr());
  this->ib_filt = int16_t(Sen->Ib_f()/ap.nP()*sp.ib_hist_n_slr());
  this->soc = int16_t(Mon->soc()*SCL_16000);
  this->soc_min = int16_t(Mon->soc_min()*SCL_16000);
  this->soc_ekf = int16_t(Mon->soc_ekf()*SCL_16000);
  this->voc_filt = int16_t((Mon->voc_stat_f() + Mon->dv_hys())*sp.vb_hist_slr());
  this->voc_stat_filt = int16_t(Mon->voc_stat_f()*sp.vb_hist_slr());
  this->e_wrap_filt = int16_t(Sen->Flt->e_wrap_filt()*sp.vb_hist_slr());
  this->e_wrap_m_filt = int16_t(Sen->Flt->e_wrap_m_filt()*sp.vb_hist_slr());
  this->e_wrap_m_trim = int16_t(Sen->Flt->e_wrap_m_trim()*sp.vb_hist_slr());
  this->e_wrap_n_filt = int16_t(Sen->Flt->e_wrap_n_filt()*sp.vb_hist_slr());
  this->fltw = Sen->Flt->fltw();
  this->falw = Sen->Flt->falw();
}

// Copy function
void Flt_st::copy_to_Flt_ram_from(Flt_st input)
{
  t_flt = input.t_flt;
  Tb_hdwe_filt = input.Tb_hdwe_filt;
  vb_hdwe_filt = input.vb_hdwe_filt;
  Vc_hdwe_sum = input.Vc_hdwe_sum;
  ib_amp_hdwe_filt = input.ib_amp_hdwe_filt;
  ib_noa_hdwe_filt = input.ib_noa_hdwe_filt;
  Tb_filt = input.Tb_filt;
  vb_filt = input.vb_filt;
  ib_filt = input.ib_filt;
  soc = input.soc;
  soc_min = input.soc_min;
  soc_ekf = input.soc_ekf;
  voc_filt = input.voc_filt;
  voc_stat_filt = input.voc_stat_filt;
  e_wrap_filt =input.e_wrap_filt;
  e_wrap_m_filt =input.e_wrap_m_filt;
  e_wrap_m_trim =input.e_wrap_m_trim;
  e_wrap_n_filt =input.e_wrap_n_filt;
  fltw = input.fltw;
  falw = input.falw;
}

// Nominal values
void Flt_st::nominal()
{
  this->t_flt = 1ULL;
  this->Tb_hdwe_filt = int16_t(0);
  this->vb_hdwe_filt = int16_t(0);
  this->Vc_hdwe_sum = int16_t(0);
  this->ib_amp_hdwe_filt = int16_t(0);
  this->ib_noa_hdwe_filt = int16_t(0);
  this->Tb_filt = int16_t(0);
  this->vb_filt = int16_t(0);
  this->ib_filt = int16_t(0);
  this->soc = int16_t(0);
  this->soc_min = int16_t(0);
  this->soc_ekf = int16_t(0);
  this->voc_filt = int16_t(0);
  this->voc_stat_filt = int16_t(0);
  this->e_wrap_filt = int16_t(0);
  this->e_wrap_m_filt = int16_t(0);
  this->e_wrap_m_trim = int16_t(0);
  this->e_wrap_n_filt = int16_t(0);
  this->fltw = uint32_t(0);
  this->falw = uint32_t(0);
  this->dummy = 0UL;
}

// Print functions
void Flt_st::pretty_print(const String code)
{
  char buffer[32];
  strcpy(buffer, "---");
  if ( this->t_flt > 1UL )
  {
    Serial.printf("code %s\n", code.c_str());
    time_long_2_str((time_t)(this->t_flt / 1000ULL), buffer);
    Serial.printf("buffer %s\n", buffer);
    Serial.printf("t %lld\n", this->t_flt);
    Serial.printf("Tb_hdwe_filt %7.3f\n", float(this->Tb_hdwe_filt)/SCL_600);
    Serial.printf("vb_hdwe_filt %7.3f\n", float(this->vb_hdwe_filt)/sp.vb_hist_slr());
    Serial.printf("Vc_hdwe_sum %7.3f\n", float(this->Vc_hdwe_sum)/SCL_3000);
    Serial.printf("ib_amp_hdwe_filt %7.3f\n", float(this->ib_amp_hdwe_filt)/sp.ib_hist_m_slr());
    Serial.printf("ib_noa_hdwe_filt %7.3f\n", float(this->ib_noa_hdwe_filt)/sp.ib_hist_n_slr());
    Serial.printf("Tb_filt %7.3f\n", float(this->Tb_filt)/SCL_600);
    Serial.printf("vb_filt %7.3f\n", float(this->vb_filt)/sp.vb_hist_slr());
    Serial.printf("ib_filt %7.3f\n", float(this->ib_filt)/sp.ib_hist_n_slr());
    Serial.printf("soc %7.4f\n", float(this->soc)/SCL_16000);
    Serial.printf("soc_min %7.4f\n", float(this->soc_min)/SCL_16000);
    Serial.printf("soc_ekf %7.4f\n", float(this->soc_ekf)/SCL_16000);
    Serial.printf("voc_filt %7.3f\n", float(this->voc_filt)/sp.vb_hist_slr());
    Serial.printf("voc_stat_filt %7.3f\n", float(this->voc_stat_filt)/sp.vb_hist_slr());
    Serial.printf("e_wrap_filt %7.3f\n", float(this->e_wrap_filt)/sp.vb_hist_slr());
    Serial.printf("e_wrap_m_filt %7.3f\n", float(this->e_wrap_m_filt)/sp.vb_hist_slr());
    Serial.printf("e_wrap_m_trim %7.3f\n", float(this->e_wrap_m_trim)/sp.vb_hist_slr());
    Serial.printf("e_wrap_n_filt %7.3f\n", float(this->e_wrap_n_filt)/sp.vb_hist_slr());
    Serial.printf("fltw %ld falw %ld\n", this->fltw, this->falw);
  }
}

void SavedPars::print_fault_header(Publish *pubList)
{
    String txBuf;

    txBuf = String::format("Config:  %s \n", pubList->unit.c_str());
    sendTxBuf(txBuf, true, IN_SERVICE);

    txBuf = String::format("fltb,  date,             time_ux,    Tb_h_f, vb_h_f, Vc_h, ib_amp_hdwe_f, ib_noa_hdwe_f, Tb_f, vb_f, ib_f, soc, soc_min, soc_ekf, voc_f, voc_stat_f, e_wrap_filt, e_wrap_m_filt, e_wrap_m_trim, e_wrap_n_filt, fltw, falw,\n");
    sendTxBuf(txBuf, true, IN_SERVICE);
}

void Flt_st::print_flt(const String code)
{
  char buffer[32];
  strcpy(buffer, "---");
  String txBuf;

  if ( this->t_flt > 1UL )
  {
    time_long_2_str((time_t)(this->t_flt / 1000ULL), buffer);
    txBuf = String::format("%s, %s, %lld, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %7.4f, %7.4f, %7.4f, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %ld, %ld,\n",
      code.c_str(), buffer, this->t_flt,
      float(this->Tb_hdwe_filt)/SCL_600,
      float(this->vb_hdwe_filt)/sp.vb_hist_slr(),
      float(this->Vc_hdwe_sum)/SCL_3000,
      float(this->ib_amp_hdwe_filt)/sp.ib_hist_m_slr(),
      float(this->ib_noa_hdwe_filt)/sp.ib_hist_n_slr(),
      float(this->Tb_filt)/SCL_600,
      float(this->vb_filt)/sp.vb_hist_slr(),
      float(this->ib_filt)/sp.ib_hist_n_slr(),
      float(this->soc)/SCL_16000,
      float(this->soc_min)/SCL_16000,
      float(this->soc_ekf)/SCL_16000,
      float(this->voc_filt)/sp.vb_hist_slr(),
      float(this->voc_stat_filt)/sp.vb_hist_slr(),
      float(this->e_wrap_filt)/sp.vb_hist_slr(),
      float(this->e_wrap_m_filt)/sp.vb_hist_slr(),
      float(this->e_wrap_m_trim)/sp.vb_hist_slr(),
      float(this->e_wrap_n_filt)/sp.vb_hist_slr(),
      this->fltw,
      this->falw);
    sendTxBuf(txBuf, true, IN_SERVICE);
  }
}

// Regular put
void Flt_st::put(Flt_st source)
{
  copy_to_Flt_ram_from(source);
}

// nominalize
void Flt_st::put_nominal()
{
  Flt_st source;
  source.nominal();
  copy_to_Flt_ram_from(source);
}

// Class fault ram to interface Flt_st to ram
Flt_ram::Flt_ram()
  : rP_(nullptr)
{
  Flt_st();
}
Flt_ram::~Flt_ram(){}

// Save all
void Flt_ram::put(const Flt_st value)
{
  copy_to_Flt_ram_from(value);
}

// nominalize
void Flt_ram::put_nominal()
{
  Flt_st source;
  source.nominal();
  put(source);
}
