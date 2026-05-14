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
#include <math.h>
#include "Battery.h"
#include "Coulombs.h"
#include "parameters.h"
#include "command.h"

#include "subs.h" // delay_no_block

extern SavedPars sp;      // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level
extern CommandPars cp;    // Various parameters to be static at system level
extern PublishPars pp;    // For publishing


// class Coulombs
Coulombs::Coulombs() {}
Coulombs::Coulombs(double *sp_delta_q, const double q_cap_rated,
  const double s_coul_eff, const float dx_voc, const float dy_voc, const float dz_voc)
  : resetting_(false), d_delta_q_(0.), delta_q_abs_(0.), delta_q_inf_(0.), delta_q_neg_(0.), delta_q_pos_(0.), dt_(0.),
    q_(q_cap_rated), q_capacity_(q_cap_rated), q_cap_rated_(q_cap_rated), q_cap_rated_scaled_(q_cap_rated), q_inf_(0.), q_min_(0.),
    sat_(true), saturated_(false), soc_(1.), soc_ekf_min_(0.), soc_inf_(0.), soc_min_(0.), sp_delta_q_(sp_delta_q),
    tb_f_(0.), tb_f_rate_(0.), Tb_f_(0.), Tb_f_rate_(0.), time_neg_(0.), time_pos_(0.), chem_()
    {
      coul_eff_ = chem_.coul_eff*s_coul_eff;
      soc_ekf_min_ = chem_.soc_ekf_min;
      put_dx_voc(dx_voc);
      put_dy_voc(dy_voc);
      put_dz_voc(dz_voc);
    }
Coulombs::~Coulombs() {}


// operators
// Pretty print
void Coulombs::pretty_print()
{
#ifndef SOFT_DEPLOY_PHOTON
  Serial.printf("Coulombs:\n");
  Serial.printf(" coul_eff%9.5f\n", coul_eff_);
  Serial.printf(" d_delta_q%9.1f, C\n", d_delta_q_);
  Serial.printf(" delta_q%9.1f, C\n", *sp_delta_q_);
  Serial.printf(" delta_q_inf/delta_q_abs%9.1f / %9.1f %8.4f C\n", delta_q_inf_, delta_q_abs(), delta_q_inf_/delta_q_abs());
  Serial.printf(" delta_q_neg%9.1f C, time_neg%9.1f s\n", delta_q_neg_, time_neg_);
  Serial.printf(" delta_q_pos%9.1f C, time_pos%9.1f s\n", delta_q_pos_, time_pos_);
  Serial.printf(" dt%9.6f, s\n", dt_);
  Serial.printf(" mod_code %d\n", mod_code());
  Serial.printf(" mod %s\n", chem_.decode(mod_code()).c_str());
  Serial.printf(" q%9.1f, C\n", q_);
  Serial.printf(" q_cap%9.1f, C\n", q_capacity_);
  Serial.printf(" q_cap_rat%9.1f, C\n", q_cap_rated_);
  Serial.printf(" q_cap_rat_scl%9.1f, C\n", q_cap_rated_scaled_);
  Serial.printf(" q_min%9.1f, C\n", q_min_);
  Serial.printf(" resetting %d\n", resetting_);
  Serial.printf(" sat %d\n", sat_);
  Serial.printf(" soc%8.4f\n", soc_);
  Serial.printf(" soc_inf%8.4f\n", soc_inf_);
  Serial.printf(" soc_min%8.4f\n", soc_min_);
  Serial.printf(" tb_f_%5.1f dg C\n", tb_f_);
  Serial.printf(" rated_t%5.1f dg C\n", chem_.rated_temp);
  Serial.printf(" tb_f_rate%9.5f dg C / s\n", tb_f_rate_);
  Serial.printf(" tb_f_rate%9.5f dg C / s\n", Tb_f_rate_);
  Serial.printf("Coulombs (mod_code=%d) ", mod_code());
  Serial.printf("Coulombs: silent DEPLOY\n");
  Serial.printf(" Chemistry::\n");
  chem_pretty_print();
#endif
}

// functions

// Scale size of battery and adjust as needed to preserve delta_q.  Tb unchanged.
// Goal is to scale battery and see no change in delta_q on screen of 
// test comparisons.   The rationale for this is that the battery is frequently saturated which
// resets all the model parameters.   This happens daily.   Then both the model and the battery
// are discharged by the same current so the delta_q will be the same.
void Coulombs::apply_cap_scale(const double scale)
{
  q_cap_rated_scaled_ = scale * q_cap_rated_;
  q_capacity_ = calculate_capacity(tb_f_);  // tb_f_ usually Tb_f to reduce electrical noise effects
  q_ = *sp_delta_q_ + q_capacity_; // preserve delta_q, deficit since last saturation (like real life)
  soc_ = q_ / q_capacity_;
  resetting_ = true;     // momentarily turn off saturation check
}

// Memory set, adjust book-keeping as needed.  delta_q, capacity, temp preserved
void Coulombs::apply_delta_q(const double delta_q)
{
  *sp_delta_q_ = delta_q;
  q_ = *sp_delta_q_ + q_capacity_;
  soc_ = q_ / q_capacity_;
  resetting_ = true;     // momentarily turn off saturation check
}

// Memory set, adjust book-keeping as needed.  q_cap_ etc presesrved
void Coulombs::apply_delta_q_t(const bool reset)
{
  if ( !reset ) return;
  q_capacity_ = calculate_capacity(tb_f_);
  q_ = q_capacity_ + *sp_delta_q_;
  soc_ = q_ / q_capacity_;
  resetting_ = true;
}
void Coulombs::apply_delta_q_t(const double delta_q, const double tb_f)
{
  tb_f_ = tb_f;
  *sp_delta_q_ = delta_q;
  apply_delta_q_t(true);
}


// Memory set, adjust book-keeping as needed.  delta_q preserved
void Coulombs::apply_soc(const double soc, const double tb_f)
{
  soc_ = soc;
  q_capacity_ = calculate_capacity(tb_f);
  q_ = soc*q_capacity_;
  *sp_delta_q_ = q_ - q_capacity_;
  resetting_ = true;     // momentarily turn off saturation check
}

// Capacity
double Coulombs::calculate_capacity(const double tb_f)
{
  return( q_cap_rated_scaled_ * (1. + chem_.dqdt*(tb_f - chem_.rated_temp)) );
}

/* Coulombs::count_coulombs:  Count coulombs based on true=actual capacity
Inputs:
  dt              Integration step, s
  reset_temp      Temperature frame reset flag, T=resetting
  tb_f            Battery temperature lagged and rate-limited, deg C
  charge_curr     Charge, A
  sat             Indication that battery is saturated, T=saturated
  coul_eff        Coulombic efficiency - the fraction of charging input that gets turned into usable Coulombs
Outputs:
  q_capacity_     Saturation charge at temperature, C
  *sp_delta_q_    Charge change since saturated, C
  resetting_      Sticky flag for initialization, T=reset
  soc_            Fraction of saturation charge (q_capacity_) available (0-1) 
  soc_min_        Estimated soc where battery BMS will shutoff current, fraction
  q_min_          Estimated charge at low voltage shutdown, C\
*/
double Coulombs::count_coulombs(Sensors *Sen, const bool reset_temp, const float charge_curr, const bool sat,
  const bool saturated)
{
    // Inputs
    dt_ = Sen->T();
    tb_f_ = Sen->Tb_f();
    tb_f_rate_ = Sen->Tb_f_rate();
    Tb_f_ = Sen->Tb_f();
    Tb_f_rate_ = Sen->Tb_f_rate();
    d_delta_q_ = charge_curr * dt_;

    // State change
    double d_delta_q_inf = d_delta_q_;
    if ( charge_curr>0. && !sp.tweak_test() ) d_delta_q_ *= coul_eff_;
    // Capacity changes withi temperature so this effect would be double if used
    // d_delta_q_ -= chem_.dqdt*q_capacity_*tb_f_rate_*dt_;
    d_delta_q_inf = d_delta_q_;
    sat_ = sat;
    saturated_ = saturated;

    // Saturation.   Goal is to set q_capacity and hold it so remember last saturation status.
    if ( saturated_ )
    {
        if ( d_delta_q_ > 0 )
        {
            d_delta_q_ = 0.;
            if ( !resetting_ )
            {
              *sp_delta_q_ = 0.;
            }
        }
        else if ( reset_temp )
        {
          *sp_delta_q_ = 0.;
        }
    }
    // else if ( reset_temp && !ap.fake_faults() ) *sp_delta_q_ = delta_q_ekf;  // Solution to booting up unsaturated
    resetting_ = false;     // one pass flag

    // Integration.   Can go to negative
    q_capacity_ = calculate_capacity(tb_f_);

    // soc integrator: gated only by reset_temp.  cp.inf_reset (HR) MUST NOT touch *sp_delta_q_,
    // otherwise HR would slam soc to 1.0 — see commit history.  Per-window pos/neg/time accumulators
    // are stepped here too; if cp.inf_reset is also asserted this cycle, the inf-counter block
    // below will zero them right after, so the net effect is correct.
    if ( !reset_temp )
    {
      *sp_delta_q_ = max(min(*sp_delta_q_ + d_delta_q_, 0.0), -q_capacity_*1.5);
      if ( d_delta_q_ > 0. )
      {
        delta_q_pos_ += d_delta_q_;
        time_pos_ += dt_;
      }
      else
      {
        delta_q_neg_ += d_delta_q_;
        time_neg_ += dt_;
      }
    }

    // History (inf-counter) family: cp.inf_reset (HR) zeroes them so soc_inf -> 1.0; reset_temp
    // rebaselines them to current *sp_delta_q_.  Neither path disturbs *sp_delta_q_ itself.
    if ( cp.inf_reset )
    {
      delta_q_inf_ = 0.;
      delta_q_abs_ = 0.;
      delta_q_pos_ = 0.;
      delta_q_neg_ = 0.;
      time_pos_ = 0.;
      time_neg_ = 0.;
      cp.inf_reset = false;
    }
    else if ( reset_temp )
    {
      delta_q_abs_ = *sp_delta_q_ / 2.;
      delta_q_inf_ = *sp_delta_q_;
      delta_q_neg_ = *sp_delta_q_;
      delta_q_pos_ = 0.;
      time_pos_ = 0.;
      time_neg_ = 0.;
    }
    else
    {
      delta_q_inf_ += d_delta_q_inf;
      delta_q_abs_ += abs(d_delta_q_inf) / 2.;
    }

    q_ = q_capacity_ + *sp_delta_q_;
    q_inf_ = q_capacity_ + delta_q_inf_;

    // Normalize
    soc_ = q_ / q_capacity_;
    soc_inf_ = q_inf_ / q_capacity_;
    soc_min_ = chem_.soc_min_T_->interp(tb_f_);
    q_min_ = soc_min_ * q_capacity_;

    // Save and return
    
    if ( sp.debug()==36 )
      sendTxBuf(String::format("BM::CC: cc %7.3f dt%9.6f dq_T%9.2f, coul_eff%7.3f d_delta_q%9.2f sp_delta_q_%9.2f q%9.2f\n",
        charge_curr, dt_, -chem_.dqdt*q_capacity_*Tb_f_rate_*dt_, coul_eff_, d_delta_q_, *sp_delta_q_, q_), true, IN_SERVICE);

    if ( sp.debug()==-99 )
      sendTxBuf(String::format("sat, dt_, tb_f_, charge_curr, dq, dqt+, ddq, q, soc_min soc, %d, %7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%12.1f,%10.7f,%10.7f,\n",
        sat, dt_, tb_f_, charge_curr, charge_curr * dt_, chem_.dqdt*q_capacity_*Tb_f_rate_*dt_, d_delta_q_, q_, soc_min_, soc_),
        true, IN_SERVICE);

    return ( soc_ );
}

// Prevent overflows
double nice_zero(const double in, const double thr)
{
    double out = thr;
    if ( abs(in) < thr )
    {
      if ( in < 0. ) out = -thr; 
    }
    else
      out = in;
    return (out);
}

float nice_zero(const float in, const float thr)
{
    float out = thr;
    if ( abs(in) < thr )
    {
      if ( in < 0. ) out = -thr; 
    }
    else
      out = in;
    return (out);
}

