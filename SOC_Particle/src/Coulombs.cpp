//
// MIT License
//
// Copyright (C) 2023 - Dave Gutz
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
Coulombs::Coulombs(double *sp_delta_q, float *sp_t_last, const float q_cap_rated, const float t_rlim,
  const double s_coul_eff, const float dx_voc, const float dy_voc, const float dz_voc)
  : q_(q_cap_rated), q_capacity_(q_cap_rated), q_cap_rated_(q_cap_rated), q_cap_rated_scaled_(q_cap_rated),
    q_min_(0.), sat_(true), soc_(1.), soc_min_(0.), sp_delta_q_(sp_delta_q), sp_t_last_(sp_t_last), t_rlim_(0.017),
    chem_()
    {
      put_dx_voc(dx_voc);
      put_dy_voc(dy_voc);
      put_dz_voc(dz_voc);
      coul_eff_ = (chem_.coul_eff*s_coul_eff);
    }
Coulombs::~Coulombs() {}


// operators
// Pretty print
void Coulombs::pretty_print()
{
#ifndef SOFT_DEPLOY_PHOTON
  Serial.printf("Coulombs:\n");
  Serial.printf(" coul_eff%9.5f\n", coul_eff_);
  Serial.printf(" delta_q%9.1f, C\n", *sp_delta_q_);
  Serial.printf(" delta_q_inf/delta_q_abs%9.1f / %9.1f %8.4f C\n", delta_q_inf_, delta_q_abs(), delta_q_inf_/delta_q_abs());
  Serial.printf(" delta_q_neg%9.1f C, time_neg%9.1f s\n", delta_q_neg_, time_neg_);
  Serial.printf(" delta_q_pos%9.1f C, time_pos%9.1f s\n", delta_q_pos_, time_pos_);
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
  Serial.printf(" t_last%5.1f dg C\n", *sp_t_last_);
  Serial.printf(" rated_t%5.1f dg C\n", chem_.rated_temp);
  Serial.printf(" t_rlim%7.3f dg C / s\n", t_rlim_);
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
void Coulombs::apply_cap_scale(const float scale)
{
  q_cap_rated_scaled_ = scale * q_cap_rated_;
  q_capacity_ = calculate_capacity(*sp_t_last_);
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
void Coulombs::apply_delta_q_t(const boolean reset)
{
  if ( !reset ) return;
  q_capacity_ = calculate_capacity(*sp_t_last_);
  q_ = q_capacity_ + *sp_delta_q_;
  soc_ = q_ / q_capacity_;
  resetting_ = true;
}
void Coulombs::apply_delta_q_t(const double delta_q, const float temp_c)
{
  *sp_delta_q_ = delta_q;
  *sp_t_last_ = temp_c;
  apply_delta_q_t(true);
}


// Memory set, adjust book-keeping as needed.  delta_q preserved
void Coulombs::apply_soc(const float soc, const float temp_c)
{
  soc_ = soc;
  q_capacity_ = calculate_capacity(temp_c);
  q_ = soc*q_capacity_;
  *sp_delta_q_ = q_ - q_capacity_;
  resetting_ = true;     // momentarily turn off saturation check
}

// Capacity
double Coulombs::calculate_capacity(const float temp_c)
{
  return( q_cap_rated_scaled_ * (1+chem_.dqdt*(temp_c - chem_.rated_temp)) );
}

/* Coulombs::count_coulombs:  Count coulombs based on true=actual capacity
Inputs:
  dt              Integration step, s
  temp_c          Battery temperature, deg C
  charge_curr     Charge, A
  sat             Indication that battery is saturated, T=saturated
  tlast           Past value of battery temperature used for rate limit memory, deg C
  coul_eff        Coulombic efficiency - the fraction of charging input that gets turned into usable Coulombs
Outputs:
  q_capacity_     Saturation charge at temperature, C
  *sp_delta_q_    Charge change since saturated, C
  *sp_t_last_     Updated value of battery temperature used for rate limit memory, deg C
  resetting_      Sticky flag for initialization, T=reset
  soc_            Fraction of saturation charge (q_capacity_) available (0-1) 
  soc_min_        Estimated soc where battery BMS will shutoff current, fraction
  q_min_          Estimated charge at low voltage shutdown, C\
*/
float Coulombs::count_coulombs(const double dt, const boolean reset_temp, const float temp_c, const float charge_curr,
  const boolean sat, const double delta_q_ekf)
{
    // Rate limit temperature.   When modeling, reset_temp.  In real world, rate limited Tb ramps Coulomb count since bms_off
    if ( reset_temp && sp.mod_vb() )
    {
      *sp_t_last_ = temp_c;
    }
    float temp_lim = max(min( temp_c, *sp_t_last_ + t_rlim_*dt), *sp_t_last_ - t_rlim_*dt);

    // State change
    double d_delta_q = charge_curr * dt;
    double d_delta_q_inf = d_delta_q;
    if ( charge_curr>0. ) d_delta_q *= coul_eff_;
    d_delta_q -= chem_.dqdt*q_capacity_*(temp_lim - *sp_t_last_);
    d_delta_q_inf = d_delta_q;
    sat_ = sat;

    // Saturation.   Goal is to set q_capacity and hold it so remember last saturation status.
    if ( sat )
    {
        if ( d_delta_q > 0 )
        {
            d_delta_q = 0.;
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
    // else if ( reset_temp && !ap.fake_faults ) *sp_delta_q_ = delta_q_ekf;  // Solution to booting up unsaturated
    resetting_ = false;     // one pass flag

    // Integration.   Can go to negative
    q_capacity_ = calculate_capacity(temp_lim);
    if ( !reset_temp && !cp.inf_reset )
    {
      *sp_delta_q_ = max(min(*sp_delta_q_ + d_delta_q, 0.0), -q_capacity_*1.5);
      delta_q_inf_ += d_delta_q_inf;
      if ( d_delta_q > 0. )
      {
        delta_q_pos_ += d_delta_q;
        time_pos_ += dt;
      }
      else
      {
        delta_q_neg_ += d_delta_q;
        time_neg_ += dt;
      }
      delta_q_abs_ += abs(d_delta_q_inf) / 2.;
    }
    else
    {
      if ( cp.inf_reset ) *sp_delta_q_ = 0.;
      delta_q_abs_ = *sp_delta_q_ / 2.;
      delta_q_inf_ = *sp_delta_q_;
      delta_q_neg_ = *sp_delta_q_;
      delta_q_pos_ = 0.;
      cp.inf_reset = false;
      time_neg_ = 0.;
      time_pos_ = 0.;
    }
    // if ( sp.debug()==-24 )Serial.printf("Mon:  charge_curr%7.3f d_delta_q%10.6f delta_q%10.1f temp_lim%7.3f t_last%7.3f\n", charge_curr, d_delta_q, *sp_delta_q_, temp_lim, *sp_t_last_);
    q_ = q_capacity_ + *sp_delta_q_;
    q_inf_ = q_capacity_ + delta_q_inf_;

    // Normalize
    soc_ = q_ / q_capacity_;
    soc_inf_ = q_inf_ / q_capacity_;
    soc_min_ = chem_.soc_min_T_->interp(temp_lim);
    q_min_ = soc_min_ * q_capacity_;

    // Save and return
    if ( sp.debug()==-99 )
      Serial.printf("sat, dt, temp_c, temp_c_z, charge_curr, dq, dqt+, ddq, q, soc_min soc, %d, %7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%12.1f,%10.7f,%10.7f,\n",
       sat, dt, temp_c, *sp_t_last_, charge_curr, charge_curr * dt, chem_.dqdt*q_capacity_*(temp_lim - *sp_t_last_), d_delta_q, q_, soc_min_, soc_);
    if ( sp.debug()==-99 )
      Serial1.printf("sat, dt, temp_c, temp_c_z, charge_curr, dq, dqt+, ddq, q, soc, %d, %7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%12.1f,%10.7f,%10.7f,\n",
       sat, dt, temp_c, *sp_t_last_, charge_curr, charge_curr * dt, chem_.dqdt*q_capacity_*(temp_lim - *sp_t_last_), d_delta_q, q_, soc_min_, soc_);
    *sp_t_last_ = temp_lim;
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

