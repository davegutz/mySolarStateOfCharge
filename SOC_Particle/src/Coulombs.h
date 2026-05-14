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
#include "Chemistry_BMS.h"

class Sensors;


// Functions
float nice_zero(const float in, const float thr);
double nice_zero(const double in, const double thr);


// Coulomb Counter Class
class Coulombs
{
public:
  Coulombs();
  Coulombs(double *sp_delta_q, const double q_cap_rated,
    const double s_coul_eff, const float dx_voc, const float dy_voc, const float dz_voc);
  ~Coulombs();
  void apply_cap_scale(const double scale);
  void apply_delta_q(const double delta_q);
  void apply_delta_q_t(const bool reset);
  void apply_delta_q_t(const double delta_q, const double tb_f);
  void apply_resetting(const bool resetting){ resetting_ = resetting; };
  void apply_soc(const double soc, const double tb_f);
  void assign_all_mod() { chem_.assign_all_chm(); };
  double calculate_capacity(const double tb_f);
  Chemistry *chem() { return &chem_; };
  void chem_pretty_print () { chem_.pretty_print(); };
  double coul_eff() { return ( coul_eff_ ); };
  void coul_eff(const double coul_eff) { coul_eff_ = coul_eff; };
  virtual double count_coulombs(Sensors *Sen, const bool reset_temp, const float charge_curr, const bool sat,
  const bool saturated);
  double d_delta_q() { return(d_delta_q_); };
  double delta_q() { return(*sp_delta_q_); };
  double delta_q_abs() { return nice_zero(delta_q_abs_, 1e-6); }
  double delta_q_inf() { return(delta_q_inf_); };
  double delta_q_neg() { return nice_zero(delta_q_neg_, 1e-6); }
  double delta_q_pos() { return nice_zero(delta_q_pos_, 1e-6); }
  uint8_t mod_code() { return chem_.mod_code; };
  virtual void pretty_print();
  void put_dx_voc(const float inp) { chem_.put_dx_voc(inp); }
  void put_dy_voc(const float inp) { chem_.put_dy_voc(inp); }
  void put_dz_voc(const float inp) { chem_.put_dz_voc(inp); }
  double q(){ return (q_); };
  double q_cap_rated(){ return (q_cap_rated_); };
  double q_cap_rated_scaled(){ return (q_cap_rated_scaled_); };
  double q_capacity(){ return (q_capacity_); };
  double q_inf(){ return (q_inf_); };
  bool sat() { return(sat_); };
  bool saturated() { return(saturated_); };
  double soc() { return(soc_); };
  double soc_inf() { return(soc_inf_); };
  double soc_min() { return(soc_min_); };
  double time_neg() { return(time_neg_); };
  double time_pos() { return(time_pos_); };
  virtual float vsat(void) = 0;
protected:
  bool resetting_ = false;  // Sticky flag to coordinate user testing of coulomb counters, T=performing an external reset of counter
  double coul_eff_;   // Coulombic efficiency - the fraction of charging input that gets turned into usable Coulombs
  double d_delta_q_;  // Change in charge for update, C
  double delta_q_abs_;// Total charge book-kept since reset, not reset on saturation, C
  double delta_q_inf_;// Charge since initialized, C
  double delta_q_neg_;// Total negative charge book-kept since reset, not reset on saturation, C
  double delta_q_pos_;// Total positive charge book-kept since reset, not reset on saturation, C
  double dt_;         // Update time, s
  double q_;          // Present charge available to use, except q_min_, C
  double q_capacity_; // Saturation charge at temperature, C
  double q_cap_rated_;// Rated capacity at t_rated_, saved for future scaling, C
  double q_cap_rated_scaled_;// Applied rated capacity at t_rated_, after scaling, C
  double q_inf_;      // Same as q_ but not reset on saturation, C
  double q_min_;      // Floor on charge available to use, C
  bool sat_;          // Indication that battery is potentially saturated, T=saturated
  bool saturated_;    // Battery is confirmed saturated, T=saturated
  double soc_;        // Fraction of saturation charge (q_capacity_) available (0-1)
  double soc_ekf_min_; // Minimum SOC for EKF operation
  double soc_inf_;    // Fraction of saturation charge (q_capacity_) available (-inf - inf)
  double soc_min_;    // As battery cools, the voltage drops and there appears a minimum soc it can deliver
  double *sp_delta_q_;// Charge since saturated, C
  double tb_f_;       // Temperature, deg C
  double tb_f_rate_;  // Tb rate, deg C / s
  double Tb_f_;       // Temperature, deg C
  double Tb_f_rate_;  // Tb rate, deg C / s
  double time_neg_;   // Time spent accumulating delta_q_neg_, s
  double time_pos_;   // Time spent accumulating delta_q_pos_, s
  Chemistry chem_;    // Chemistry
};
