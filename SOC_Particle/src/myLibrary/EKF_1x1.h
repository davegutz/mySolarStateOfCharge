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

class BatteryMonitor;

// Lightweight general purpose state space for embedded application
class EKF_1x1
{
public:
  EKF_1x1();
  ~EKF_1x1();
  // operators
  // functions
  void predict_ekf(const double u, const bool freeze);
  virtual void pretty_print(void);
  void print_ekf_serial(BatteryMonitor *Mon);
  double Tb_f_for_hx() { return ( Tb_f_for_hx_); };
  void update_ekf(const double z, double x_min, double x_max);
  double x() { return ( x_ ); };
  double x_f_for_hx() { return ( x_for_hx_); };
  double y() { return ( y_ ); };
  double z() { return ( z_ ); };
  void init_ekf(double soc, double Pinit);
protected:
  double Fx_; // State transition
  double Bu_; // Control transition
  double Q_;  // Process uncertainty
  double R_;  // State uncertainty
  double P_;  // Uncertainty covariance
  double S_;  // System uncertainty
  double K_;  // Kalman gain
  double u_;  // Control input
  double x_;  // Kalman state variable
  double y_;  // Residual z - hx
  double z_;  // Observation of state x
  double x_prior_;  
  double P_prior_;
  double x_post_;
  double P_post_;
  double hx_; // Output of observation function h(x)
  double H_;  // Jacobian of h(x)
  bool freeze_;  // Command to freeze x_ and P_
  uint64_t now_ekf_;  // Time value extracted from sensors, ms
  double dt_ekf_;   // Update time for EKF major frame
  double Tb_f_for_hx_;  // Tb_f used for the hx_ calculation, C
  double x_for_hx_;     // soc used for the hx_ calculation, scalar
  /*
    Implement this function for your EKF model.
    @param fx gets output of state-transition function f(x)
    @param F gets Jacobian of f(x)
    @param hx gets output of observation function h(x)
    @param H gets Jacobian of h(x)
  */
  virtual void ekf_predict(double *Fx, double *Bu) = 0;
  virtual void ekf_update(double *hx, double *H, double *x, double *tb_f) = 0;
};

// Methods
