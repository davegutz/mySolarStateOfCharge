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

class BatteryMonitor;

#pragma once


class KalmanFilter {
public:
  KalmanFilter(const double dt, const double init_pos, const double Q_std, const double R_std);
  ~KalmanFilter();
  double calculate(const bool reset, const double dt, const double in);
  bool get_reset() { return reset_; };
  double **Fx() { return Fx_; };
  double *x() { return x_; };
  void kf_init(const double in);
  double kf_u() { return u_; };
  double kf_v() { return x_[1]; };
  double kf_x() { return x_[0]; };
  void predict();
  void pretty_print(void);
  void print_serial_header(const char suffix);
  void print_serial();
  void q_std(const double q) { Q_stdsq_ = max(q * q, 0.); };
  void r_std(const double r) { R_stdsq_ = max(r * r, 0.); };
  double update(const double meas);
private:
  double dt_;     // Update time, s
  const int ROWS_ = 2;
  const int COLS_ = 2;
  double **Fx_;   // State transition
  double *G_;     // Control B matrix mapping inputs to states
  double *H_;     // Jacobian
  double *K_;     // Kalman gain
  double **P_;    // Kalman probability matrix
  double **P_prior_;  // Intermediate Kalman probability matrix
  double **Q_;    
  double Q_stdsq_; // Standard deviation squared of the process noise
  bool reset_; // Reset command status
  double R_stdsq_;    // Standard deviation squared of the measurement noise
  double S_;
  double u_;      // Measurement update for x
  double *x_;
  double *x_prior_;  // Intermediate calculation
  double y_;
};
