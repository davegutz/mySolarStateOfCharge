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

class BatteryMonitor;

#pragma once


class KalmanFilter {
public:
  KalmanFilter(const double dt, const double init_pos, const double Q_std, const double R);
  ~KalmanFilter();
  double calculate(const boolean reset, const double dt, const double in);
  boolean get_reset() { return reset_; };
  double **Fx() { return Fx_; };
  double *x() { return x_; };
  double get_u() { return u_; };
  double get_v() { return x_[1]; };
  double get_x() { return x_[0]; };
  void kf_init(const double in);
  void predict(const double dt);
  void pretty_print(void);
  void print_serial_header(const char suffix);
  void print_serial();
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
  double **Q_;    
  double Qstdsq_; // Standard deviation squared of the process noise
  boolean reset_; // Reset command status
  double Rsq_;    // Standard deviation squared of the measurement noise
  double S_;
  double u_;      // Measurement update for x
  double *x_;
  double y_;
};
