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

/**
* Implementation of KalmanFilter class.
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/
#include "application.h"
#include "KF_1x1.h"
#include <math.h>
#include "parameters.h"
extern SavedPars sp; // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level
// extern int8_t debug();

#include "KF_1x1.h"

/*  1x1 General Purpose Extended Kalman Filter.   Inherit from this class and include kf_predict and
    kf_update methods in the parent
*/
KalmanFilter::KalmanFilter(const double dt, const double init_pos, double Q_std, const double R_std):
 dt_(dt), Q_stdsq_(Q_std*Q_std), R_stdsq_(R_std*R_std), S_(0.)
{
/*
Initializes a 1D Kalman filter with a constant velocity model.

Args:
    initial_position (float): Initial estimate of the position.
    dt (float): Time step between measurements.
    Qstd = proc_noise_std (float): Standard deviation of the process noise (velocity).
    R = u_noise_std (float): Standard deviation of the measurement noise (position).
*/

    G_ = new double[2];
    H_ = new double[2];
    Fx_ = new double*[ROWS_];
    K_ = new double[2];
    P_ = new double*[ROWS_];
    P_prior_ = new double*[ROWS_];
    Q_ = new double*[ROWS_];
    x_prior_ = new double[ROWS_];
    x_ = new double[ROWS_];
    for (int i=0; i<ROWS_; i++)
    {
        Fx_[i] = new double[COLS_];
        P_prior_[i] = new double[COLS_];
        P_[i] = new double[COLS_];
        Q_[i] = new double[COLS_];
    }
    kf_init(init_pos);
}

KalmanFilter::~KalmanFilter()
{
    for (int i = 0; i < ROWS_; ++i)
    {
        delete[] Fx_[i];
        delete[] P_prior_[i];
        delete[] P_[i];
        delete[] Q_[i];
    }
    delete[] Fx_;
    Fx_ = nullptr;
    delete G_;
    G_ = nullptr;
    delete H_;
    H_ = nullptr;
    delete K_;
    K_ = nullptr;
    delete[] P_prior_;
    P_prior_ = nullptr;
    delete[] P_;
    P_ = nullptr;
    delete[] Q_;
    Q_ = nullptr;
    delete x_;
    x_prior_ = nullptr;
    delete x_prior_;
    x_ = nullptr;
}

double KalmanFilter::calculate(const bool reset, const double dt, const double in)
{
    double out = 0.;
    reset_ = reset;
    dt_ = dt;

    if ( reset_ )
    {
        kf_init(in);
        return ( in );
    }

    predict();
    out = update(in);

    return ( out );
}

void KalmanFilter::kf_init(const double in)
{
    u_ = in;
    Fx_[0][0] = 1.0;       Fx_[0][1] = dt_;
    Fx_[1][0] = 0.0;       Fx_[1][1] = 1.0;
    H_[0] = 1.0;           H_[1] = 0.;
    S_ = R_stdsq_;
    K_[0] = 0.;            K_[1] = 0.;
    x_prior_[0] = in;      x_prior_[1] = 0.;
    x_[0] = in;            x_[1] = 0.;
    y_ = 0.;
    for (int i=0; i<ROWS_; i++)
    {
        for (int j=0; j<COLS_; j++)
        {
            P_prior_[i][j] = 0.;
            P_[i][j] = 0.;
            Q_[i][j] = 0.;
        }
    }
}

void KalmanFilter::predict()
{
/*
Performs the prediction step of the Kalman filter.
Inputs:
    u   1x1 input, =ib, A
    Bu  1x1 control transition, Ohms
    Fx  2x2 state transition, V/V
Outputs:
    G   2x1 Covariance matrix mapping input to states
    x   2x1 Kalman state variable =
    P   2x2 Kalman probability
*/

    // State transition matrix (constant velocity model)
    Fx_ [0][1] = dt_;

    // Process noise covariance matrix (assuming noise affects acceleration)
    G_[0] = 0.5*dt_*dt_;    G_[1] = dt_;

    // Q = G @ G.T * Q_std**2
    double Q_fac = dt_*dt_*Q_stdsq_;
    Q_[0][0] = dt_*dt_/4.;      Q_[0][1] = dt_/2.;
    Q_[1][0] = dt_/2.;          Q_[1][1] = 1.;
    for ( int i=0; i<2; i++) for ( int j=0; j<2; j++) Q_[i][j] *= Q_fac;

    // Predict state and covariance
    // x = Fx @ x
    x_[0] = Fx_[0][0]*x_[0] + Fx_[0][1]*x_[1];
    x_[1] = Fx_[1][0]*x_[0] + Fx_[1][1]*x_[1];
    for ( int i=0; i<2; i++) x_prior_[i] = x_[i];

    // P = Fx @ P @ Fx.T + Q
    P_[0][0] = P_[0][0] + P_[0][1]*dt_ + P_[1][0]*dt_ + P_[1][1]*dt_*dt_ + Q_[0][0];
    P_[0][1] = P_[0][1] + P_[1][1]*dt_ + Q_[0][1];
    P_[1][0] = P_[1][0] + P_[1][1]*dt_ + Q_[1][0];
    P_[1][1] = P_[1][1] + Q_[1][1];
    for ( int i=0; i<2; i++) for ( int j=0; j<2; j++) P_prior_[i][j] = P_[i][j];
}

double KalmanFilter::update(const double meas)
{
/*
Updates the state estimate and covariance matrix based on the new measurement.

Args:
    meas: The new x[0] measurement

Inputs:
    u   1x1 input, =ib, A
    Bu  1x1 control transition, Ohms
    Fx  2x2 state transition, V/V
Outputs:
    S   1x1 Kalman gain
    K   2x1 Kalman gain
    y   1x1 output, units of input u (unity gain filter)
    H   1x2 Jacobian
    x   2x1 Kalman state variable = [input units, rate of change of input units]
    P   2x2 Kalman probability matrix
*/
    u_ = meas;
    // Kalman Gain
    // S = H @ P @ H.T + R
    S_ = P_prior_[0][0] + R_stdsq_;

    // K = P @ H.T @ inv(S)
    K_[0] = P_prior_[0][0] / (P_prior_[0][0] + R_stdsq_);
    K_[1] = P_prior_[1][0] / (P_prior_[0][0] + R_stdsq_);
 
    // Update state estimate
    // y = measurement - (H @ x)  # Innovation
    y_ = u_ - x_[0];

    // x = x + (K @ y)
    x_[0] = x_prior_[0] + y_*K_[0];
    x_[1] = x_prior_[1] + y_*K_[1];

    // Update covariance matrix
    // P = (np.eye(x.shape[0]) - K @ H) @ P
    P_[0][0] = (1. - K_[0])*P_prior_[0][0];
    P_[0][1] = (1. - K_[0])*P_prior_[0][1];
    P_[1][0] = -K_[1]*P_prior_[0][0] + P_prior_[1][0];
    P_[1][1] = -K_[1]*P_prior_[0][1] + P_prior_[1][1];

    return ( x_[0] );
}

// Pretty Print
 void KalmanFilter::pretty_print(void)
 {
#ifndef SOFT_DEPLOY_PHOTON
  Serial.printf("KF:\n");
  Serial.printf("In:\n");
  Serial.printf(" u   %8.4f, V\n", u_);
  Serial.printf(" dt_ %8.4f, s\n", dt_);
  Serial.printf(" Fx [ %8.4f, %8.4f]\n    [ %8.4f, %8.4f]\n",
     Fx_[0][0], Fx_[0][1], Fx_[1][0], Fx_[1][1]);
  Serial.printf(" G  [ %8.4f, \n      %8.4f]\n", G_[0], G_[1]);
  Serial.printf(" R_stdsq%8.4g\n", R_stdsq_);
  Serial.printf(" Q_stdsq%8.4g\n", Q_stdsq_);
  Serial.printf(" Q  [%8.4f, %8.4f]\n    [%8.4f, %8.4f]\n",
     Q_[0][0], Q_[0][1], Q_[1][0], Q_[1][1]);
  Serial.printf(" H  [ %8.4f, %8.4f ]\n", H_[0], H_[1]);
  Serial.printf("Out:\n");
  Serial.printf(" x_prior  [%8.4f, \n     %8.4f]\n", x_prior_[0], x_prior_[1]);
  Serial.printf(" x  [%8.4f, \n     %8.4f]\n", x_[0], x_[1]);
  Serial.printf(" y   %8.4f, units of x\n", y_);
  Serial.printf(" P_prior  [%8.4f, %8.4f]\n    [%8.4f, %8.4f]\n",
     P_prior_[0][0], P_prior_[0][1], P_prior_[1][0], P_prior_[1][1]);
  Serial.printf(" P  [%8.4f, %8.4f]\n    [%8.4f, %8.4f]\n",
     P_[0][0], P_[0][1], P_[1][0], P_[1][1]);
  Serial.printf(" K  [%8.4f, \n     %8.4f  ]\n", K_[0], K_[1]);
  Serial.printf(" S   %8.4f\n", S_);
#else
     Serial.printf("EKF_1x1: silent DEPLOY\n");
#endif
 }
