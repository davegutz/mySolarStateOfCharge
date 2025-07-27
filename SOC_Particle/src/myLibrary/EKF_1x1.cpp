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
#include "EKF_1x1.h"
#include <math.h>
#include "parameters.h"
extern SavedPars sp; // Various parameters to be static at system level and saved through power cycle
extern VolatilePars ap; // Various adjustment parameters shared at system level
// extern int8_t debug();


// class EKF_1x1
// constructors
EKF_1x1::EKF_1x1(){}
EKF_1x1::~EKF_1x1() {}

// operators

// functions
//1x1 Extended Kalman Filter predict
void EKF_1x1::predict_ekf(double u, const boolean freeze)
{
  /*
  1x1 Extended Kalman Filter predict
  Inputs:
    u   1x1 input, =ib, A
    Bu  1x1 control transition, Ohms
    Fx  1x1 state transition, V/V
  Outputs:
    x   1x1 Kalman state variable = Vsoc (0-1 fraction)
    P   1x1 Kalman probability
  */
  u_ = u;
  freeze_ = freeze;
  this->ekf_predict(&Fx_, &Bu_);
  x_ = Fx_*x_ + Bu_*u_;
  if ( isnan(P_) ) P_ = 0.;   // reset overflow
  P_ = Fx_*P_*Fx_ + Q_*ap.ekf_q*ap.ekf_q;
  x_prior_ = x_;
  P_prior_ = P_;
}

// y <- C@x + D@u
// Backward Euler integration of x
void EKF_1x1::update_ekf(const double z, double x_min, double x_max)
{
  /*1x1 Extended Kalman Filter update
  Inputs:
    z   1x1 input, =voc, dynamic predicted by other model, V
    R   1x1 Kalman state uncertainty
    Q   1x1 Kalman process uncertainty
    H   1x1 Jacobian sensitivity dV/dSOC
  Outputs:
    x   1x1 Kalman state variable = Vsoc (0-1 fraction)
    hx  1x1 Output of observation function h(x)
    y   1x1 Residual z-hx, V
    P   1x1 Kalman uncertainty covariance
    K   1x1 Kalman gain
    S   1x1 system uncertainty
    SI  1x1 system uncertainty inverse
  */
  this->ekf_update(&hx_, &H_);
  z_ = z;
  double pht = P_*H_;
  S_ = H_*pht + R_*ap.ekf_r*ap.ekf_r;
  if ( abs(S_) > 1e-12) K_ = pht / S_;  // Using last-good-value if S_ = 0
  y_ = z_ - hx_;
  x_ = max(min( x_ + K_*y_, x_max), x_min);
  if ( ap.ekf_x != 0. )
  {
    x_ = ap.ekf_x;
    ap.ekf_x = 0.;
  }
  double i_kh = 1. - K_*H_;
  P_ *= i_kh;
  if ( ap.ekf_p != 0. )
  {
    P_ = ap.ekf_p;
    ap.ekf_p = 0.;
  }
  x_post_ = x_;
  P_post_ = P_;
  if ( sp.debug()==35 )
  {
    Serial.printf("EKF_1x1::update_ekf, u_,frz_,z_,hx_,x_Prior_,x_,P_,H_,S_,K_,y_,  %7.4f, %d, %7.4f, %7.4f,%11.8f,%11.8f,%11.8f, %11.8f, %7.4f, %7.4f,%10.7f,%7.4f,\n",
      u_, freeze_, z_, hx_, x_prior_, x_, P_, P_prior_, H_, S_, K_, y_);
    Serial1.printf("EKF_1x1::update_ekf, u_,frz_,z_,hx_,x_Pr,ior_,x_,P_,H_,S_,K_,y_,  %7.4f, %d, %7.4f, %7.4f,%11.8f,%11.8f,%11.8f, %11.8f, %7.4f, %7.4f,%10.7f,%7.4f,\n",
      u_, freeze_, z_, hx_, x_prior_, x_, P_, P_prior_, H_, S_, K_, y_);
  }
}

// Initialize
void EKF_1x1::init_ekf(double soc, double Pinit)
{
  x_ = soc;
  P_ = Pinit;
}

// Pretty Print
 void EKF_1x1::pretty_print(void)
 {
#ifndef SOFT_DEPLOY_PHOTON
  Serial.printf("EKF_1x1:\n");
  Serial.printf("In:\n");
  Serial.printf("  u  %8.4f, A\n", u_);
  Serial.printf("  frz %d, T=frz\n", freeze_);
  Serial.printf("  z  %8.4f, V\n", z_);
  Serial.printf("  R%11.8f\n", R_);
  Serial.printf("  Q%11.8f\n", Q_);
  Serial.printf("  H   %7.3f\n", H_);
  Serial.printf("Out:\n");
  Serial.printf("  xp %11.8f, Vsoc (0-1 fraction)\n", x_prior_);
  Serial.printf("  x  %11.8f, Vsoc (0-1 fraction)\n", x_);
  Serial.printf("  Fx %11.8f\n", Fx_);
  Serial.printf("  Bu %11.8f\n", Bu_);
  Serial.printf("  hx %8.4f\n", hx_);
  Serial.printf("  y   %8.4f, V\n", y_);
  Serial.printf("  Pp%11.8f\n", P_prior_);
  Serial.printf("  P%11.8f\n", P_);
  Serial.printf("  K%11.8f\n", K_);
  Serial.printf("  S%11.8f\n", S_);
#else
     Serial.printf("EKF_1x1: silent DEPLOY\n");
#endif
 }

// Serial print
 void EKF_1x1::serial_print(const unsigned long long now, const float dt)
 {
  double cTime = double(now)/1000.;

  Serial.printf("unit_ekf,%13.3f,%7.3f,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,%10.7g,\n",
    cTime, dt, Fx_, Bu_, Q_, R_, P_, S_, K_, u_, x_, y_, z_, x_prior_, P_prior_, x_post_, P_post_, hx_, H_);
 }
