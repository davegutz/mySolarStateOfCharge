/***************************************************
  A simple solver library

  Class code for embedded application.

  04-Nov-2022   Dave Gutz   Created
 ****************************************************/
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

#pragma once

#include "application.h"   // Needed for Photon?
#include "math.h"

// signum/sgn function
template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

class Iterator
{
public:
  Iterator();
  Iterator(const String desc);
  ~Iterator();
  // operators
  // functions
  uint16_t count() { return count_; };
  double dx() { return dx_; };
  double e() { return e_; };
  void e(const double e_in) { e_ = e_in; };
  void increment() { count_++; };
  void init(const double xmax, const double xmin, const double eInit);
  double iterate(const boolean verbose, const uint16_t success_count, const boolean en_no_soln);
  double x() { return x_; };
protected:
    uint16_t count_;    // Iteration counter
    String desc_;       // Description
    double de_;          // Error change
    double des_;         // Scaled error
    double dx_;          // Input change
    double e_;           // Error
    double ep_;          // Past error
    boolean limited_;   // On limits
    double x_;           // Input
    double xmax_;        // Maximum input
    double xmin_;        // Minimum input
    double xp_;          // Past input
};
