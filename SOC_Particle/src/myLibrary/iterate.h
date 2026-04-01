/***************************************************
  A simple solver library

  Class code for embedded application.

  04-Nov-2022   Dave Gutz   Created
 ****************************************************/

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
  double iterate(const bool verbose, const uint16_t success_count, const bool en_no_soln);
  double x() { return x_; };
protected:
    uint16_t count_;    // Iteration counter
    String desc_;       // Description
    double de_;          // Error change
    double des_;         // Scaled error
    double dx_;          // Input change
    double e_;           // Error
    double ep_;          // Past error
    bool limited_;   // On limits
    double x_;           // Input
    double xmax_;        // Maximum input
    double xmin_;        // Minimum input
    double xp_;          // Past input
};
