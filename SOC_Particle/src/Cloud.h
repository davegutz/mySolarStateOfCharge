#pragma once

#include "Battery.h"

// Publishing
struct Publish
{
  uint32_t now;
  String unit;
  String hm_string;
  double Tb;
  float Ib;
  float Voc;
  bool sat;
  bool saturated;
  float tcharge;
  float Amp_hrs_remaining_ekf;
  float Amp_hrs_remaining_soc;
};

void assign_publist(Publish* pubList, const unsigned long long now, const String unit, const String hm_string,
  Sensors* Sen, const int num_timeouts, BatteryMonitor* Mon);
