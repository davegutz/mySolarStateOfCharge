/*  Heart rate and pulseox calculation Constants

18-Dec-2020 	DA Gutz 	Created from MAXIM code.
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

*/

#ifndef DEBUG_H_
#define DEBUG_H_
#include "subs.h"

void add_verify(String *src, const String addend);

#ifdef DEBUG_DETAIL
    void debug_m1(BatteryMonitor *Mon, Sensors *Sen);
#endif

void debug_4(BatteryMonitor *Mon, Sensors *Sen);
void debug_m7(BatteryMonitor *Mon, Sensors *Sen);

#ifndef HDWE_PHOTON
    void debug_12(BatteryMonitor *Mon, Sensors *Sen);
    void debug_m13(Sensors *Sen);
    void debug_m23(Sensors *Sen);
    void debug_m24(Sensors *Sen);
#endif

void debug_98(BatteryMonitor *Mon, Sensors *Sen);
void debug_99(BatteryMonitor *Mon, Sensors *Sen);
void debug_q(BatteryMonitor *Mon, Sensors *Sen);

#ifdef SOFT_DEBUG_QUEUE
    void debug_queue(const String who);
#endif

#endif  // DEBUG_H
