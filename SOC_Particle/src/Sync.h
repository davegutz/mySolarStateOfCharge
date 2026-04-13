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
//
// 17-Feb-2021  Dave Gutz   Create

#pragma once

// Duct Sim Class
class Sync
{
public:
  // Constructors
  Sync(void);
  Sync(uint64_t delay);
  // Functions
  bool update(bool reset, uint64_t now, bool andCheck);
  bool update(uint64_t now, bool reset, bool andCheck);
  bool update(uint64_t now, bool reset);
  bool updateN(uint64_t now, bool reset, bool orCheck);
  uint64_t delay() { return(delay_); }
  void delay(uint64_t new_delay) { delay_ = new_delay; updateTimeInput_ = float(delay_)/1000.f; }
  void delay(uint64_t new_delay, uint64_t now) { delay_ = new_delay; updateTimeInput_ = float(delay_)/1000.f; last_ = now; }
  uint64_t last() { return(last_); }
  bool stat() { return(stat_); }
  uint64_t updateDiff() { return(updateDiff_); }
  double updateTime() { return(updateTime_); }
  double updateTimeInput() { return(updateTimeInput_); }
  uint64_t now() { return(now_); }
private:
  uint64_t delay_;
  uint64_t last_;
  uint64_t now_;
  bool stat_;
  uint64_t updateDiff_;
  double updateTime_;
  double updateTimeInput_;
};