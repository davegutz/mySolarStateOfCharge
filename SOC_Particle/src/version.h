/*  Low-energy Bluetooth low-level utilities

27-Dec-2025 	DA Gutz 	Created
// Copyright (C) 2025 - Dave Gutz
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

#pragma once

#include "application.h"  // for String
#define xstr(s) str(s)
#define str(s) #s

const String version = "g20250612a";  // deviceOS@5.6.0
// g20250612 is catch functional Vb failure (soft) and revert voc(soc) for BB. 'a' is nom vsat sp
// g20241006 is fix for amp wrap windup limits
// g20240909 is bug fix for noise sensitivity
// g20240902 is initial release of HI_LO sensor configuration
// g20240704 is HI_LO sensor configuration
// g20240331 is garage modifications and two-stage current sensing
// g20240109 is full testing, e.g. allIn
// g20231111b is Talk function streamline
// g20231111a (tab in GitHub) is g20231111 cleaned up for a rogue Talk.h function that was printing to stdout continuously
