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
// Library for creating strings for 64-bit integers in binary, octal, decimal, or hex. 
// Unsigned (uint64_t) can be converted to any case. 
// Signed (int64_t) can be converted to decimal only.
//
// Github: https://github.com/rickkas7/Print64
// License: MIT (can be used in #ifndef __PRINT64_H

#pragma once

#define __PRINT64_H

// Library for creating strings for 64-bit integers in binary, octal, decimal, or hex. 
// Unsigned (uint64_t) can be converted to any case. 
// Signed (int64_t) can be converted to decimal only.
//
// Github: https://github.com/rickkas7/Print64
// License: MIT (can be used in closed-source commercial products)

#include "Particle.h"

/**
 * @brief Convert an unsigned 64-bit integer to a string
 * 
 * @param value The value to convert
 * 
 * @param base The number base. Acceptable values are 2, 8, 10, and 16. Default is 10 (decimal).
 * 
 * @return A String object containing an ASCII representation of the value.
 */
String toString(uint64_t value, unsigned char base = 10);

/**
 * @brief Convert an signed 64-bit integer to a string (ASCII decimal signed integer)
 * 
 * @param value The value to convert
 * 
 * @return A String object containing an ASCII representation of the value (decimal)
 */
String toString(int64_t value);
