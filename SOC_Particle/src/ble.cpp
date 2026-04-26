/*  Low-energy Bluetooth low-level utilities

27-Dec-2025 	DA Gutz 	Created
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

*/

#include "ble.h"
#include "serial.h"
#include "command.h"
#include "debug.h"

extern CommandPars cp;  // Various parameters to be static at system level
extern BleCharacteristic txCharacteristic;

// Blocking BLE write
void bleSendChunked(BleCharacteristic& chr, const uint8_t* data, size_t length)
{
  size_t offset = 0;
  while (offset < length) {
      size_t chunkLen = min(BLE_CHUNK_SIZE, length - offset);
      chr.setValue(data + offset, chunkLen);
      offset += chunkLen;
      delay(5);  // allow notify queue to drain
  }
}

// BLE receive
void onBLE_DataReceived(const uint8_t* data, size_t len, const BlePeerDevice& peer, void* context)
{
  size_t ii = 0;

  // Logic for bootup:  solitary character treated as possible boot command (e.g. y/n)
  if ( len==1 )
  {
    cp.ble_first_char = data[0];
  }
  else
    cp.ble_first_char = '\0';


  // Validate input to Serial only
  Serial.printf("from BLE::");
  for (; ii < len; ii++)
  {
    Serial.write(data[ii]);
  }
  Serial.printf("\n");

  // Parse input
  static String serial_str = "";
  static bool serial_ready = false;
  // Each pass try to complete input from avaiable
  ii = 0;
  while ( !serial_ready && ( ii < len ) )
  {
    char in_char = (char) data[ii++];  // get the new byte

    // Intake
    // if the incoming character to finish, add a ';' and set flags so the main loop can do something about it:
    if ( is_finished(in_char) )
    {
        serial_str += ';';
        serial_ready = true;
        break;
    }
    else if ( in_char == '\r' )
        Serial.printf("\n");  // scroll user terminal
    else if ( in_char == '\b' && serial_str.length() )
    {
        Serial.printf("\b \b");  // scroll user terminal
        serial_str.remove(serial_str.length() -1 );  // backspace
    }
    else
        serial_str += in_char;  // process new valid character
  }

  // Pass info to inp_str
  if ( serial_ready )
  {
    if ( !cp.inp_token )
    {
        cp.inp_token = true;
        add_verify(&cp.inp_str, serial_str);
        Serial.printf("add_verified %s\n", serial_str.c_str());
        serial_ready = false;
        cp.inp_token = false;
        serial_str = "";
    }     
  }
}
