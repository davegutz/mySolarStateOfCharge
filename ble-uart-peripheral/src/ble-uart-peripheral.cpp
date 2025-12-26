#include "Particle.h"

// This example does not require the cloud so you can run it in manual mode or
// normal cloud-connected mode
// SYSTEM_MODE(MANUAL);

const size_t UART_TX_BUF_SIZE = 20;

void onDataReceived(const uint8_t* data, size_t len, const BlePeerDevice& peer, void* context);

// These UUIDs were defined by Nordic Semiconductor and are now the defacto standard for
// UART-like services over BLE. Many apps support the UUIDs now, like the Adafruit Bluefruit app.
const BleUuid serviceUuid("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
const BleUuid rxUuid("6E400002-B5A3-F393-E0A9-E50E24DCCA9E");
const BleUuid txUuid("6E400003-B5A3-F393-E0A9-E50E24DCCA9E");

// This connects USB serial to Bluefruit
BleCharacteristic txCharacteristic("tx", BleCharacteristicProperty::NOTIFY, txUuid, serviceUuid);

// Echo BLE receptions on local Serial
// Without this:  UART won't be available on Bluefruit; and Bluefruit transmissions not echoed to USB Serial
// though still show on Bluefruit due to it's own echo
BleCharacteristic rxCharacteristic("rx", BleCharacteristicProperty::WRITE_WO_RSP, rxUuid, serviceUuid, onDataReceived, NULL);
void onDataReceived(const uint8_t* data, size_t len, const BlePeerDevice& peer, void* context)
{
    Serial.printf("From BLE app::");
    for (size_t ii = 0; ii < len; ii++)
    {
        Serial.write(data[ii]);
    }
}

void setup() {
    Serial.begin();

	BLE.on();
    BLE.addCharacteristic(txCharacteristic);
    BLE.addCharacteristic(rxCharacteristic);  // Without this no UART on Bluefruit app
    BleAdvertisingData data;
    data.appendServiceUUID(serviceUuid);
    BLE.advertise(&data);
}

void loop()
{
  	size_t txLen = 0;
    uint8_t txBuf[UART_TX_BUF_SIZE];
    if (BLE.connected())
    {
    	while(Serial.available() && txLen < UART_TX_BUF_SIZE) {
            txBuf[txLen++] = Serial.read();
        }
    }
    if (txLen > 0)
    {
        txCharacteristic.setValue(txBuf, txLen);
    }
}
