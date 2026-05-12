#include <Arduino.h>

#include <Adafruit_BNO08x.h>
#include <Adafruit_MCP3421.h>

struct Quaternion { float r, i, j, k; };
struct Vector { float x, y, z; };

Adafruit_MCP3421 mcp;
Adafruit_BNO08x bno08x;

sh2_SensorValue_t sensorValue;

long reportInterval_us = 20000;

Quaternion rotation;
Vector linearAccel;
Vector gyro;
float pressure;

bool hasPressure = false;
bool hasRotation = false;
bool hasAccel = false;
bool hasGyro = false;

bool isStreaming = false;

void printCalibrationStatus(int8_t status = -1);

void setup() {
  Serial.begin(921600);
  while (!Serial) delay(10); 

  Serial.println("INFO: Pen Streamer Starting...");

  Wire.begin();
  Wire.setClock(100000); // standard low speed

  Serial1.begin(921600); // high speed
  Serial1.setRxBufferSize(4096);  // bigger buffer
  
  if (!bno08x.begin_UART(&Serial1)) {
    Serial.println("INFO: Failed to find BNO085 via UART!");
    while (1) { delay(100); }
  } else {
    Serial.println("INFO: BNO085 Found via UART!");
  }

  bno08x.enableReport(SH2_ROTATION_VECTOR, reportInterval_us);
  bno08x.enableReport(SH2_LINEAR_ACCELERATION, reportInterval_us);
  bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, reportInterval_us);

  if (!mcp.begin(0x68, &Wire)) { 
    Serial.println("INFO: Failed to find MCP3421 chip");
    while (1) { delay(100); }
  } else {
    Serial.println("INFO: MCP3421 Found!");
  }

  mcp.setGain(GAIN_1X);
  mcp.setResolution(RESOLUTION_14_BIT); // 240hz=12-bit enob
  mcp.setMode(MODE_CONTINUOUS); 

  Serial.println("INFO: Sensors Ready. Send 'S' to Start.");
}

void loop() {
  // check for commands from the script
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'S') {
      isStreaming = true;
      Serial.println("INFO: Streaming started.");
    } else if (cmd == 'X') {
      isStreaming = false;
      Serial.println("INFO: Streaming stopped.");
    }
  }

  // if the sensor was reset, set the resports
  if (bno08x.wasReset()) {
    Serial.println("INFO: Sensor was reset!");

    bno08x.enableReport(SH2_ROTATION_VECTOR, reportInterval_us);
    bno08x.enableReport(SH2_LINEAR_ACCELERATION, reportInterval_us);
    bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, reportInterval_us);
  }

  if (mcp.isReady()) {
      pressure = (8191.0 - mcp.readADC()) / 8192; // scale to [0, 1) range
      hasPressure = true;
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        rotation.i = sensorValue.un.rotationVector.i;
        rotation.j = sensorValue.un.rotationVector.j;
        rotation.k = sensorValue.un.rotationVector.k;
        rotation.r = sensorValue.un.rotationVector.real;
        hasRotation = true;
        break;

      case SH2_LINEAR_ACCELERATION:
        linearAccel.x = sensorValue.un.linearAcceleration.x;
        linearAccel.y = sensorValue.un.linearAcceleration.y;
        linearAccel.z = sensorValue.un.linearAcceleration.z;
        hasAccel = true;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        gyro.x = sensorValue.un.gyroscope.x;
        gyro.y = sensorValue.un.gyroscope.y;
        gyro.z = sensorValue.un.gyroscope.z;
        hasGyro = true;
        break;
    }
  }

  if (isStreaming && hasRotation && hasAccel && hasGyro && hasPressure) {
      Serial.print(pressure, 6); Serial.print(",");

      Serial.print(linearAccel.x, 6); Serial.print(",");
      Serial.print(linearAccel.y, 6); Serial.print(",");
      Serial.print(linearAccel.z, 6); Serial.print(","); 

      Serial.print(gyro.x, 6); Serial.print(",");
      Serial.print(gyro.y, 6); Serial.print(",");
      Serial.print(gyro.z, 6); Serial.print(",");

      Serial.print(rotation.r, 6); Serial.print(",");
      Serial.print(rotation.i, 6); Serial.print(",");
      Serial.print(rotation.j, 6); Serial.print(",");
      Serial.print(rotation.k, 6);
      Serial.println();

      hasPressure = false;
      hasRotation = false; 
      hasAccel = false; 
      hasGyro = false;
  }
}