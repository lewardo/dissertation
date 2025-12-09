#include <Arduino.h>
#include <Adafruit_BNO08x.h>
#include <Adafruit_MCP3421.h>

// --- Helper Structs ---
struct Quaternion { float i, j, k, real; };
struct Vector { float x, y, z; };

Adafruit_MCP3421 mcp;
Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;

// 50Hz Report Interval (20000us)
long reportInterval_us = 20000;

// --- Timer & Flags ---
unsigned long lastSendTime = 0;
unsigned long sendInterval = 20; 

Quaternion rotationQuat;
Vector linearAccel;
Vector gyro;
float pressure;

bool hasPressure = false;
bool hasRotation = false;
bool hasAccel = false;
bool hasGyro = false;
bool isStreaming = false;

// Forward declarations
Quaternion quat_conjugate(Quaternion q);
Quaternion quat_multiply(Quaternion q1, Quaternion q2);
Vector quat_rotateVector(Quaternion q, Vector v);
void printCalibrationStatus(int8_t status = -1);

void setup() {
  Serial.begin(921600);
  while (!Serial) delay(10); 

  Serial.println("INFO: Pen Streamer Starting...");

  Wire.begin();
  Wire.setClock(100000);

  Serial1.begin(921600);
  Serial1.setRxBufferSize(4096); 
  
  if (!bno08x.begin_UART(&Serial1)) {
    Serial.println("INFO: Failed to find BNO085 via UART!");
    while (1) { delay(100); }
  }
  Serial.println("INFO: BNO085 Found via UART!");

  // Enable reports
  bno08x.enableReport(SH2_ROTATION_VECTOR, reportInterval_us);
  bno08x.enableReport(SH2_LINEAR_ACCELERATION, reportInterval_us);
  bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, reportInterval_us);

    // Begin can take an optional address and Wire interface
  if (!mcp.begin(0x68, &Wire)) { 
    Serial.println("INFO: Failed to find MCP3421 chip");
    while (1) { delay(100); }
  }

  Serial.println("INFO: MCP3421 Found!");

  mcp.setGain(GAIN_1X);
  mcp.setResolution(RESOLUTION_14_BIT); // 240 SPS (12-bit)
  mcp.setMode(MODE_CONTINUOUS); // Options: MODE_CONTINUOUS, MODE_ONE_SHOT

  Serial.println("INFO: Sensors Ready. Send 'S' to Start.");
}

void loop() {
  // 1. Handle Serial Commands
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

  // 2. Reset Handling
  if (bno08x.wasReset()) {
    Serial.println("INFO: Sensor was reset!");

    bno08x.enableReport(SH2_ROTATION_VECTOR, reportInterval_us);
    bno08x.enableReport(SH2_LINEAR_ACCELERATION, reportInterval_us);
    bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, reportInterval_us);
  }

  if (mcp.isReady()) {
      pressure = (8191.0 - mcp.readADC()) / 8192; // Read ADC value
      hasPressure = true;
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        rotationQuat.i = sensorValue.un.rotationVector.i;
        rotationQuat.j = sensorValue.un.rotationVector.j;
        rotationQuat.k = sensorValue.un.rotationVector.k;
        rotationQuat.real = sensorValue.un.rotationVector.real;
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
    static long lastSend = millis();

  if (isStreaming && hasRotation && hasAccel && hasGyro && hasPressure) {
      // Quaternion q_inverse = quat_conjugate(rotationQuat);
      // Vector bodyAccel = {linearAccel.x, linearAccel.y, linearAccel.z};
      // Vector bodyGyro = {gyro.x, gyro.y, gyro.z};
      
      // Vector worldAccel = quat_rotateVector(q_inverse, bodyAccel);
      // Vector worldGyro = quat_rotateVector(q_inverse, bodyGyro);

      // Serial.print(pressure, 6); Serial.print(",");

      // Serial.print(linearAccel.x, 6); Serial.print(",");
      // Serial.print(linearAccel.y, 6); Serial.print(",");
      // Serial.print(linearAccel.z, 6); Serial.print(","); 

      // Serial.print(gyro.x, 6); Serial.print(",");
      // Serial.print(gyro.y, 6); Serial.print(",");
      // Serial.print(gyro.z, 6); Serial.print(",");

      // Serial.print(rotationQuat.real, 6); Serial.print(",");
      // Serial.print(rotationQuat.i, 6); Serial.print(",");
      // Serial.print(rotationQuat.j, 6); Serial.print(",");
      // Serial.print(rotationQuat.k, 6);

      Serial.print(millis() - lastSend);
      lastSend = millis();
      Serial.println();

      hasPressure = false;
      hasRotation = false; 
      hasAccel = false; 
      hasGyro = false;
  }
}

// Quaternion quat_conjugate(Quaternion q) {
//   return {-q.i, -q.j, -q.k, q.real};
// }

// Quaternion quat_multiply(Quaternion q1, Quaternion q2) {
//   Quaternion q_out;
//   q_out.real = q1.real * q2.real - q1.i * q2.i - q1.j * q2.j - q1.k * q2.k;
//   q_out.i    = q1.real * q2.i + q1.i * q2.real + q1.j * q2.k - q1.k * q2.j;
//   q_out.j    = q1.real * q2.j - q1.i * q2.k + q1.j * q2.real + q1.k * q2.i;
//   q_out.k    = q1.real * q2.k + q1.i * q2.j - q1.j * q2.i + q1.k * q2.real;
//   return q_out;
// }

// Vector quat_rotateVector(Quaternion q, Vector v) {
//   Quaternion v_quat = {v.x, v.y, v.z, 0.0f};
//   Quaternion q_conj = quat_conjugate(q);
//   Quaternion v_rotated_quat = quat_multiply(q, quat_multiply(v_quat, q_conj));
//   return {v_rotated_quat.i, v_rotated_quat.j, v_rotated_quat.k};
// }