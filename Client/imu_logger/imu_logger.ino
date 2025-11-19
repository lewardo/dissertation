/**
 * BNO085 50Hz World-Frame Streamer (UART Version)
 * Target: ESP32-S3
 *
 * CONNECTION RE-MAPPING (Qwiic/I2C Cable):
 * The ESP32-S3 allows us to map Serial1 to the I2C pins.
 *
 * - ESP32 SDA Pin -> Acts as RX (Connects to Sensor TX)
 * - ESP32 SCL Pin -> Acts as TX (Connects to Sensor RX)
 *
 * HARDWARE REQUIREMENT:
 * You MUST bridge the PS0 jumper on the BNO085 to enable UART-SHTP mode.
 */

#include <Arduino.h>
#include <Adafruit_BNO08x.h>

// --- Helper Structs ---
struct Quaternion { float i, j, k, real; };
struct Vector { float x, y, z; };

// --- BNO08x Setup ---
Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;

void printCalibrationStatus(int8_t = -1);

// 50Hz Report Interval (20000us)
long reportInterval_us = 20000;

// --- Timer & Flags ---
unsigned long lastSendTime = 0;
unsigned long sendInterval = 20;
Quaternion rotationQuat;
Vector linearAccel;
Vector gyro;
bool hasRotation = false;
bool hasAccel = false;
bool hasGyro = false;
bool calibReady = false;
bool isStreaming = false;

// =========================================================================
//     SETUP
// =========================================================================
void setup() {
  // 1. specific for ESP32-S3 USB-CDC (Native USB)
  // If you use the "USB OTG" port, 'Serial' refers to the internal USB.
  Serial.begin(115200);
  
  // Give a moment for USB to catch up (optional, good for debugging)
  while (!Serial) delay(10); 

  Serial.println("INFO: BNO085 UART Streamer Starting...");

  // 2. Initialize the Hardware Serial on the "I2C" pins
  // Format: begin(baud, config, rx_pin, tx_pin)
  // We use 3Mbps (3000000) which BNO085 supports, or 115200 for safety.
  // Let's start with 115200 to ensure connection, though 3M is better for 
  // high bandwidth. The library usually negotiates.
  Serial1.begin(115200, SERIAL_8N1, SDA, SCL);

  // 3. Initialize BNO using UART
  // Note: We pass -1 for reset pin if you are using a standard 4-wire Qwiic cable
  // (Power cycle is the only way to reset in this config)
  if (!bno08x.begin_UART(&Serial1)) {
    Serial.println("INFO: Failed to find BNO085 via UART!");
    Serial.println("INFO: Did you bridge the PS0 jumper?");
    while (1) { delay(100); }
  }
  Serial.println("INFO: BNO085 Found via UART!");

  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, reportInterval_us)) {
    Serial.println("INFO: Could not enable Rotation Vector");
  }

  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, reportInterval_us)) {
    Serial.println("INFO: Could not enable Linear Acceleration");
  }

  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, reportInterval_us)) {
    Serial.println("INFO: Could not enable Gyroscope");
  }

  Serial.println("INFO: Sensor Ready. Send 'S' to Start.");
}

// =========================================================================
//     LOOP
// =========================================================================
void loop() {
  // Check for commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'S') {
      isStreaming = true;
      lastSendTime = millis();
      Serial.println("INFO: Streaming started.");
      printCalibrationStatus();
    } else if (cmd == 'X') {
      isStreaming = false;
      Serial.println("INFO: Streaming stopped.");
    }
  }

  // Poll Sensor
  if (bno08x.wasReset()) {
    Serial.println("INFO: Sensor was reset!");
    bno08x.enableReport(SH2_ROTATION_VECTOR, reportInterval_us);
    bno08x.enableReport(SH2_LINEAR_ACCELERATION, reportInterval_us);
    bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, reportInterval_us);
  }

  while (!hasRotation || !hasAccel || !hasGyro) {
    bno08x.getSensorEvent(&sensorValue);
    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        rotationQuat.i = sensorValue.un.rotationVector.i;
        rotationQuat.j = sensorValue.un.rotationVector.j;
        rotationQuat.k = sensorValue.un.rotationVector.k;
        rotationQuat.real = sensorValue.un.rotationVector.real;
        printCalibrationStatus(sensorValue.status);
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

  // Stream Data
  if (isStreaming && (millis() - lastSendTime >= sendInterval)) {
    if (hasRotation && hasAccel && hasGyro && calibReady) {

      Quaternion q_inverse = quat_conjugate(rotationQuat);
      
      Vector bodyAccel = {linearAccel.x, linearAccel.y, linearAccel.z};
      Vector bodyGyro = {gyro.x, gyro.y, gyro.z};
      
      Vector worldAccel = quat_rotateVector(q_inverse, bodyAccel);
      Vector worldGyro = quat_rotateVector(q_inverse, bodyGyro);

      Serial.print(worldAccel.x, 4); Serial.print(",");
      Serial.print(worldAccel.y, 4); Serial.print(",");
      Serial.print(-worldAccel.z, 4); Serial.print(",");
      Serial.print(worldGyro.x, 4); Serial.print(",");
      Serial.print(worldGyro.y, 4); Serial.print(",");
      Serial.print(-worldGyro.z, 4); Serial.println();
    }

    lastSendTime += sendInterval;
    hasRotation = false; hasAccel = false; hasGyro = false;
  }
}

// =========================================================================
//     HELPERS & MATH (Same as before)
// =========================================================================

void printCalibrationStatus(int8_t status) {
  static uint8_t lastStatus = 0;
  uint8_t calStatus = status & 0x03;
  if (status < 0 || calStatus != lastStatus) {
    if (status < 0) {
      calStatus = lastStatus;
    } else {
      lastStatus = calStatus;
    }

    Serial.print("CAL: ");
    switch (calStatus) {
      case 0: 
      case 1: 
      case 2:
        Serial.println("Unreliable Accuracy"); 
        calibReady = false;
        break;

      case 3: 
        Serial.println("High Accuracy"); 
        calibReady = true;
        break;
    }
  }
}

Quaternion quat_conjugate(Quaternion q) {
  return {-q.i, -q.j, -q.k, q.real};
}

Quaternion quat_multiply(Quaternion q1, Quaternion q2) {
  Quaternion q_out;
  q_out.real = q1.real * q2.real - q1.i * q2.i - q1.j * q2.j - q1.k * q2.k;
  q_out.i    = q1.real * q2.i + q1.i * q2.real + q1.j * q2.k - q1.k * q2.j;
  q_out.j    = q1.real * q2.j - q1.i * q2.k + q1.j * q2.real + q1.k * q2.i;
  q_out.k    = q1.real * q2.k + q1.i * q2.j - q1.j * q2.i + q1.k * q2.real;
  return q_out;
}

Vector quat_rotateVector(Quaternion q, Vector v) {
  Quaternion v_quat = {v.x, v.y, v.z, 0.0f};
  Quaternion q_conj = quat_conjugate(q);
  Quaternion v_rotated_quat = quat_multiply(q, quat_multiply(v_quat, q_conj));
  return {v_rotated_quat.i, v_rotated_quat.j, v_rotated_quat.k};
}