#include <blynk.h>
#include <cmath>

char auth[] = "C2QLGdvd4GIYo-2EfeXDKFc4OZFK0zOR";

const int pinServo1 = D2;
const int pinServo2 = D3;
const int laserSig = A3;
const int ultraIn = D4;
const int ultraOut = D5;
const int dir = D6;
const int step = D7;

// Stepper vals
const int stepPerRev = 400;
const double degrees = 360.0 / stepPerRev;

// Laser Vals
boolean laserOn = false;

// Time Constraints
double SPEED_SOUND_CM_ROOM_TEMP_FAHR = 0.03444;

// Variable Constraints
double diag1 = 0;
double angle1 = 0;
double stepAngle1 = 0;
double diag2 = 0;
double angle2 = 0;
double stepAngle2 = 0;
double diag3 = 0;
double angle3 = 0;
double stepAngle3 = 0;
double diag4 = 0;
double angle4 = 0;
double stepAngle4 = 0;
double currAngle = 0;
double currStepAngle = 0;
double p1x = 0;
double p1y = 0;
double p1z = 0;
double p2x = 0;
double p2y = 0;
double p2z = 0;
double p3x = 0;
double p3y = 0;
double p3z = 0;
double p4x = 0;
double p4y = 0;
double p4z = 0;
int count;

// Set up terminal
WidgetTerminal terminal(V7);

// Set up Servo Motors
Servo servo1;
Servo servo2;

void setup()
{
  delay(5000);
  Blynk.begin(auth);

  pinMode(pinServo1, OUTPUT);
  pinMode(pinServo2, OUTPUT);
  pinMode(laserSig, OUTPUT);
  pinMode(ultraIn, INPUT);
  pinMode(ultraOut, OUTPUT);
  pinMode(dir, OUTPUT);
  pinMode(step, OUTPUT);

  Serial.begin(9600);
  servo1.attach(pinServo1);
  servo2.attach(pinServo2);

  // Reset board
  Blynk.virtualWrite(V0, 0);
  Blynk.virtualWrite(V1, 0);
  Blynk.virtualWrite(V2, 0);
  Blynk.virtualWrite(V3, 0);
  Blynk.virtualWrite(V4, 0);
  Blynk.virtualWrite(V5, 0);
  Blynk.virtualWrite(V6, 0);
  Blynk.virtualWrite(V7, 0);
  Blynk.virtualWrite(V8, 0);
  Blynk.virtualWrite(V9, 0);
  Blynk.virtualWrite(V10, 0);

  // Provide Instructions
  startup();
}

void loop()
{
  Blynk.run();
}

// Handle Stepper CW movement
BLYNK_WRITE(V0)
{
  // Set spin direction
  digitalWrite(dir, HIGH);

  // Move a bit
  digitalWrite(step, HIGH);
  delayMicroseconds(500);
  digitalWrite(step, LOW);
  delayMicroseconds(200);
  currStepAngle = currStepAngle - degrees;
  if (currStepAngle < 360)
  {
    currStepAngle = currStepAngle + 360;
  }

  Serial.println("Degree CW: " + String(currStepAngle));
}
// Handle Stepper CCW movement
BLYNK_WRITE(V1)
{
  // Set spin direction
  digitalWrite(dir, LOW);

  // Move a bit
  digitalWrite(step, HIGH);
  delayMicroseconds(500);
  digitalWrite(step, LOW);
  delayMicroseconds(200);
  currStepAngle = currStepAngle + degrees;
  if (currStepAngle > 360)
  {
    currStepAngle = currStepAngle - 360.0;
  }

  Serial.println("Degree CCW: " + String(currStepAngle));
}

// Handle Servo
BLYNK_WRITE(V8)
{
  int deg = param.asInt();

  if (deg > 95 && deg < 175)
  {
    servo1.write(deg);
    delay(100);
    servo2.write(deg);
  }
  currAngle = deg;
}

// Handle intake measurements position 1
BLYNK_WRITE(V2)
{
  double total = 0;
  // Take 5 measurements
  for (int i = 0; i < 5; i++)
  {
    digitalWrite(ultraOut, LOW);
    delayMicroseconds(2);
    digitalWrite(ultraOut, HIGH);
    delayMicroseconds(10);
    digitalWrite(ultraOut, LOW);
    double timeRoundTrip = pulseIn(ultraIn, HIGH);

    total += timeRoundTrip * SPEED_SOUND_CM_ROOM_TEMP_FAHR / 2.54;
    delay(500);
  }

  // Take Average
  diag1 = total / 5;
  angle1 = currAngle;
  Serial.println("Measureing first: " + String(diag1));
}

// Handle intake measurements position 2
BLYNK_WRITE(V3)
{
  double total = 0;
  // Take 5 measurements
  for (int i = 0; i < 5; i++)
  {
    digitalWrite(ultraOut, LOW);
    delayMicroseconds(2);
    digitalWrite(ultraOut, HIGH);
    delayMicroseconds(10);
    digitalWrite(ultraOut, LOW);
    double timeRoundTrip = pulseIn(ultraIn, HIGH);

    total += timeRoundTrip * SPEED_SOUND_CM_ROOM_TEMP_FAHR / 2.54;
    delay(500);
  }

  // Take Average
  diag2 = total / 5;
  angle2 = currAngle;
}

// Handle intake measurements position 3
BLYNK_WRITE(V4)
{
  double total = 0;
  // Take 5 measurements
  for (int i = 0; i < 5; i++)
  {
    digitalWrite(ultraOut, LOW);
    delayMicroseconds(2);
    digitalWrite(ultraOut, HIGH);
    delayMicroseconds(10);
    digitalWrite(ultraOut, LOW);
    double timeRoundTrip = pulseIn(ultraIn, HIGH);

    total += timeRoundTrip * SPEED_SOUND_CM_ROOM_TEMP_FAHR / 2.54;
    delay(500);
  }

  // Take Average
  diag3 = total / 5;
  angle3 = currAngle;
}

// Handle intake measurements position 4
BLYNK_WRITE(V5)
{
  double total = 0;
  // Take 5 measurements
  for (int i = 0; i < 5; i++)
  {
    digitalWrite(ultraOut, LOW);
    delayMicroseconds(2);
    digitalWrite(ultraOut, HIGH);
    delayMicroseconds(10);
    digitalWrite(ultraOut, LOW);
    double timeRoundTrip = pulseIn(ultraIn, HIGH);

    total += timeRoundTrip * SPEED_SOUND_CM_ROOM_TEMP_FAHR / 2.54;
    delay(500);
  }

  // Take Average
  diag4 = total / 5;
  angle4 = currAngle;
}

// Handle turn off/on laser
BLYNK_WRITE(V6)
{
  if (laserOn)
  {
    digitalWrite(laserSig, LOW);
    laserOn = false;
  }
  else
  {
    digitalWrite(laserSig, HIGH);
    laserOn = true;
  }
}

// Take measurement button
BLYNK_WRITE(V7)
{
  if (count == 0)
  {
    p1x = param.asDouble();
    Serial.println(String(p1x));
  }
  else if (count == 1)
  {
    p1y = param.asDouble();
  }
  else if (count == 2)
  {
    p1z = param.asDouble();
  }
  else if (count == 3)
  {
    p2x = param.asDouble();
  }
  else if (count == 4)
  {
    p2y = param.asDouble();
  }
  else if (count == 5)
  {
    p2z = param.asDouble();
  }
  else if (count == 6)
  {
    p3x = param.asDouble();
  }
  else if (count == 7)
  {
    p3y = param.asDouble();
  }
  else if (count == 8)
  {
    p3z = param.asDouble();
  }
  else if (count == 9)
  {
    p4x = param.asDouble();
  }
  else if (count == 10)
  {
    p4y = param.asDouble();
  }
  else if (count == 11)
  {
    p4z = param.asDouble();
  }
  else
  {
    terminal.println("Something got messed up, restart machine");
  }
  count++;
}

// Calulate
BLYNK_WRITE(V10)
{
  // Solve for diagonal line on object
  double diagFlat1 = flatDiagLine(diag1, angle1);
  double diagFlat2 = flatDiagLine(diag2, angle2);
  double diagFlat3 = flatDiagLine(diag3, angle3);
  double diagFlat4 = flatDiagLine(diag4, angle4);

  // Solve for Z
  double z1 = solveZ(diag1, angle1);
  double z2 = solveZ(diag2, angle2);
  double z3 = solveZ(diag3, angle3);
  double z4 = solveZ(diag4, angle4);

  // Solve for x
  double x1 = solveX(diagFlat1, stepAngle1);
  double x2 = solveX(diagFlat2, stepAngle2);
  double x3 = solveX(diagFlat3, stepAngle3);
  double x4 = solveX(diagFlat4, stepAngle4);

  // Solve for y
  double y1 = solveY(diagFlat1, stepAngle1);
  double y2 = solveY(diagFlat2, stepAngle2);
  double y3 = solveY(diagFlat3, stepAngle3);
  double y4 = solveY(diagFlat4, stepAngle4);
}

// Calibrate Stepper
BLYNK_WRITE(V9)
{
  currStepAngle = 0;
}

void startup()
{
  terminal.println(F("Resetting Servo Motors"));
  servo1.write(175);
  delay(500);
  servo2.write(175);
  terminal.flush();
  terminal.clear();
  terminal.println(F("Move stepper until parallel with Robot y-axis"));
  delay(3000);
  terminal.flush();
  terminal.println(F("Enter position coordinates: "));
  delay(2000);
  terminal.flush();
  terminal.clear();
}

double flatDiagLine(double dist, double angleTilt)
{
  double tiltDeg = 180 / M_PI * angleTilt;
  return dist * sin(tiltDeg);
}

double solveX(double hyp, double stepAngle)
{
  double stepDeg = 180 / M_PI * stepAngle;
  return hyp * cos(stepDeg);
}

double solveY(double hyp, double stepAngle)
{
  double stepDeg = 180 / M_PI * stepAngle;
  return hyp * sin(stepDeg);
}

double solveZ(double dist, double angleTilt)
{
  double tiltDeg = 180 / M_PI * angleTilt;
  return dist * cos(tiltDeg);
}
