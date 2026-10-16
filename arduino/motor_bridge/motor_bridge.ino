/*
 * Motor bridge for nano/motor_control.py (serial backend).
 *
 * The Jetson sends one text line per command over USB (115200 baud):
 *
 *   F <ms> [duty]   forward for <ms> milliseconds
 *   B <ms> [duty]   reverse
 *   L <ms> [duty]   turn left in place
 *   R <ms> [duty]   turn right in place
 *   S               stop now
 *   ?               identify
 *
 * duty is the PWM duty cycle in percent (0-100, default 70). Each command is
 * answered with "OK ..." or "ERR ...". The pulse is timed here, not on the
 * Jetson, so the motors stop even if the Jetson side crashes or the cable is
 * pulled mid-pulse. A new command replaces the one in progress.
 *
 * Pin numbers below are the ones printed next to the L298N_CONTROL port on
 * the car shield (ENA, ENB, IN1..IN4). Check them against your board.
 */

const uint8_t PIN_ENA = 5;   // PWM, left side speed
const uint8_t PIN_ENB = 6;   // PWM, right side speed
const uint8_t PIN_IN1 = 7;   // left side direction
const uint8_t PIN_IN2 = 8;
const uint8_t PIN_IN3 = 9;   // right side direction
const uint8_t PIN_IN4 = 11;

// Flip one of these if that side spins backwards on "F".
const bool INVERT_LEFT = false;
const bool INVERT_RIGHT = true;

const unsigned long MAX_PULSE_MS = 2000;  // longer requests are clamped
const uint8_t DEFAULT_DUTY = 70;

char lineBuf[32];
uint8_t lineLen = 0;
bool lineOverflow = false;

bool moving = false;
unsigned long pulseStart = 0;
unsigned long pulseMs = 0;

// dir: +1 forward, -1 reverse, 0 off
void setSide(uint8_t pinA, uint8_t pinB, int dir, bool invert) {
  if (invert) dir = -dir;
  digitalWrite(pinA, dir > 0 ? HIGH : LOW);
  digitalWrite(pinB, dir < 0 ? HIGH : LOW);
}

void stopMotors() {
  analogWrite(PIN_ENA, 0);
  analogWrite(PIN_ENB, 0);
  setSide(PIN_IN1, PIN_IN2, 0, false);
  setSide(PIN_IN3, PIN_IN4, 0, false);
  moving = false;
}

void startPulse(int leftDir, int rightDir, unsigned long ms, uint8_t duty) {
  setSide(PIN_IN1, PIN_IN2, leftDir, INVERT_LEFT);
  setSide(PIN_IN3, PIN_IN4, rightDir, INVERT_RIGHT);
  uint8_t pwm = (uint8_t)((unsigned int)duty * 255 / 100);
  analogWrite(PIN_ENA, pwm);
  analogWrite(PIN_ENB, pwm);
  pulseStart = millis();
  pulseMs = ms;
  moving = true;
}

void handleLine(char *line) {
  char cmd = line[0];
  if (cmd >= 'a' && cmd <= 'z') cmd -= 32;

  if (cmd == '?') {
    Serial.println(F("EPCAR 1"));
    return;
  }
  if (cmd == 'S') {
    stopMotors();
    Serial.println(F("OK S"));
    return;
  }

  int leftDir, rightDir;
  switch (cmd) {
    case 'F': leftDir = 1;  rightDir = 1;  break;
    case 'B': leftDir = -1; rightDir = -1; break;
    case 'L': leftDir = -1; rightDir = 1;  break;
    case 'R': leftDir = 1;  rightDir = -1; break;
    default:
      Serial.println(F("ERR unknown command"));
      return;
  }

  char *end;
  long ms = strtol(line + 1, &end, 10);
  if (end == line + 1 || ms < 0) {
    Serial.println(F("ERR missing duration"));
    return;
  }
  if ((unsigned long)ms > MAX_PULSE_MS) ms = MAX_PULSE_MS;

  char *dutyStart = end;
  long duty = strtol(dutyStart, &end, 10);
  if (end == dutyStart) duty = DEFAULT_DUTY;
  if (duty < 0) duty = 0;
  if (duty > 100) duty = 100;

  if (ms == 0) {
    stopMotors();
  } else {
    startPulse(leftDir, rightDir, (unsigned long)ms, (uint8_t)duty);
  }
  Serial.print(F("OK "));
  Serial.print(cmd);
  Serial.print(' ');
  Serial.print(ms);
  Serial.print(' ');
  Serial.println(duty);
}

void setup() {
  pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_ENB, OUTPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);
  stopMotors();
  Serial.begin(115200);
  Serial.println(F("EPCAR 1"));
}

void loop() {
  if (moving && millis() - pulseStart >= pulseMs) {
    stopMotors();
  }

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineOverflow) {
        Serial.println(F("ERR line too long"));
      } else if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineOverflow = true;
    }
  }
}
