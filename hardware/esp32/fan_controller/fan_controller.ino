/*
 * SRACE v2 — ESP32 Fan + LED Controller
 * 
 * Controls 4 DC fans via 2× L298N motor drivers AND 4 zone LEDs
 * over USB serial. Receives commands from Python optimizer.
 * 
 * Wiring (ESP32 DevKit v1 30-pin):
 * ─────────────────────────────────────────────────
 *   L298N #1 (Fan 1 + Fan 2):
 *     IN1 → D2   (Fan 1 forward)
 *     IN2 → D4   (Fan 1 reverse — always LOW)
 *     IN3 → D18  (Fan 2 forward)
 *     IN4 → D19  (Fan 2 reverse — always LOW)
 *     ENA → jumpered HIGH (always enabled)
 *     ENB → jumpered HIGH (always enabled)
 *
 *   L298N #2 (Fan 3 + Fan 4):
 *     IN1 → D22  (Fan 3 forward)
 *     IN2 → D23  (Fan 3 reverse — always LOW)
 *     IN3 → D12  (Fan 4 forward)
 *     IN4 → D14  (Fan 4 reverse — always LOW)
 *     ENA → jumpered HIGH (always enabled)
 *     ENB → jumpered HIGH (always enabled)
 *
 *   LEDs (with 220Ω–330Ω resistor to GND each):
 *     LED_Z1  → D25
 *     LED_Z4  → D26
 *     LED_Z7  → D27
 *     LED_Z12 → D33
 *
 *   Power: L298N 12V input from external supply, GND shared with ESP32.
 * ─────────────────────────────────────────────────
 * 
 * Fan Protocol (115200 baud serial):
 *   Send:    "1,0,1,1\n"          → Fan1 ON, Fan2 OFF, Fan3 ON, Fan4 ON
 *   Reply:   "OK:1,0,1,1"         → Confirmation echo
 *   Send:    "STATUS\n"           → Query current fan states
 *   Reply:   "STATE:1,0,1,1"      → Current fan states
 *   Send:    "ALLOFF\n"           → Emergency stop (fans only)
 *   Reply:   "OK:0,0,0,0"         → All fans off
 *
 * LED Protocol (same serial line):
 *   Send:    "LED:1,0,1,0\n"      → LED Z1 ON, Z4 OFF, Z7 ON, Z12 OFF
 *   Reply:   "LEDOK:1,0,1,0"      → Confirmation echo
 *   Send:    "LEDSTATUS\n"        → Query current LED states
 *   Reply:   "LEDSTATE:1,0,1,0"   → Current LED states
 *   Send:    "LEDOFF\n"           → All LEDs off
 *   Reply:   "LEDOK:0,0,0,0"      → All LEDs off
 *
 * Common:
 *   Startup: "READY"              → ESP32 booted and ready
 *   Send:    "PING\n"             → Keepalive
 *   Reply:   "PONG"
 *
 * Safety: 30-second watchdog — if no command of ANY kind received
 *         within 30s, ALL fans AND LEDs turn OFF automatically.
 */

// ═══════════════════════════════════════════════════
//  FAN PIN DEFINITIONS — 2× L298N, 4 fans (UNCHANGED)
// ═══════════════════════════════════════════════════

// Each fan uses a pin pair: [forward_pin, reverse_pin]
// Fan ON  = forward HIGH, reverse LOW
// Fan OFF = both LOW

// L298N Board #1
const int FAN1_FWD = 2;    // IN1 → Fan 1 forward
const int FAN1_REV = 4;    // IN2 → Fan 1 reverse (always LOW)
const int FAN2_FWD = 18;   // IN3 → Fan 2 forward
const int FAN2_REV = 19;   // IN4 → Fan 2 reverse (always LOW)

// L298N Board #2
const int FAN3_FWD = 22;   // IN1 → Fan 3 forward
const int FAN3_REV = 23;   // IN2 → Fan 3 reverse (always LOW)
const int FAN4_FWD = 12;   // IN3 → Fan 4 forward
const int FAN4_REV = 14;   // IN4 → Fan 4 reverse (always LOW)

const int NUM_FANS = 4;

// Pin arrays for easy iteration
const int FWD_PINS[NUM_FANS] = {FAN1_FWD, FAN2_FWD, FAN3_FWD, FAN4_FWD};
const int REV_PINS[NUM_FANS] = {FAN1_REV, FAN2_REV, FAN3_REV, FAN4_REV};

// ═══════════════════════════════════════════════════
//  LED PIN DEFINITIONS — 4 zone LEDs (NEW)
// ═══════════════════════════════════════════════════

// Each LED: GPIO → 220Ω resistor → LED → GND
// ON = HIGH, OFF = LOW (binary only, no PWM)

const int LED_Z1  = 25;   // Zone 1 LED
const int LED_Z4  = 26;   // Zone 4 LED
const int LED_Z7  = 27;   // Zone 7 LED
const int LED_Z12 = 33;   // Zone 12 LED

const int NUM_LEDS = 4;

const int LED_PINS[NUM_LEDS] = {LED_Z1, LED_Z4, LED_Z7, LED_Z12};

// ═══════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════

int fanStates[NUM_FANS] = {0, 0, 0, 0};    // Current fan states
int ledStates[NUM_LEDS] = {0, 0, 0, 0};    // Current LED states (NEW)
unsigned long lastCommandTime = 0;           // For watchdog
const unsigned long WATCHDOG_MS = 30000;     // 30 second timeout

// ═══════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  
  // Configure fan pins as outputs (UNCHANGED)
  for (int i = 0; i < NUM_FANS; i++) {
    pinMode(FWD_PINS[i], OUTPUT);
    pinMode(REV_PINS[i], OUTPUT);
    digitalWrite(FWD_PINS[i], LOW);
    digitalWrite(REV_PINS[i], LOW);
  }

  // Configure LED pins as outputs (NEW)
  for (int i = 0; i < NUM_LEDS; i++) {
    pinMode(LED_PINS[i], OUTPUT);
    digitalWrite(LED_PINS[i], LOW);
  }

  delay(500);  // Let serial settle
  Serial.println("READY");
  lastCommandTime = millis();
}

// ═══════════════════════════════════════════════════
//  FAN CONTROL (UNCHANGED)
// ═══════════════════════════════════════════════════

void setFan(int fanIndex, int state) {
  if (fanIndex < 0 || fanIndex >= NUM_FANS) return;
  
  fanStates[fanIndex] = state;
  
  if (state) {
    // Fan ON: forward HIGH, reverse LOW
    digitalWrite(FWD_PINS[fanIndex], HIGH);
    digitalWrite(REV_PINS[fanIndex], LOW);
  } else {
    // Fan OFF: both LOW
    digitalWrite(FWD_PINS[fanIndex], LOW);
    digitalWrite(REV_PINS[fanIndex], LOW);
  }
}

void allFansOff() {
  for (int i = 0; i < NUM_FANS; i++) {
    setFan(i, 0);
  }
}

// ═══════════════════════════════════════════════════
//  LED CONTROL (NEW)
// ═══════════════════════════════════════════════════

void setLed(int ledIndex, int state) {
  if (ledIndex < 0 || ledIndex >= NUM_LEDS) return;
  
  ledStates[ledIndex] = state ? 1 : 0;
  digitalWrite(LED_PINS[ledIndex], ledStates[ledIndex] ? HIGH : LOW);
}

void allLedsOff() {
  for (int i = 0; i < NUM_LEDS; i++) {
    setLed(i, 0);
  }
}

// ═══════════════════════════════════════════════════
//  SERIAL COMMAND PARSER
// ═══════════════════════════════════════════════════

void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.length() == 0) return;
  
  lastCommandTime = millis();  // Reset watchdog on ANY valid command
  
  // ── PING keepalive ──
  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  // ────────────────────────────────────────────────
  //  LED COMMANDS (NEW — checked first via prefix)
  // ────────────────────────────────────────────────

  // ── LED state command: "LED:1,0,1,0" ──
  if (cmd.startsWith("LED:")) {
    String payload = cmd.substring(4);  // strip "LED:" prefix
    payload.trim();
    
    int states[NUM_LEDS];
    int idx = 0;
    int start = 0;
    
    for (int i = 0; i <= (int)payload.length(); i++) {
      if (i == (int)payload.length() || payload[i] == ',') {
        String val = payload.substring(start, i);
        val.trim();
        states[idx] = val.toInt();
        idx++;
        start = i + 1;
        if (idx >= NUM_LEDS) break;
      }
    }
    
    if (idx < NUM_LEDS) {
      Serial.print("ERR:LED expected ");
      Serial.print(NUM_LEDS);
      Serial.print(" values, got ");
      Serial.println(idx);
      return;
    }
    
    for (int i = 0; i < NUM_LEDS; i++) {
      setLed(i, states[i] ? 1 : 0);
    }
    
    // Echo confirmation
    Serial.print("LEDOK:");
    for (int i = 0; i < NUM_LEDS; i++) {
      Serial.print(ledStates[i]);
      if (i < NUM_LEDS - 1) Serial.print(",");
    }
    Serial.println();
    return;
  }

  // ── LEDSTATUS query ──
  if (cmd == "LEDSTATUS") {
    Serial.print("LEDSTATE:");
    for (int i = 0; i < NUM_LEDS; i++) {
      Serial.print(ledStates[i]);
      if (i < NUM_LEDS - 1) Serial.print(",");
    }
    Serial.println();
    return;
  }

  // ── LEDOFF — all LEDs off ──
  if (cmd == "LEDOFF") {
    allLedsOff();
    Serial.println("LEDOK:0,0,0,0");
    return;
  }

  // ────────────────────────────────────────────────
  //  FAN COMMANDS (UNCHANGED — original logic below)
  // ────────────────────────────────────────────────

  // ── STATUS query ──
  if (cmd == "STATUS") {
    Serial.print("STATE:");
    for (int i = 0; i < NUM_FANS; i++) {
      Serial.print(fanStates[i]);
      if (i < NUM_FANS - 1) Serial.print(",");
    }
    Serial.println();
    return;
  }
  
  // ── ALLOFF emergency stop ──
  if (cmd == "ALLOFF") {
    allFansOff();
    Serial.println("OK:0,0,0,0");
    return;
  }
  
  // ── Fan state command: "1,0,1,1" ──
  int states[NUM_FANS];
  int idx = 0;
  int start = 0;
  
  for (int i = 0; i <= (int)cmd.length(); i++) {
    if (i == (int)cmd.length() || cmd[i] == ',') {
      String val = cmd.substring(start, i);
      val.trim();
      states[idx] = val.toInt();
      idx++;
      start = i + 1;
      if (idx >= NUM_FANS) break;
    }
  }
  
  // Validate we got exactly 4 values
  if (idx < NUM_FANS) {
    Serial.print("ERR:expected ");
    Serial.print(NUM_FANS);
    Serial.print(" values, got ");
    Serial.println(idx);
    return;
  }
  
  // Apply states
  for (int i = 0; i < NUM_FANS; i++) {
    setFan(i, states[i] ? 1 : 0);
  }
  
  // Echo confirmation
  Serial.print("OK:");
  for (int i = 0; i < NUM_FANS; i++) {
    Serial.print(fanStates[i]);
    if (i < NUM_FANS - 1) Serial.print(",");
  }
  Serial.println();
}

// ═══════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════

void loop() {
  // ── Read serial commands ──
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
  
  // ── Watchdog: kill fans AND LEDs if no command in 30s ──
  if (millis() - lastCommandTime > WATCHDOG_MS) {
    // Check if anything is actually on before printing
    bool anyOn = false;
    for (int i = 0; i < NUM_FANS; i++) {
      if (fanStates[i]) { anyOn = true; break; }
    }
    if (!anyOn) {
      for (int i = 0; i < NUM_LEDS; i++) {
        if (ledStates[i]) { anyOn = true; break; }
      }
    }
    
    if (anyOn) {
      allFansOff();
      allLedsOff();
      Serial.println("WATCHDOG:all_off");
    }
    
    lastCommandTime = millis();  // Reset to avoid spamming
  }
}
