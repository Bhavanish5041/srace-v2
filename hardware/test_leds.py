"""
hardware/test_leds.py — Test script for ESP32 zone LED control.

Connects to the ESP32, verifies the LED protocol, and cycles through
each LED individually before turning all on/off.

Usage:
    python hardware/test_leds.py
    python hardware/test_leds.py --port /dev/ttyUSB0
"""

import sys
import os
import time
import glob
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import serial
except ImportError:
    print("✗ pyserial not installed — pip install pyserial")
    sys.exit(1)

NUM_LEDS = 4
LED_NAMES = ["Z1", "Z4", "Z7", "Z12"]
BAUD = 115200


def auto_detect_port():
    """Find ESP32 serial port."""
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    for port in candidates:
        try:
            s = serial.Serial(port, BAUD, timeout=0.1)
            s.close()
            return port
        except (serial.SerialException, OSError):
            continue
    return None


def send_cmd(ser, cmd, timeout=2.0):
    """Send a command and read one response line."""
    ser.write((cmd + "\n").encode())
    ser.flush()
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                return line
        time.sleep(0.02)
    return None


def safe_shutdown(ser):
    """Send ALLOFF + LEDOFF before closing."""
    try:
        ser.write(b"ALLOFF\n")
        time.sleep(0.2)
        ser.write(b"LEDOFF\n")
        time.sleep(0.2)
        # Drain any responses
        while ser.in_waiting:
            ser.readline()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="SRACE LED Test")
    parser.add_argument("--port", type=str, default=None)
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    if not port:
        print("✗ No ESP32 found. Check USB connection.")
        sys.exit(1)

    print(f"\n{'═' * 50}")
    print(f"  SRACE v2 — LED Wiring Test")
    print(f"  Port: {port}")
    print(f"{'═' * 50}\n")

    ser = serial.Serial(port, BAUD, timeout=2)
    print("  Opened. Waiting 5s for ESP32 reset...", flush=True)
    time.sleep(5)

    # Drain boot messages (ROM bootloader output, etc.)
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(f"  Boot: {line}")

    # Try PING up to 3 times (ESP32 may still be settling)
    connected = False
    for attempt in range(3):
        resp = send_cmd(ser, "PING", timeout=2.0)
        if resp == "PONG":
            connected = True
            break
        print(f"  Attempt {attempt + 1}: got '{resp}', retrying...", flush=True)
        time.sleep(1)

    if not connected:
        print(f"  ✗ ESP32 not responding after 3 attempts")
        print(f"    → Make sure the updated .ino sketch is flashed")
        ser.close()
        sys.exit(1)
    print("  ✓ ESP32 alive (PONG)\n")

    try:
        # ── Step 1: Query LED status ──
        print("  1. Querying LEDSTATUS...")
        resp = send_cmd(ser, "LEDSTATUS")
        print(f"     → {resp}\n")

        # ── Step 2: Verify fans still work ──
        print("  2. Quick fan check (STATUS)...")
        resp = send_cmd(ser, "STATUS")
        print(f"     → {resp}\n")

        # ── Step 3: Cycle each LED ──
        print("  3. LED sweep test (1s each):\n")
        for i in range(NUM_LEDS):
            states = [0] * NUM_LEDS
            states[i] = 1
            cmd_str = ",".join(str(s) for s in states)
            resp = send_cmd(ser, f"LED:{cmd_str}")
            marker = "█" if states[i] else "·"
            led_vis = "  ".join("█" if s else "·" for s in states)
            print(f"     LED {LED_NAMES[i]:>3} ON   [{led_vis}]  → {resp}")
            time.sleep(1.0)

        # ── Step 4: All LEDs on ──
        print(f"\n  4. All LEDs ON...")
        resp = send_cmd(ser, "LED:1,1,1,1")
        print(f"     → {resp}")
        time.sleep(2.0)

        # ── Step 5: LEDOFF ──
        print(f"\n  5. LEDOFF...")
        resp = send_cmd(ser, "LEDOFF")
        print(f"     → {resp}")

        # ── Step 6: Combo test (fans + LEDs together) ──
        print(f"\n  6. Combo test: Fan 1 + LED Z1 ON together...")
        resp1 = send_cmd(ser, "1,0,0,0")
        print(f"     Fan  → {resp1}")
        resp2 = send_cmd(ser, "LED:1,0,0,0")
        print(f"     LED  → {resp2}")
        time.sleep(2.0)

        # Clean up combo
        send_cmd(ser, "ALLOFF")
        send_cmd(ser, "LEDOFF")

        print(f"\n{'═' * 50}")
        print(f"  ✓ LED TEST COMPLETE")
        print(f"  Verify each LED lit up in order: {', '.join(LED_NAMES)}")
        print(f"{'═' * 50}\n")

    except KeyboardInterrupt:
        print("\n\n  Interrupted! Shutting down safely...")

    finally:
        safe_shutdown(ser)
        ser.close()
        print("  ✓ All fans OFF, all LEDs OFF, port closed.\n")


if __name__ == "__main__":
    main()
