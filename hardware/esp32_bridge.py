"""
hardware/esp32_bridge.py — Serial bridge to ESP32 fan + LED controller.

Sends binary fan state commands [0/1, 0/1, 0/1, 0/1] and LED state commands
[0/1, 0/1, 0/1, 0/1] to the ESP32 over USB serial. The ESP32 translates
these into L298N motor driver pin states (fans) and GPIO outputs (LEDs).

Usage:
    from hardware.esp32_bridge import ESP32Bridge

    bridge = ESP32Bridge()        # auto-detects port
    bridge.connect()              # waits for "READY"
    bridge.send_fan_states([1, 0, 1, 1])
    bridge.disconnect()           # all fans OFF + close port

Requires: pip install pyserial
"""

import glob
import sys
import time
import threading
from typing import Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠ pyserial not installed — ESP32 bridge disabled")
    print("  Install with: pip install pyserial")


NUM_FANS = 4
NUM_LEDS = 4
BAUD_RATE = 115200
READY_TIMEOUT = 5.0
COMMAND_TIMEOUT = 2.0
KEEPALIVE_INTERVAL = 10.0


def auto_detect_port() -> Optional[str]:
    """Auto-detect ESP32 serial port on Linux/Mac/Windows."""
    if sys.platform.startswith("linux"):
        candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    elif sys.platform == "darwin":
        candidates = glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB*")
    elif sys.platform == "win32":
        candidates = [f"COM{i}" for i in range(1, 20)]
    else:
        candidates = []

    valid = []
    for port in candidates:
        try:
            s = serial.Serial(port, BAUD_RATE, timeout=0.1)
            s.close()
            valid.append(port)
        except (serial.SerialException, OSError):
            continue

    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    for preferred in ["/dev/ttyUSB0", "/dev/ttyACM0"]:
        if preferred in valid:
            return preferred
    return valid[0]


class ESP32Bridge:
    """
    Serial bridge to ESP32 fan + LED controller. Thread-safe.

    Attributes:
        port: Serial port path
        connected: Whether bridge is connected and ready
        fan_states: Current fan states [0/1, 0/1, 0/1, 0/1]
        led_states: Current LED states [0/1, 0/1, 0/1, 0/1]
    """

    def __init__(self, port: Optional[str] = None, baud_rate: int = BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.connected = False
        self.fan_states = [0] * NUM_FANS
        self.led_states = [0] * NUM_LEDS
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_running = False

    def connect(self, timeout: float = READY_TIMEOUT) -> bool:
        """Connect to ESP32, wait for READY signal."""
        if not SERIAL_AVAILABLE:
            print("✗ pyserial not installed")
            return False

        if self.port is None:
            self.port = auto_detect_port()
            if self.port is None:
                print("✗ No ESP32 found! Check USB connection.")
                print("  Run: ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null")
                return False
            print(f"  Auto-detected ESP32 on {self.port}")

        try:
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=1.0)
            print(f"  Opening {self.port} at {self.baud_rate} baud...")
        except serial.SerialException as e:
            print(f"✗ Failed to open {self.port}: {e}")
            print("  → Check permissions: sudo usermod -a -G dialout $USER")
            return False

        time.sleep(2.0)  # ESP32 resets on serial open
        self._serial.reset_input_buffer()

        print("  Waiting for ESP32 READY signal...")
        start = time.time()
        while time.time() - start < timeout:
            if self._serial.in_waiting > 0:
                line = self._serial.readline().decode(errors="ignore").strip()
                if line == "READY":
                    self.connected = True
                    print(f"✓ ESP32 connected on {self.port}")
                    self._start_keepalive()
                    return True
            time.sleep(0.1)

        # Fallback: try PING
        self._serial.write(b"PING\n")
        time.sleep(0.3)
        if self._serial.in_waiting > 0:
            line = self._serial.readline().decode(errors="ignore").strip()
            if line == "PONG":
                self.connected = True
                print(f"✓ ESP32 connected on {self.port} (via PING)")
                self._start_keepalive()
                return True

        print(f"✗ ESP32 did not respond within {timeout}s")
        self._serial.close()
        self._serial = None
        return False

    def disconnect(self):
        """Disconnect. Turns all fans and LEDs OFF first."""
        self._keepalive_running = False
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=2.0)

        if self._serial and self._serial.is_open:
            try:
                self._serial.write(b"ALLOFF\n")
                time.sleep(0.1)
                if self._serial.in_waiting > 0:
                    self._serial.readline()
                self._serial.write(b"LEDOFF\n")
                time.sleep(0.1)
                if self._serial.in_waiting > 0:
                    self._serial.readline()
            except serial.SerialException:
                pass
            finally:
                self._serial.close()

        self.connected = False
        self.fan_states = [0] * NUM_FANS
        self.led_states = [0] * NUM_LEDS
        print("✓ ESP32 disconnected (all fans + LEDs OFF)")

    def send_fan_states(self, states: list[int]) -> bool:
        """Send fan states to ESP32. states: list of 4 ints (0 or 1)."""
        if not self.connected or not self._serial:
            return False

        if len(states) != NUM_FANS:
            print(f"⚠ Expected {NUM_FANS} fan states, got {len(states)}")
            return False

        binary = [1 if s else 0 for s in states]
        cmd = ",".join(str(s) for s in binary) + "\n"

        with self._lock:
            try:
                self._serial.write(cmd.encode())
                self._serial.flush()
                response = self._read_response(timeout=COMMAND_TIMEOUT)
                if response and response.startswith("OK:"):
                    self.fan_states = binary
                    return True
                elif response and response.startswith("ERR:"):
                    print(f"  ⚠ ESP32 error: {response}")
                    return False
                else:
                    self.fan_states = binary
                    return False
            except serial.SerialException as e:
                print(f"  ✗ Serial error: {e}")
                self.connected = False
                return False

    def get_status(self) -> Optional[list[int]]:
        """Query current fan states from ESP32."""
        if not self.connected or not self._serial:
            return None
        with self._lock:
            try:
                self._serial.write(b"STATUS\n")
                response = self._read_response(timeout=COMMAND_TIMEOUT)
                if response and response.startswith("STATE:"):
                    values = response[6:].split(",")
                    states = [int(v.strip()) for v in values]
                    self.fan_states = states
                    return states
                return None
            except (serial.SerialException, ValueError):
                return None

    def all_off(self) -> bool:
        """Emergency stop — turn all fans OFF."""
        if not self.connected or not self._serial:
            return False
        with self._lock:
            try:
                self._serial.write(b"ALLOFF\n")
                self._read_response(timeout=COMMAND_TIMEOUT)
                self.fan_states = [0] * NUM_FANS
                return True
            except serial.SerialException:
                return False

    # ── LED control methods ──────────────────────────────────

    def send_led_states(self, states: list[int]) -> bool:
        """Send LED states to ESP32. states: list of 4 ints (0 or 1)."""
        if not self.connected or not self._serial:
            return False

        if len(states) != NUM_LEDS:
            print(f"⚠ Expected {NUM_LEDS} LED states, got {len(states)}")
            return False

        binary = [1 if s else 0 for s in states]
        cmd = "LED:" + ",".join(str(s) for s in binary) + "\n"

        with self._lock:
            try:
                self._serial.write(cmd.encode())
                self._serial.flush()
                response = self._read_response(timeout=COMMAND_TIMEOUT)
                if response and response.startswith("LEDOK:"):
                    self.led_states = binary
                    return True
                elif response and response.startswith("ERR:"):
                    print(f"  ⚠ ESP32 LED error: {response}")
                    return False
                else:
                    self.led_states = binary
                    return False
            except serial.SerialException as e:
                print(f"  ✗ Serial error: {e}")
                self.connected = False
                return False

    def get_led_status(self) -> Optional[list[int]]:
        """Query current LED states from ESP32."""
        if not self.connected or not self._serial:
            return None
        with self._lock:
            try:
                self._serial.write(b"LEDSTATUS\n")
                response = self._read_response(timeout=COMMAND_TIMEOUT)
                if response and response.startswith("LEDSTATE:"):
                    values = response[9:].split(",")
                    states = [int(v.strip()) for v in values]
                    self.led_states = states
                    return states
                return None
            except (serial.SerialException, ValueError):
                return None

    def leds_off(self) -> bool:
        """Turn all LEDs OFF."""
        if not self.connected or not self._serial:
            return False
        with self._lock:
            try:
                self._serial.write(b"LEDOFF\n")
                self._read_response(timeout=COMMAND_TIMEOUT)
                self.led_states = [0] * NUM_LEDS
                return True
            except serial.SerialException:
                return False

    def all_devices_off(self) -> bool:
        """Emergency stop — turn all fans AND LEDs OFF."""
        fan_ok = self.all_off()
        led_ok = self.leds_off()
        return fan_ok and led_ok

    def _read_response(self, timeout: float = COMMAND_TIMEOUT) -> Optional[str]:
        """Read one line from ESP32 with timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self._serial.in_waiting > 0:
                line = self._serial.readline().decode(errors="ignore").strip()
                if line:
                    return line
            time.sleep(0.02)
        return None

    def _start_keepalive(self):
        """Start background PING thread to prevent ESP32 watchdog."""
        self._keepalive_running = True
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="esp32-keepalive",
        )
        self._keepalive_thread.start()

    def _keepalive_loop(self):
        """Send periodic PINGs."""
        while self._keepalive_running and self.connected:
            time.sleep(KEEPALIVE_INTERVAL)
            if not self._keepalive_running:
                break
            with self._lock:
                try:
                    if self._serial and self._serial.is_open:
                        self._serial.write(b"PING\n")
                        self._read_response(timeout=0.5)
                except serial.SerialException:
                    self.connected = False
                    break

    @property
    def status(self) -> dict:
        """Current bridge status dict (for API endpoints)."""
        return {
            "connected": self.connected,
            "port": self.port,
            "fan_states": self.fan_states,
            "fans_on": sum(self.fan_states),
            "total_fans": NUM_FANS,
            "led_states": self.led_states,
            "leds_on": sum(self.led_states),
            "total_leds": NUM_LEDS,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESP32 fan bridge test")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Run fan sweep test")
    args = parser.parse_args()

    bridge = ESP32Bridge(port=args.port)
    if not bridge.connect():
        sys.exit(1)

    if args.test:
        print("\n  === Fan Sweep Test ===")
        for i in range(NUM_FANS):
            states = [0] * NUM_FANS
            states[i] = 1
            print(f"  → F{i+1} ON  {states}")
            bridge.send_fan_states(states)
            time.sleep(5.0)
        print("  → All OFF")
        bridge.all_off()
        time.sleep(0.5)
        print("  → All ON")
        bridge.send_fan_states([1, 1, 1, 1])
        time.sleep(2.0)
        print("  → All OFF")
        bridge.all_off()
        print("  === Test Complete ===\n")
    else:
        print("\n  Interactive: type '1,0,1,1' or 'q' to quit\n")
        try:
            while True:
                cmd = input("  > ").strip()
                if cmd.lower() in ("q", "quit", "exit"):
                    break
                elif cmd.lower() == "status":
                    print(f"    State: {bridge.get_status()}")
                elif cmd.lower() == "off":
                    bridge.all_off()
                    print("    All fans OFF")
                else:
                    try:
                        states = [int(x.strip()) for x in cmd.split(",")]
                        ok = bridge.send_fan_states(states)
                        print(f"    {'✓' if ok else '✗'} Sent: {states}")
                    except ValueError:
                        print("    ⚠ Format: 1,0,1,1")
        except (KeyboardInterrupt, EOFError):
            pass

    bridge.disconnect()
