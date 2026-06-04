"""
backend/mqtt_bridge.py — MQTT real-time communication bridge for SRACE v2.

Provides instant pub/sub communication between:
  - Python backend (optimizer/PPO decisions)
  - Unity simulation (appliance state updates)
  - React dashboard (live room state)
  - Raspberry Pi (LED panel control)

Topics:
    srace/occupancy        — zone people counts (from camera/API)
    srace/appliance_states — fan/light/projector on/off (from optimizer)
    srace/room_state       — full room state JSON (for dashboard/Unity)
    srace/environment      — temp/CO₂/lux readings
    srace/command          — manual overrides (e.g. toggle fan F3)
    srace/anomaly          — anomaly alerts from River detector

Usage:
    from backend.mqtt_bridge import MQTTBridge
    bridge = MQTTBridge(broker="localhost", port=1883)
    bridge.start()
    bridge.publish_room_state(state_dict)

Requires: paho-mqtt >= 2.0
"""

import json
import threading
import time
from typing import Callable, Optional

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠ paho-mqtt not installed — MQTT bridge disabled")
    print("  Install with: pip install paho-mqtt>=2.0")


# ══════════════════════════════════════════════════════════════
#  MQTT TOPICS
# ══════════════════════════════════════════════════════════════

TOPIC_OCCUPANCY = "srace/occupancy"
TOPIC_APPLIANCE_STATES = "srace/appliance_states"
TOPIC_ROOM_STATE = "srace/room_state"
TOPIC_ENVIRONMENT = "srace/environment"
TOPIC_COMMAND = "srace/command"
TOPIC_ANOMALY = "srace/anomaly"


class MQTTBridge:
    """
    MQTT bridge for SRACE v2 real-time communication.

    Publishes optimizer results and subscribes to occupancy updates
    and manual commands. Thread-safe for use with FastAPI.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        client_id: str = "srace-backend",
        on_occupancy: Optional[Callable] = None,
        on_command: Optional[Callable] = None,
    ):
        """
        Initialize the MQTT bridge.

        Args:
            broker: MQTT broker hostname.
            port: MQTT broker port.
            client_id: Unique client identifier.
            on_occupancy: Callback when occupancy update received.
                          Signature: on_occupancy(zone_people: list[int])
            on_command: Callback when manual command received.
                        Signature: on_command(command: dict)
        """
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.on_occupancy = on_occupancy
        self.on_command = on_command

        self.connected = False
        self._client = None
        self._lock = threading.Lock()

        if not MQTT_AVAILABLE:
            return

        # Create MQTT client
        self._client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Will message: notify subscribers if we disconnect unexpectedly
        self._client.will_set(
            "srace/status",
            payload=json.dumps({"status": "offline", "client": client_id}),
            qos=1,
            retain=True,
        )

    def start(self) -> bool:
        """
        Connect to the MQTT broker and start the network loop.
        Returns True if connection initiated, False if MQTT unavailable.
        """
        if not MQTT_AVAILABLE or self._client is None:
            print("⚠ MQTT not available — bridge not started")
            return False

        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()  # Non-blocking background thread
            print(f"✓ MQTT bridge connecting to {self.broker}:{self.port}...")
            return True
        except Exception as e:
            print(f"✗ MQTT connection failed: {e}")
            print("  Make sure Mosquitto is running: sudo systemctl start mosquitto")
            return False

    def stop(self):
        """Disconnect and stop the network loop."""
        if self._client is not None:
            self._client.publish(
                "srace/status",
                json.dumps({"status": "offline", "client": self.client_id}),
                qos=1,
                retain=True,
            )
            self._client.loop_stop()
            self._client.disconnect()
            self.connected = False
            print("✓ MQTT bridge disconnected")

    # ── Callbacks ──────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Called when connection to broker is established."""
        self.connected = True
        print(f"✓ MQTT connected to {self.broker}:{self.port}")

        # Subscribe to incoming topics
        client.subscribe(TOPIC_OCCUPANCY, qos=1)
        client.subscribe(TOPIC_COMMAND, qos=1)

        # Publish online status
        client.publish(
            "srace/status",
            json.dumps({"status": "online", "client": self.client_id}),
            qos=1,
            retain=True,
        )

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Called when disconnected from broker."""
        self.connected = False
        if rc != 0:
            print(f"⚠ MQTT unexpected disconnect (rc={rc}), will reconnect...")

    def _on_message(self, client, userdata, msg):
        """Route incoming messages to appropriate handlers."""
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"⚠ MQTT: Invalid payload on {msg.topic}")
            return

        if msg.topic == TOPIC_OCCUPANCY and self.on_occupancy:
            zone_people = payload.get("zone_people", [])
            self.on_occupancy(zone_people)

        elif msg.topic == TOPIC_COMMAND and self.on_command:
            self.on_command(payload)

    # ── Publishers ──────────────────────────────────────────

    def publish_room_state(self, state: dict):
        """
        Publish full room state for dashboard/Unity consumption.

        Args:
            state: Room state dict (same format as /room_state response).
        """
        self._publish(TOPIC_ROOM_STATE, state)

    def publish_appliance_states(
        self,
        fan_states: list[bool],
        light_states: list[bool],
        projector_states: list[bool] = None,
    ):
        """
        Publish appliance on/off states for RPi LED panel.

        Args:
            fan_states: Boolean list for each fan.
            light_states: Boolean list for each light.
            projector_states: Boolean list for each projector (optional).
        """
        payload = {
            "fans": fan_states,
            "lights": light_states,
            "projectors": projector_states or [],
            "timestamp": time.time(),
        }
        self._publish(TOPIC_APPLIANCE_STATES, payload)

    def publish_environment(
        self,
        avg_temp: float,
        avg_co2: float,
        avg_lux: float,
        zone_temps: list[float] = None,
        zone_co2: list[float] = None,
    ):
        """
        Publish environment sensor readings.

        Args:
            avg_temp: Average room temperature °C.
            avg_co2: Average CO₂ concentration ppm.
            avg_lux: Average illuminance lux.
            zone_temps: Per-zone temperatures (optional).
            zone_co2: Per-zone CO₂ levels (optional).
        """
        payload = {
            "avg_temp": round(avg_temp, 1),
            "avg_co2": round(avg_co2, 0),
            "avg_lux": round(avg_lux, 1),
            "timestamp": time.time(),
        }
        if zone_temps:
            payload["zone_temps"] = [round(t, 1) for t in zone_temps]
        if zone_co2:
            payload["zone_co2"] = [round(c, 0) for c in zone_co2]

        self._publish(TOPIC_ENVIRONMENT, payload)

    def publish_anomaly(self, anomaly: dict):
        """
        Publish anomaly alert from River detector.

        Args:
            anomaly: Dict with keys: type, score, description, timestamp.
        """
        self._publish(TOPIC_ANOMALY, anomaly)

    def _publish(self, topic: str, payload: dict):
        """Thread-safe publish."""
        if not self.connected or self._client is None:
            return
        with self._lock:
            try:
                self._client.publish(topic, json.dumps(payload), qos=0)
            except Exception as e:
                print(f"⚠ MQTT publish failed on {topic}: {e}")

    @property
    def status(self) -> dict:
        """Current MQTT connection status."""
        return {
            "connected": self.connected,
            "broker": f"{self.broker}:{self.port}",
            "client_id": self.client_id,
            "available": MQTT_AVAILABLE,
        }
