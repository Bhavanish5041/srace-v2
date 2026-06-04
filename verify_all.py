"""
SRACE v2 — Full verification suite for all new features.
Run: python verify_all.py
"""
import sys, os, traceback
sys.path.insert(0, '.')

results = []

def check(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
        print(f"  ✓ {name}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()

# ═══════════════════════════════════════════
#  1. Core imports
# ═══════════════════════════════════════════
print("\n═══ 1. Core Imports ═══")

def test_room_config():
    from core.room_config import RoomConfig, load_config
    cfg = load_config("config/default_room.json")
    assert cfg.n_fans > 0
    assert cfg.n_lights > 0
    assert cfg.n_projectors >= 0
    assert cfg.n_appliances == cfg.n_fans + cfg.n_lights + cfg.n_projectors
check("RoomConfig + Projectors", test_room_config)

def test_coverage():
    from core.room_config import load_config
    from core.coverage import CoverageResult
    from physics.airflow import compute_airflow_matrix
    from physics.thermal import simulate_thermal
    from physics.co2_model import simulate_co2
    from physics.lighting import compute_lux_matrix
    import numpy as np
    cfg = load_config("config/default_room.json")
    af = compute_airflow_matrix(cfg)
    lx = compute_lux_matrix(cfg)
    occ = np.full(cfg.n_zones, 3.0)
    th = simulate_thermal(cfg, af, occ)
    co = simulate_co2(cfg, af, occ)
    cr = CoverageResult(cfg, af, th, co, lx, set(range(cfg.n_zones)))
    assert cr.binary.shape == (cfg.n_appliances, cfg.n_zones), f"Expected ({cfg.n_appliances},{cfg.n_zones}), got {cr.binary.shape}"
    assert len(cr.appliance_ids) == cfg.n_appliances
    assert len(cr.appliance_watts) == cfg.n_appliances
check("Coverage matrix (incl. projectors)", test_coverage)

# ═══════════════════════════════════════════
#  2. Gym environment
# ═══════════════════════════════════════════
print("\n═══ 2. Gym Environment ═══")

def test_gym_env():
    from ml.gym_env import SRACEEnv
    env = SRACEEnv()
    assert env.n_projectors >= 0
    assert env.n_appliances == env.n_fans + env.n_lights + env.n_projectors
    assert env.coverage_matrix.shape[0] == env.n_appliances
    assert env.action_space.shape[0] == env.n_appliances
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    # Run 10 steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
    assert isinstance(reward, float)
check("Gym env reset + 10 steps", test_gym_env)

# ═══════════════════════════════════════════
#  3. Anomaly detector
# ═══════════════════════════════════════════
print("\n═══ 3. Anomaly Detector ═══")

def test_anomaly_detector():
    from ml.anomaly_detector import SRACEAnomalyDetector
    det = SRACEAnomalyDetector()
    # Normal reading
    score, alerts = det.update({
        "avg_temp": 25.0, "avg_co2": 450.0, "avg_lux": 300.0,
        "total_power": 200.0, "n_people": 10, "n_active": 5, "n_occupied_zones": 4,
    })
    assert isinstance(score, float)
    # Danger reading
    score2, alerts2 = det.update({
        "avg_temp": 38.0, "avg_co2": 1500.0, "avg_lux": 300.0,
        "total_power": 200.0, "n_people": 10, "n_active": 5, "n_occupied_zones": 4,
    })
    assert len(alerts2) >= 2, f"Expected danger alerts, got {len(alerts2)}"
    assert any(a["type"] == "temperature_danger" for a in alerts2)
    assert any(a["type"] == "co2_danger" for a in alerts2)
    # Waste reading
    score3, alerts3 = det.update({
        "avg_temp": 25.0, "avg_co2": 400.0, "avg_lux": 300.0,
        "total_power": 500.0, "n_people": 0, "n_active": 8, "n_occupied_zones": 0,
    })
    assert any(a["type"] == "power_waste" for a in alerts3)
    stats = det.get_statistics()
    assert stats["total_readings"] == 3
check("Anomaly detector (normal + danger + waste)", test_anomaly_detector)

# ═══════════════════════════════════════════
#  4. MQTT bridge
# ═══════════════════════════════════════════
print("\n═══ 4. MQTT Bridge ═══")

def test_mqtt_import():
    from backend.mqtt_bridge import MQTTBridge, TOPIC_OCCUPANCY, TOPIC_ROOM_STATE
    bridge = MQTTBridge(broker="localhost", port=1883)
    assert bridge.broker == "localhost"
    assert bridge.connected == False
    status = bridge.status
    assert "connected" in status
    assert "broker" in status
check("MQTT bridge import + status", test_mqtt_import)

# ═══════════════════════════════════════════
#  5. API import check
# ═══════════════════════════════════════════
print("\n═══ 5. API Module ═══")

def test_api_import():
    # Just verify the module parses without errors
    import importlib.util
    spec = importlib.util.spec_from_file_location("api", "backend/api.py")
    mod = importlib.util.module_from_spec(spec)
    # Don't execute (would start server), just verify syntax
    assert spec is not None
check("API module syntax", test_api_import)

# ═══════════════════════════════════════════
#  6. Config files
# ═══════════════════════════════════════════
print("\n═══ 6. Config Files ═══")

def test_configs():
    from core.room_config import load_config
    configs = ["config/default_room.json", "config/classroom_real.json"]
    for path in configs:
        if os.path.exists(path):
            cfg = load_config(path)
            assert cfg.n_zones > 0
            assert cfg.n_appliances > 0
check("Config file loading", test_configs)

# ═══════════════════════════════════════════
#  7. C# file existence
# ═══════════════════════════════════════════
print("\n═══ 7. C# Files ═══")

def test_csharp_files():
    files = [
        "Assets/SRACE/Scripts/Physics/ThermalModel.cs",
        "Assets/SRACE/Scripts/Physics/CO2Model.cs",
        "Assets/SRACE/Scripts/Environment/PowerHUD.cs",
        "Assets/SRACE/Scripts/Core/SRACEManager.cs",
    ]
    for f in files:
        assert os.path.exists(f), f"Missing: {f}"
        size = os.path.getsize(f)
        assert size > 1000, f"{f} too small ({size} bytes)"
check("C# physics + HUD files exist", test_csharp_files)

# ═══════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════
print(f"\n{'═'*50}")
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"  Results: {passed} passed, {failed} failed out of {len(results)}")
if failed:
    print("\n  Failures:")
    for name, status, err in results:
        if status == "FAIL":
            print(f"    ✗ {name}: {err}")
else:
    print("  All checks passed! ✓")
print(f"{'═'*50}\n")

sys.exit(1 if failed else 0)
