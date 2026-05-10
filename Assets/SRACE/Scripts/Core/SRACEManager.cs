// SRACEManager.cs — Main orchestrator MonoBehaviour
// Loads JSON config → builds room → spawns fans/lights → runs physics → updates visuals.
//
// Modes:
//   Local  — C# physics + greedy (keyboard presets 1-5, Space)
//   API    — Python backend via SRACEApiClient (polls /room_state every 5s)
//
// Demo controls:
//   [1] Empty room        [2] Single person (Z0)    [3] Center cluster
//   [4] Full room          [5] Front row only
//   [Space] Re-run optimizer and update visuals
//   [F] Toggle all fans    [L] Toggle all lights

using System.Collections.Generic;
using UnityEngine;
using SRACE.Core;
using SRACE.Physics;
using SRACE.Environment;

namespace SRACE.Core
{
    public class SRACEManager : MonoBehaviour
    {
        [Header("Config")]
        [Tooltip("Drag default_room.json TextAsset here, or leave null to load from Resources.")]
        public TextAsset roomConfigAsset;

        // ── Runtime State ──
        private RoomConfig config;
        private CoverageResult coverage;

        private GameObject roomRoot;
        private FanObject[] fanObjects;
        private LightObject[] lightObjects;
        private ZoneHeatmap heatmap;
        private ClassroomCamera orbitCamera;

        private int[] currentOccupancy;
        private bool[] activeFans;
        private bool[] activeLights;

        // ── Public accessors for API client ──
        public RoomConfig Config => config;
        public FanObject[] FanObjects => fanObjects;
        public LightObject[] LightObjects => lightObjects;
        public ZoneHeatmap Heatmap => heatmap;

        // ══════════════════════════════════════════
        //  LIFECYCLE
        // ══════════════════════════════════════════

        private void Start()
        {
            LoadConfig();
            BuildRoom();
            SpawnFans();
            SpawnLights();
            SetupHeatmap();
            SetupCamera();
            InitializeState();

            config.PrintSummary();
            Debug.Log("SRACE Manager initialized. Press [1–5] for occupancy presets, [Space] to run optimizer.");
        }

        private void Update()
        {
            HandleInput();
        }

        // ══════════════════════════════════════════
        //  CONFIG
        // ══════════════════════════════════════════

        private void LoadConfig()
        {
            string json;
            if (roomConfigAsset != null)
            {
                json = roomConfigAsset.text;
            }
            else
            {
                // Try loading from Resources folder
                var asset = Resources.Load<TextAsset>("classroom_real");
                if (asset == null)
                {
                    Debug.LogError("No room config found! Drag a TextAsset to SRACEManager " +
                                   "or place default_room.json in Assets/SRACE/Resources/");
                    return;
                }
                json = asset.text;
            }

            config = RoomConfigLoader.LoadFromJson(json);
        }

        // ══════════════════════════════════════════
        //  ROOM BUILDING
        // ══════════════════════════════════════════

        private void BuildRoom()
        {
            roomRoot = RoomBuilder.Build(config);
        }

        private void SpawnFans()
        {
            fanObjects = new FanObject[config.NFans];

            var fansRoot = new GameObject("Fans");
            fansRoot.transform.SetParent(roomRoot.transform);

            for (int i = 0; i < config.NFans; i++)
            {
                var fan = config.fans[i];
                var fanGO = new GameObject($"Fan_{fan.id}");
                fanGO.transform.SetParent(fansRoot.transform);

                // JSON x,y → Unity x,z. Fan hangs from ceiling.
                fanGO.transform.localPosition = new Vector3(
                    fan.x, config.ceilingHeight, fan.y);

                var fanObj = fanGO.AddComponent<FanObject>();
                fanObj.BuildGeometry(fan.id, config.ceilingHeight);
                fanObjects[i] = fanObj;
            }
        }

        private void SpawnLights()
        {
            lightObjects = new LightObject[config.NLights];

            var lightsRoot = new GameObject("Lights");
            lightsRoot.transform.SetParent(roomRoot.transform);

            for (int i = 0; i < config.NLights; i++)
            {
                var light = config.lights[i];
                var lightGO = new GameObject($"Light_{light.id}");
                lightGO.transform.SetParent(lightsRoot.transform);

                // JSON x,y → Unity x,z. Light mounts at ceiling.
                lightGO.transform.localPosition = new Vector3(
                    light.x, config.ceilingHeight, light.y);

                var lightObj = lightGO.AddComponent<LightObject>();
                lightObj.BuildGeometry(light.id, light.lumens);
                lightObjects[i] = lightObj;
            }
        }

        private void SetupHeatmap()
        {
            var heatmapGO = new GameObject("ZoneHeatmap");
            heatmapGO.transform.SetParent(roomRoot.transform);
            heatmap = heatmapGO.AddComponent<ZoneHeatmap>();
            heatmap.Initialize(config);
        }

        private void SetupCamera()
        {
            // Find or create main camera
            var cam = Camera.main;
            if (cam == null)
            {
                var camGO = new GameObject("MainCamera");
                camGO.tag = "MainCamera";
                cam = camGO.AddComponent<Camera>();
            }

            orbitCamera = cam.gameObject.GetComponent<ClassroomCamera>();
            if (orbitCamera == null)
                orbitCamera = cam.gameObject.AddComponent<ClassroomCamera>();

            orbitCamera.FrameRoom(config.width, config.depth, config.ceilingHeight);

            // Set background to dark blue-grey
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.12f, 0.12f, 0.16f);
        }

        private void InitializeState()
        {
            currentOccupancy = new int[config.NZones];
            activeFans = new bool[config.NFans];
            activeLights = new bool[config.NLights];
        }

        // ══════════════════════════════════════════
        //  PHYSICS + OPTIMIZER
        // ══════════════════════════════════════════

        private void RunPipelineAndUpdateVisuals()
        {
            // 1. Compute physics matrices
            var airflowMatrix = AirflowModel.ComputeAirflowMatrix(config);
            var luxMatrix = LightingModel.ComputeLuxMatrix(config);

            // Estimate thermal impact from airflow (analytical, no ODE solver needed)
            // ΔT ≈ convCoeff × airflow × (T_ambient − T_target) × forecastTime
            // This approximates the marginal cooling each fan provides.
            const float convCoeff = 0.15f;    // convection coefficient (matches Python)
            const float forecastSec = 300f;   // 5-minute forecast window
            float tempDelta = config.ambientTemp - config.comfort.targetTempC;
            var thermalImpact = new float[config.NFans, config.NZones];
            var co2Reduction = new float[config.NFans, config.NZones];
            for (int fi = 0; fi < config.NFans; fi++)
            {
                for (int zi = 0; zi < config.NZones; zi++)
                {
                    // Thermal: cooling proportional to airflow and temperature difference
                    thermalImpact[fi, zi] = convCoeff * airflowMatrix[fi, zi]
                                            * Mathf.Max(tempDelta, 0f) * forecastSec * 0.01f;
                    // CO₂: ventilation proportional to airflow
                    co2Reduction[fi, zi] = airflowMatrix[fi, zi] * 50f; // rough ppm reduction
                }
            }

            // 2. Find occupied zones
            var occupiedZones = new HashSet<int>();
            for (int i = 0; i < config.NZones; i++)
            {
                if (currentOccupancy[i] > 0)
                    occupiedZones.Add(i);
            }

            // 3. Build coverage result
            coverage = new CoverageResult(config, airflowMatrix, thermalImpact,
                                          co2Reduction, luxMatrix, occupiedZones);
            coverage.PrintSummary();

            // 4. Simple greedy: activate appliances that cover occupied zones
            // (Reuses the coverage binary matrix for a quick visual demo)
            GreedyActivate(occupiedZones);

            // 5. Update heatmap
            var coveredZones = new bool[config.NZones];
            for (int zi = 0; zi < config.NZones; zi++)
            {
                if (!occupiedZones.Contains(zi)) continue;
                for (int ai = 0; ai < config.NAppliances; ai++)
                {
                    bool applianceOn = ai < config.NFans ? activeFans[ai] : activeLights[ai - config.NFans];
                    if (applianceOn && coverage.Binary[ai, zi] == 1)
                    {
                        coveredZones[zi] = true;
                        break;
                    }
                }
            }
            heatmap.UpdateHeatmap(currentOccupancy, coveredZones);

            // 6. Log power usage
            float totalPower = 0;
            for (int i = 0; i < config.NFans; i++)
                if (activeFans[i]) totalPower += config.fans[i].powerWatts;
            for (int i = 0; i < config.NLights; i++)
                if (activeLights[i]) totalPower += config.lights[i].powerWatts;

            float savings = config.MaxPowerWatts > 0
                ? (1f - totalPower / config.MaxPowerWatts) * 100f : 0f;

            Debug.Log($"⚡ Power: {totalPower:F0}W / {config.MaxPowerWatts:F0}W  " +
                      $"({savings:F1}% savings)  |  Occupied zones: {occupiedZones.Count}/{config.NZones}");
        }

        /// <summary>
        /// Greedy activation with SEPARATE phases for fans and lights.
        /// Fans cover airflow/thermal needs, lights cover illumination.
        /// Running them together causes lights (cheaper per watt) to "steal"
        /// coverage from fans, leaving occupied zones with zero airflow.
        /// </summary>
        private void GreedyActivate(HashSet<int> occupiedZones)
        {
            // Reset all
            for (int i = 0; i < activeFans.Length; i++) activeFans[i] = false;
            for (int i = 0; i < activeLights.Length; i++) activeLights[i] = false;

            if (occupiedZones.Count == 0)
            {
                ApplyApplianceStates();
                return;
            }

            // Phase 1: Greedy over FANS only (airflow coverage)
            GreedySubset(occupiedZones, 0, config.NFans);

            // Phase 2: Greedy over LIGHTS only (illumination coverage)
            GreedySubset(occupiedZones, config.NFans, config.NAppliances);

            ApplyApplianceStates();
        }

        /// <summary>
        /// Run greedy over a subset of appliances [startIdx, endIdx).
        /// </summary>
        private void GreedySubset(HashSet<int> occupiedZones, int startIdx, int endIdx)
        {
            var uncovered = new HashSet<int>(occupiedZones);

            while (uncovered.Count > 0)
            {
                int bestApp = -1;
                float bestScore = -1f;

                for (int ai = startIdx; ai < endIdx; ai++)
                {
                    bool alreadyOn = ai < config.NFans ? activeFans[ai] : activeLights[ai - config.NFans];
                    if (alreadyOn) continue;

                    int coversCount = 0;
                    for (int zi = 0; zi < config.NZones; zi++)
                    {
                        if (uncovered.Contains(zi) && coverage.Binary[ai, zi] == 1)
                            coversCount++;
                    }

                    if (coversCount == 0) continue;

                    float watts = config.GetApplianceWatts(ai);
                    float score = coversCount / watts; // coverage per watt
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestApp = ai;
                    }
                }

                if (bestApp < 0) break; // no more progress

                // Activate best appliance
                if (bestApp < config.NFans)
                    activeFans[bestApp] = true;
                else
                    activeLights[bestApp - config.NFans] = true;

                // Remove newly covered zones
                for (int zi = 0; zi < config.NZones; zi++)
                {
                    if (coverage.Binary[bestApp, zi] == 1)
                        uncovered.Remove(zi);
                }
            }
        }

        private void ApplyApplianceStates()
        {
            for (int i = 0; i < fanObjects.Length; i++)
                fanObjects[i].SetActive(activeFans[i]);
            for (int i = 0; i < lightObjects.Length; i++)
                lightObjects[i].SetActive(activeLights[i]);
        }

        // ══════════════════════════════════════════
        //  API-DRIVEN STATE (called by SRACEApiClient)
        // ══════════════════════════════════════════

        /// <summary>
        /// Apply a full room state received from the Python backend.
        /// Bypasses local C# physics — Python is the source of truth.
        /// </summary>
        public void ApplyApiState(
            int[] occupancy,
            bool[] fanStates,
            bool[] lightStates,
            bool[] coveredZones,
            float totalPowerW,
            float powerSavedPct)
        {
            // Update occupancy
            if (occupancy != null && occupancy.Length == config.NZones)
                currentOccupancy = occupancy;

            // Update fans
            if (fanStates != null && fanStates.Length == fanObjects.Length)
            {
                activeFans = fanStates;
                for (int i = 0; i < fanObjects.Length; i++)
                    fanObjects[i].SetActive(activeFans[i]);
            }

            // Update lights
            if (lightStates != null && lightStates.Length == lightObjects.Length)
            {
                activeLights = lightStates;
                for (int i = 0; i < lightObjects.Length; i++)
                    lightObjects[i].SetActive(activeLights[i]);
            }

            // Update heatmap
            if (coveredZones != null && coveredZones.Length == config.NZones)
                heatmap.UpdateHeatmap(currentOccupancy, coveredZones);

            Debug.Log($"🌐 API → Power: {totalPowerW:F0}W  |  Saved: {powerSavedPct:F1}%  |  " +
                      $"Fans: {CountTrue(activeFans)}/{activeFans.Length}  " +
                      $"Lights: {CountTrue(activeLights)}/{activeLights.Length}");
        }

        private static int CountTrue(bool[] arr)
        {
            int n = 0;
            foreach (var b in arr) if (b) n++;
            return n;
        }

        // ══════════════════════════════════════════
        //  INPUT / DEMO PRESETS
        // ══════════════════════════════════════════

        private void HandleInput()
        {
            bool changed = false;

            // Preset occupancy scenarios (same as Python main.py)
            if (Input.GetKeyDown(KeyCode.Alpha1))
            {
                SetOccupancy("Empty Room", new int[config.NZones]);
                changed = true;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha2))
            {
                var occ = new int[config.NZones];
                occ[0] = 1; // single person in zone 0
                SetOccupancy("Single Person (Zone 0)", occ);
                changed = true;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha3))
            {
                var occ = new int[config.NZones];
                // Center cluster: zones 5,6,9,10 (middle 2x2 block in a 4-col grid)
                if (config.NZones > 10) { occ[5] = 3; occ[6] = 3; occ[9] = 3; occ[10] = 2; }
                SetOccupancy("Center Cluster", occ);
                changed = true;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha4))
            {
                var occ = new int[config.NZones];
                for (int i = 0; i < occ.Length; i++) occ[i] = 4;
                SetOccupancy("Full Room", occ);
                changed = true;
            }
            else if (Input.GetKeyDown(KeyCode.Alpha5))
            {
                var occ = new int[config.NZones];
                int frontCols = Mathf.Min(config.nZoneCols, occ.Length);
                for (int i = 0; i < frontCols; i++) occ[i] = 5;
                SetOccupancy("Front Row Only", occ);
                changed = true;
            }

            // Re-run optimizer
            if (Input.GetKeyDown(KeyCode.Space) || changed)
            {
                RunPipelineAndUpdateVisuals();
            }

            // Toggle all fans
            if (Input.GetKeyDown(KeyCode.F))
            {
                bool anyOn = false;
                foreach (var f in activeFans) if (f) { anyOn = true; break; }
                bool newState = !anyOn;
                for (int i = 0; i < activeFans.Length; i++) activeFans[i] = newState;
                ApplyApplianceStates();
                Debug.Log($"All fans {(newState ? "ON" : "OFF")}");
            }

            // Toggle all lights
            if (Input.GetKeyDown(KeyCode.L))
            {
                bool anyOn = false;
                foreach (var l in activeLights) if (l) { anyOn = true; break; }
                bool newState = !anyOn;
                for (int i = 0; i < activeLights.Length; i++) activeLights[i] = newState;
                ApplyApplianceStates();
                Debug.Log($"All lights {(newState ? "ON" : "OFF")}");
            }
        }

        private void SetOccupancy(string scenarioName, int[] occupancy)
        {
            currentOccupancy = occupancy;
            Debug.Log($"━━━ Scenario: {scenarioName} ━━━");

            // Also notify the Python backend if the API client is present
            if (apiClient == null)
                apiClient = GetComponent<SRACEApiClient>();
            if (apiClient == null)
                apiClient = FindObjectOfType<SRACEApiClient>();

            if (apiClient != null)
                apiClient.SendOccupancy(occupancy);
        }

        private SRACEApiClient apiClient; // cached reference
    }
}
