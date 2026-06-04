// SRACEManager.cs — Main orchestrator MonoBehaviour
// Loads JSON config → builds room → spawns fans/lights/projectors → runs physics → updates visuals.
//
// Modes:
//   Local  — C# physics + greedy (keyboard presets 1-5, Space)
//   API    — Python backend via SRACEApiClient (polls /room_state every 5s)
//
// Demo controls:
//   [1] Empty room        [2] Single person (Z0)    [3] Center cluster
//   [4] Full room          [5] Front row only
//   [Space] Re-run optimizer and update visuals
//   [F] Toggle all fans    [L] Toggle all lights    [P] Toggle all projectors

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
        private ProjectorObject[] projectorObjects;
        private ZoneHeatmap heatmap;
        private ClassroomCamera orbitCamera;
        private PowerHUD powerHUD;

        private int[] currentOccupancy;
        private bool[] activeFans;
        private bool[] activeLights;
        private bool[] activeProjectors;

        // ── Persistent environment state (ticked each frame) ──
        private float[] zoneTemps;
        private float[] zoneCO2;
        private float[] zoneLux;
        private float[,] airflowMatrixCache; // static — only recomputed on config change
        private float[,] luxMatrixCache;     // static

        // ── Public accessors for API client ──
        public RoomConfig Config => config;
        public FanObject[] FanObjects => fanObjects;
        public LightObject[] LightObjects => lightObjects;
        public ProjectorObject[] ProjectorObjects => projectorObjects;
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
            SpawnProjectors();
            SetupHeatmap();
            SetupCamera();
            InitializeState();
            PrecomputeStaticPhysics();
            SetupHUD();

            config.PrintSummary();
            Debug.Log("SRACE Manager initialized. Press [1–5] for occupancy presets, [Space] to run optimizer.");
        }

        private void Update()
        {
            HandleInput();
            TickEnvironmentPhysics();
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

        private void SpawnProjectors()
        {
            projectorObjects = new ProjectorObject[config.NProjectors];

            if (config.NProjectors == 0) return;

            var projRoot = new GameObject("Projectors");
            projRoot.transform.SetParent(roomRoot.transform);

            for (int i = 0; i < config.NProjectors; i++)
            {
                var proj = config.projectors[i];
                var projGO = new GameObject($"Projector_{proj.id}");
                projGO.transform.SetParent(projRoot.transform);

                // JSON x,y → Unity x,z. Projector mounts at its specified height.
                projGO.transform.localPosition = new Vector3(
                    proj.x, proj.heightAboveFloor, proj.y);

                var projObj = projGO.AddComponent<ProjectorObject>();
                projObj.BuildGeometry(proj.id, proj.screenLux, proj.coverageRadius);
                projectorObjects[i] = projObj;
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
            activeProjectors = new bool[config.NProjectors];

            // Initialize persistent environment state
            zoneTemps = new float[config.NZones];
            zoneCO2 = new float[config.NZones];
            zoneLux = new float[config.NZones];
            for (int i = 0; i < config.NZones; i++)
            {
                zoneTemps[i] = config.ambientTemp;
                zoneCO2[i] = config.ambientCO2;
                zoneLux[i] = config.ambientLux;
            }
        }

        /// <summary>
        /// Pre-compute airflow and lux matrices (static — only depend on room layout).
        /// </summary>
        private void PrecomputeStaticPhysics()
        {
            airflowMatrixCache = AirflowModel.ComputeAirflowMatrix(config);
            luxMatrixCache = LightingModel.ComputeLuxMatrix(config);
            Debug.Log($"✓ Static physics: airflow({config.NFans}×{config.NZones}), lux({config.NLights}×{config.NZones})");
        }

        /// <summary>
        /// Run simplified physics every frame for live temp/CO₂/lux tracking.
        /// Uses QuickTick methods (same as PPO gym_env) for speed.
        /// </summary>
        private void TickEnvironmentPhysics()
        {
            // Compute total airflow from active fans
            float[] airflowPerZone = AirflowModel.TotalAirflowPerZone(airflowMatrixCache, activeFans);

            // Update temperature
            ThermalModel.QuickTick(zoneTemps, airflowPerZone, currentOccupancy,
                config.comfort.targetTempC, config.ambientTemp);

            // Update CO₂
            CO2Model.QuickTick(zoneCO2, airflowPerZone, currentOccupancy, config.ambientCO2);

            // Update lux
            bool anyLightsOn = false;
            foreach (var l in activeLights) if (l) { anyLightsOn = true; break; }
            if (anyLightsOn)
                zoneLux = LightingModel.TotalLuxPerZone(luxMatrixCache, activeLights, config.ambientLux);
            else
                for (int i = 0; i < zoneLux.Length; i++) zoneLux[i] = config.ambientLux;

            // Update HUD with live data
            UpdateHUD();
        }

        private void SetupHUD()
        {
            var hudGO = new GameObject("PowerHUD");
            hudGO.transform.SetParent(roomRoot.transform);
            powerHUD = hudGO.AddComponent<PowerHUD>();

            float maxW = config.MaxPowerWatts;
            powerHUD.UpdateStats(0f, maxW, 100f, 0, 0, 0,
                config.NFans, config.NLights, config.NProjectors, 0,
                config.ambientTemp, config.ambientCO2, "⏳ Ready");
        }

        private void UpdateHUD(string scenario = null)
        {
            if (powerHUD == null) return;

            float totalPower = 0f;
            int fOn = 0, lOn = 0, pOn = 0, people = 0;
            for (int i = 0; i < activeFans.Length; i++)
                if (activeFans[i]) { fOn++; totalPower += config.fans[i].powerWatts; }
            for (int i = 0; i < activeLights.Length; i++)
                if (activeLights[i]) { lOn++; totalPower += config.lights[i].powerWatts; }
            for (int i = 0; i < activeProjectors.Length; i++)
                if (activeProjectors[i]) { pOn++; totalPower += config.projectors[i].powerWatts; }
            for (int i = 0; i < currentOccupancy.Length; i++)
                people += currentOccupancy[i];

            float maxW = config.MaxPowerWatts;
            float saved = maxW > 0 ? (1f - totalPower / maxW) * 100f : 100f;

            // Compute average temp and CO₂ from live zone state
            float avgTemp = 0f, avgCO2 = 0f;
            for (int i = 0; i < config.NZones; i++)
            {
                avgTemp += zoneTemps[i];
                avgCO2 += zoneCO2[i];
            }
            avgTemp /= config.NZones;
            avgCO2 /= config.NZones;

            powerHUD.UpdateStats(totalPower, maxW, saved, fOn, lOn, pOn,
                config.NFans, config.NLights, config.NProjectors, people,
                avgTemp, avgCO2, scenario ?? scenarioLabel);
        }

        private string scenarioLabel = "⏳ Ready";

        // ══════════════════════════════════════════
        //  PHYSICS + OPTIMIZER
        // ══════════════════════════════════════════

        private void RunPipelineAndUpdateVisuals()
        {
            // 1. Use cached static physics matrices
            var airflowMatrix = airflowMatrixCache;
            var luxMatrix = luxMatrixCache;

            // 2. Run full RK4 ODE thermal + CO₂ simulations for accurate marginal analysis
            var thermalImpact = ThermalModel.SimulateThermal(
                config, airflowMatrix, currentOccupancy, zoneTemps);
            var co2Reduction = CO2Model.SimulateCO2(
                config, airflowMatrix, currentOccupancy, zoneCO2);

            Debug.Log($"🌡 Thermal ODE: peak ΔT = {MaxValue(thermalImpact):F2}°C  |  " +
                      $"🌬 CO₂ ODE: peak ΔC = {MaxValue(co2Reduction):F0} ppm");

            // 3. Find occupied zones
            var occupiedZones = new HashSet<int>();
            for (int i = 0; i < config.NZones; i++)
            {
                if (currentOccupancy[i] > 0)
                    occupiedZones.Add(i);
            }

            // 4. Build coverage result
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
                    bool applianceOn;
                    if (ai < config.NFans)
                        applianceOn = activeFans[ai];
                    else if (ai < config.NFans + config.NLights)
                        applianceOn = activeLights[ai - config.NFans];
                    else
                        applianceOn = activeProjectors[ai - config.NFans - config.NLights];

                    if (applianceOn && coverage.Binary[ai, zi] == 1)
                    {
                        coveredZones[zi] = true;
                        break;
                    }
                }
            }
            heatmap.UpdateHeatmap(currentOccupancy, coveredZones);

            // 7. Log power and environment state
            float totalPower = 0;
            for (int i = 0; i < config.NFans; i++)
                if (activeFans[i]) totalPower += config.fans[i].powerWatts;
            for (int i = 0; i < config.NLights; i++)
                if (activeLights[i]) totalPower += config.lights[i].powerWatts;
            for (int i = 0; i < config.NProjectors; i++)
                if (activeProjectors[i]) totalPower += config.projectors[i].powerWatts;

            float savings = config.MaxPowerWatts > 0
                ? (1f - totalPower / config.MaxPowerWatts) * 100f : 0f;

            float avgT = 0f, avgC = 0f;
            for (int i = 0; i < config.NZones; i++) { avgT += zoneTemps[i]; avgC += zoneCO2[i]; }
            avgT /= config.NZones; avgC /= config.NZones;

            Debug.Log($"⚡ Power: {totalPower:F0}W / {config.MaxPowerWatts:F0}W  " +
                      $"({savings:F1}% savings)  |  Zones: {occupiedZones.Count}/{config.NZones}  " +
                      $"Temp: {avgT:F1}°C  CO₂: {avgC:F0}ppm");
        }

        /// <summary>Helper: find max value in a 2D array.</summary>
        private static float MaxValue(float[,] arr)
        {
            float max = float.MinValue;
            int r = arr.GetLength(0), c = arr.GetLength(1);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    if (arr[i, j] > max) max = arr[i, j];
            return max;
        }

        /// <summary>
        /// Greedy activation with SEPARATE phases for fans, lights, and projectors.
        /// Fans cover airflow/thermal needs, lights cover illumination,
        /// projectors cover additional lux in their coverage radius.
        /// Running them together causes cheaper appliances to "steal" coverage.
        /// </summary>
        private void GreedyActivate(HashSet<int> occupiedZones)
        {
            // Reset all
            for (int i = 0; i < activeFans.Length; i++) activeFans[i] = false;
            for (int i = 0; i < activeLights.Length; i++) activeLights[i] = false;
            for (int i = 0; i < activeProjectors.Length; i++) activeProjectors[i] = false;

            if (occupiedZones.Count == 0)
            {
                ApplyApplianceStates();
                return;
            }

            int fanEnd = config.NFans;
            int lightEnd = config.NFans + config.NLights;
            int projEnd = config.NAppliances;

            // Phase 1: Greedy over FANS only (airflow coverage)
            GreedySubset(occupiedZones, 0, fanEnd);

            // Phase 2: Greedy over LIGHTS only (illumination coverage)
            GreedySubset(occupiedZones, fanEnd, lightEnd);

            // Phase 3: Greedy over PROJECTORS only (additional lux coverage)
            if (config.NProjectors > 0)
                GreedySubset(occupiedZones, lightEnd, projEnd);

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
                    bool alreadyOn;
                    if (ai < config.NFans)
                        alreadyOn = activeFans[ai];
                    else if (ai < config.NFans + config.NLights)
                        alreadyOn = activeLights[ai - config.NFans];
                    else
                        alreadyOn = activeProjectors[ai - config.NFans - config.NLights];

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
                else if (bestApp < config.NFans + config.NLights)
                    activeLights[bestApp - config.NFans] = true;
                else
                    activeProjectors[bestApp - config.NFans - config.NLights] = true;

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
            for (int i = 0; i < projectorObjects.Length; i++)
                projectorObjects[i].SetActive(activeProjectors[i]);

            UpdateHUD();
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
            bool[] projectorStates,
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

            // Update projectors
            if (projectorStates != null && projectorStates.Length == projectorObjects.Length)
            {
                activeProjectors = projectorStates;
                for (int i = 0; i < projectorObjects.Length; i++)
                    projectorObjects[i].SetActive(activeProjectors[i]);
            }

            // Update heatmap
            if (coveredZones != null && coveredZones.Length == config.NZones)
                heatmap.UpdateHeatmap(currentOccupancy, coveredZones);

            Debug.Log($"🌐 API → Power: {totalPowerW:F0}W  |  Saved: {powerSavedPct:F1}%  |  " +
                      $"Fans: {CountTrue(activeFans)}/{activeFans.Length}  " +
                      $"Lights: {CountTrue(activeLights)}/{activeLights.Length}  " +
                      $"Projectors: {CountTrue(activeProjectors)}/{activeProjectors.Length}");

            UpdateHUD("🌐 API Mode");
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

            // Toggle all projectors
            if (Input.GetKeyDown(KeyCode.P))
            {
                if (activeProjectors.Length > 0)
                {
                    bool anyOn = false;
                    foreach (var p in activeProjectors) if (p) { anyOn = true; break; }
                    bool newState = !anyOn;
                    for (int i = 0; i < activeProjectors.Length; i++) activeProjectors[i] = newState;
                    ApplyApplianceStates();
                    Debug.Log($"All projectors {(newState ? "ON" : "OFF")}");
                }
            }
        }

        private void SetOccupancy(string scenarioName, int[] occupancy)
        {
            currentOccupancy = occupancy;
            scenarioLabel = scenarioName;
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
