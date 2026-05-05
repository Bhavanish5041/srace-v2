// SRACEApiClient.cs — Unity ↔ Python backend bridge
// Polls GET http://localhost:8000/room_state every 5 seconds via UnityWebRequest.
// Parses the JSON response and pushes appliance states into SRACEManager.
//
// Also sends POST /set_occupancy when keyboard presets (1–5) change occupancy,
// so Python physics stay in sync with the Unity demo.
//
// Usage:
//   1. Attach this component to the same GameObject as SRACEManager
//      (or any GameObject in the scene).
//   2. Drag the SRACEManager reference in the Inspector.
//   3. Start the Python backend: uvicorn backend.api:app --port 8000
//   4. Press Play in Unity.

using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using SRACE.Core;

namespace SRACE.Core
{
    public class SRACEApiClient : MonoBehaviour
    {
        [Header("Connection")]
        [Tooltip("Base URL of the SRACE Python backend.")]
        public string apiBaseUrl = "http://localhost:8000";

        [Tooltip("How often to poll /room_state (seconds).")]
        public float pollInterval = 5f;

        [Header("References")]
        [Tooltip("Drag the SRACEManager component here.")]
        public SRACEManager manager;

        [Header("State")]
        [Tooltip("Enable/disable API polling at runtime.")]
        public bool pollingEnabled = true;

        private bool isPolling = false;
        private int successCount = 0;
        private int errorCount = 0;

        // ══════════════════════════════════════════
        //  LIFECYCLE
        // ══════════════════════════════════════════

        private void Start()
        {
            // Auto-find SRACEManager if not assigned
            if (manager == null)
                manager = GetComponent<SRACEManager>();
            if (manager == null)
                manager = FindObjectOfType<SRACEManager>();
            if (manager == null)
            {
                Debug.LogError("[SRACEApiClient] No SRACEManager found! Disabling.");
                enabled = false;
                return;
            }

            Debug.Log($"[SRACEApiClient] Starting — polling {apiBaseUrl}/room_state every {pollInterval}s");
            StartCoroutine(PollLoop());
        }

        // ══════════════════════════════════════════
        //  POLLING LOOP
        // ══════════════════════════════════════════

        private IEnumerator PollLoop()
        {
            // Wait a frame for SRACEManager to finish Start()
            yield return null;

            while (true)
            {
                if (pollingEnabled && !isPolling)
                {
                    yield return StartCoroutine(FetchRoomState());
                }

                yield return new WaitForSeconds(pollInterval);
            }
        }

        // ══════════════════════════════════════════
        //  GET /room_state
        // ══════════════════════════════════════════

        private IEnumerator FetchRoomState()
        {
            isPolling = true;
            string url = $"{apiBaseUrl}/room_state";

            using (var request = UnityWebRequest.Get(url))
            {
                request.timeout = 5; // 5-second timeout

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.ConnectionError ||
                    request.result == UnityWebRequest.Result.ProtocolError)
                {
                    errorCount++;
                    if (errorCount <= 3 || errorCount % 10 == 0)
                    {
                        Debug.LogWarning(
                            $"[SRACEApiClient] GET /room_state failed ({errorCount}x): {request.error}\n" +
                            $"  Is the backend running? → uvicorn backend.api:app --port 8000");
                    }
                }
                else
                {
                    successCount++;
                    string json = request.downloadHandler.text;

                    try
                    {
                        ParseAndApply(json);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"[SRACEApiClient] JSON parse error: {ex.Message}");
                    }
                }
            }

            isPolling = false;
        }

        // ══════════════════════════════════════════
        //  JSON PARSING
        // ══════════════════════════════════════════

        // Unity's JsonUtility doesn't handle the nested response well,
        // so we use wrapper classes that mirror the API response exactly.

        [Serializable]
        private class ApiRoomState
        {
            public string room_name;
            public ApiDimensions dimensions;
            public ApiZoneInfo[] zones;
            public ApiApplianceInfo[] appliances;
            public float total_power_watts;
            public float max_power_watts;
            public float power_saved_pct;
            public string solver;
            public float solve_time_ms;
            public int occupied_zone_count;
            public int total_people;
        }

        [Serializable]
        private class ApiDimensions
        {
            public float width;
            public float depth;
            public float ceiling_height;
        }

        [Serializable]
        private class ApiZoneInfo
        {
            public int index;
            public int row;
            public int col;
            public float cx;
            public float cy;
            public int occupancy;
            public bool covered;
            public float airflow_ms;
            public float lux;
        }

        [Serializable]
        private class ApiApplianceInfo
        {
            public string id;
            public string type; // "fan" or "light"
            public float x;
            public float y;
            public float power_watts;
            public bool active;
        }

        private void ParseAndApply(string json)
        {
            var state = JsonUtility.FromJson<ApiRoomState>(json);

            if (state == null)
            {
                Debug.LogWarning("[SRACEApiClient] Received null state from API.");
                return;
            }

            var config = manager.Config;
            if (config == null) return;

            // ── Extract occupancy ──
            int nZones = config.NZones;
            int[] occupancy = new int[nZones];
            bool[] coveredZones = new bool[nZones];

            if (state.zones != null)
            {
                for (int i = 0; i < state.zones.Length && i < nZones; i++)
                {
                    occupancy[state.zones[i].index] = state.zones[i].occupancy;
                    coveredZones[state.zones[i].index] = state.zones[i].covered;
                }
            }

            // ── Extract appliance states ──
            int nFans = config.NFans;
            int nLights = config.NLights;
            bool[] fanStates = new bool[nFans];
            bool[] lightStates = new bool[nLights];

            // Track fan/light indices separately
            int fanIdx = 0;
            int lightIdx = 0;

            if (state.appliances != null)
            {
                for (int i = 0; i < state.appliances.Length; i++)
                {
                    var app = state.appliances[i];
                    if (app.type == "fan" && fanIdx < nFans)
                    {
                        fanStates[fanIdx] = app.active;
                        fanIdx++;
                    }
                    else if (app.type == "light" && lightIdx < nLights)
                    {
                        lightStates[lightIdx] = app.active;
                        lightIdx++;
                    }
                }
            }

            // ── Push to SRACEManager ──
            manager.ApplyApiState(
                occupancy,
                fanStates,
                lightStates,
                coveredZones,
                state.total_power_watts,
                state.power_saved_pct
            );

            if (successCount <= 1 || successCount % 20 == 0)
            {
                Debug.Log($"[SRACEApiClient] ✓ Poll #{successCount} — " +
                          $"{state.total_people} people, {state.occupied_zone_count} zones occupied, " +
                          $"{state.power_saved_pct:F1}% saved ({state.solve_time_ms:F1}ms solve)");
            }
        }

        // ══════════════════════════════════════════
        //  POST /set_occupancy (send Unity state → Python)
        // ══════════════════════════════════════════

        /// <summary>
        /// Send occupancy data to the Python backend.
        /// Call this from SRACEManager when the user presses a preset key (1–5),
        /// so Python physics recalculate for the new occupancy.
        /// </summary>
        public void SendOccupancy(int[] zonePeople)
        {
            StartCoroutine(PostOccupancy(zonePeople));
        }

        private IEnumerator PostOccupancy(int[] zonePeople)
        {
            string url = $"{apiBaseUrl}/set_occupancy";

            // Build JSON: {"zone_people": [0, 0, 3, ...]}
            var sb = new StringBuilder();
            sb.Append("{\"zone_people\":[");
            for (int i = 0; i < zonePeople.Length; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(zonePeople[i]);
            }
            sb.Append("]}");

            byte[] bodyRaw = Encoding.UTF8.GetBytes(sb.ToString());

            using (var request = new UnityWebRequest(url, "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                request.timeout = 5;

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.ConnectionError ||
                    request.result == UnityWebRequest.Result.ProtocolError)
                {
                    Debug.LogWarning($"[SRACEApiClient] POST /set_occupancy failed: {request.error}");
                }
                else
                {
                    Debug.Log($"[SRACEApiClient] ✓ Occupancy sent to backend → " +
                              $"{request.downloadHandler.text}");
                }
            }
        }

        // ══════════════════════════════════════════
        //  PUBLIC HELPERS
        // ══════════════════════════════════════════

        /// <summary>Status string for debug UI / HUD.</summary>
        public string GetStatusText()
        {
            if (!pollingEnabled) return "API: Disabled";
            if (errorCount > 0 && successCount == 0)
                return $"API: Connecting... ({errorCount} errors)";
            return $"API: Connected (poll #{successCount})";
        }
    }
}
