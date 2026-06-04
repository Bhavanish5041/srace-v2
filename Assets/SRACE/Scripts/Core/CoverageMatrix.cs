// CoverageMatrix.cs — Binary coverage matrix from physics outputs
// Port of Python core/coverage.py
// Combines airflow, thermal, CO₂, and lux into a single binary matrix C[appliance][zone].

using System;
using System.Collections.Generic;
using UnityEngine;

namespace SRACE.Core
{
    /// <summary>
    /// Complete coverage analysis for the current room state.
    /// </summary>
    public class CoverageResult
    {
        public RoomConfig Config { get; private set; }

        /// <summary>(nAppliances, nZones) — 1 if appliance meaningfully covers zone.</summary>
        public int[,] Binary { get; private set; }

        /// <summary>(nFans, nZones) — raw airflow in m/s.</summary>
        public float[,] AirflowMatrix { get; private set; }

        /// <summary>(nFans, nZones) — ΔT cooling per fan.</summary>
        public float[,] ThermalImpact { get; private set; }

        /// <summary>(nFans, nZones) — CO₂ ppm reduction per fan.</summary>
        public float[,] CO2Reduction { get; private set; }

        /// <summary>(nLights, nZones) — lux contribution per light.</summary>
        public float[,] LuxMatrix { get; private set; }

        /// <summary>Set of occupied zone indices.</summary>
        public HashSet<int> OccupiedZones { get; private set; }

        /// <summary>Ordered appliance IDs (fans first, then lights, then projectors).</summary>
        public string[] ApplianceIds { get; private set; }

        /// <summary>Ordered power consumption array.</summary>
        public float[] ApplianceWatts { get; private set; }

        public int NAppliances => ApplianceIds.Length;

        public CoverageResult(
            RoomConfig cfg,
            float[,] airflowMatrix,
            float[,] thermalImpact,
            float[,] co2Reduction,
            float[,] luxMatrix,
            HashSet<int> occupiedZones)
        {
            Config = cfg;
            AirflowMatrix = airflowMatrix;
            ThermalImpact = thermalImpact;
            CO2Reduction = co2Reduction;
            LuxMatrix = luxMatrix;
            OccupiedZones = new HashSet<int>(occupiedZones);

            // Build appliance metadata (fans first, then lights, then projectors)
            ApplianceIds = new string[cfg.NAppliances];
            ApplianceWatts = new float[cfg.NAppliances];

            for (int i = 0; i < cfg.NFans; i++)
            {
                ApplianceIds[i] = cfg.fans[i].id;
                ApplianceWatts[i] = cfg.fans[i].powerWatts;
            }
            for (int i = 0; i < cfg.NLights; i++)
            {
                ApplianceIds[cfg.NFans + i] = cfg.lights[i].id;
                ApplianceWatts[cfg.NFans + i] = cfg.lights[i].powerWatts;
            }
            for (int i = 0; i < cfg.NProjectors; i++)
            {
                int idx = cfg.NFans + cfg.NLights + i;
                ApplianceIds[idx] = cfg.projectors[i].id;
                ApplianceWatts[idx] = cfg.projectors[i].powerWatts;
            }

            // Build binary coverage matrix
            Binary = BuildBinaryCoverage(cfg);
        }

        /// <summary>
        /// Construct binary coverage matrix.
        /// Fan covers zone if: airflow >= minAirflowMs OR thermalImpact >= 0.5°C
        /// Light covers zone if: lux >= 20 (meaningful contribution)
        /// Projector covers zone if: distance from projector to zone center <= coverageRadius
        /// </summary>
        private int[,] BuildBinaryCoverage(RoomConfig cfg)
        {
            int nFans = cfg.NFans;
            int nLights = cfg.NLights;
            int nProjectors = cfg.NProjectors;
            int nZones = cfg.NZones;
            int nTotal = nFans + nLights + nProjectors;

            var binary = new int[nTotal, nZones];

            // Fan coverage
            // Use 80% of min threshold for "meaningful contribution" — same logic
            // as lights using 20 lux instead of the full 300 target.
            // Without this, Gaussian decay causes adjacent zones to land at 0.49
            // when threshold is 0.5, making greedy select zero fans.
            float airflowThreshold = cfg.comfort.minAirflowMs * 0.8f;
            for (int fi = 0; fi < nFans; fi++)
            {
                for (int zi = 0; zi < nZones; zi++)
                {
                    bool airflowOk = AirflowMatrix[fi, zi] >= airflowThreshold;
                    bool thermalOk = ThermalImpact[fi, zi] >= 0.5f; // 0.5°C threshold
                    if (airflowOk || thermalOk)
                        binary[fi, zi] = 1;
                }
            }

            // Light coverage — 20 lux = meaningful contribution
            const float luxThreshold = 20f;
            for (int li = 0; li < nLights; li++)
            {
                for (int zi = 0; zi < nZones; zi++)
                {
                    if (LuxMatrix[li, zi] >= luxThreshold)
                        binary[nFans + li, zi] = 1;
                }
            }

            // Projector coverage — distance-based (zone center within coverage radius)
            for (int pi = 0; pi < nProjectors; pi++)
            {
                var proj = cfg.projectors[pi];
                for (int zi = 0; zi < nZones; zi++)
                {
                    var zone = cfg.zones[zi];
                    float dx = proj.x - zone.cx;
                    float dy = proj.y - zone.cy;
                    float dist = Mathf.Sqrt(dx * dx + dy * dy);
                    if (dist <= proj.coverageRadius)
                        binary[nFans + nLights + pi, zi] = 1;
                }
            }

            return binary;
        }

        /// <summary>Set of zone indices covered by a given appliance.</summary>
        public HashSet<int> CoveredZones(int applianceIdx)
        {
            var zones = new HashSet<int>();
            int nZones = Config.NZones;
            for (int zi = 0; zi < nZones; zi++)
            {
                if (Binary[applianceIdx, zi] == 1)
                    zones.Add(zi);
            }
            return zones;
        }

        /// <summary>Print coverage summary to Unity console.</summary>
        public void PrintSummary()
        {
            Debug.Log("──────────────────────────────────────");
            Debug.Log("  Coverage Matrix Summary");
            Debug.Log("──────────────────────────────────────");

            for (int ai = 0; ai < NAppliances; ai++)
            {
                var zonesCovered = CoveredZones(ai);
                var occCovered = new HashSet<int>(zonesCovered);
                occCovered.IntersectWith(OccupiedZones);
                Debug.Log($"  {ApplianceIds[ai],4}  ({ApplianceWatts[ai]:F0}W)  " +
                          $"covers {zonesCovered.Count,2} zones  ({occCovered.Count} occupied)");
            }

            var totalCoverable = new HashSet<int>();
            for (int ai = 0; ai < NAppliances; ai++)
                totalCoverable.UnionWith(CoveredZones(ai));

            var uncoverable = new HashSet<int>(OccupiedZones);
            uncoverable.ExceptWith(totalCoverable);
            if (uncoverable.Count > 0)
                Debug.LogWarning($"  ⚠ {uncoverable.Count} occupied zones not coverable by any appliance!");

            Debug.Log("──────────────────────────────────────");
        }
    }
}

