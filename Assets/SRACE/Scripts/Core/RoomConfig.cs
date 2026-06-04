// RoomConfig.cs — SRACE v2 Core Data Model
// Direct port of Python core/room_config.py
// Defines all data structures for room layout, appliances, and comfort targets.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace SRACE.Core
{
    /// <summary>
    /// A rectangular zone within the room grid.
    /// </summary>
    [System.Serializable]
    public class Zone
    {
        public int row;
        public int col;
        public float xMin;
        public float xMax;
        public float yMin;
        public float yMax;
        public float cx; // center x
        public float cy; // center y
        public float area; // m²

        private int totalCols;

        /// <summary>Flat index for matrix operations (row-major).</summary>
        public int Index => row * totalCols + col;

        public void SetGridInfo(int cols)
        {
            totalCols = cols;
        }

        public Zone(int row, int col, float xMin, float xMax, float yMin, float yMax)
        {
            this.row = row;
            this.col = col;
            this.xMin = xMin;
            this.xMax = xMax;
            this.yMin = yMin;
            this.yMax = yMax;
            this.cx = (xMin + xMax) / 2f;
            this.cy = (yMin + yMax) / 2f;
            this.area = (xMax - xMin) * (yMax - yMin);
        }
    }

    /// <summary>
    /// A ceiling fan appliance.
    /// </summary>
    [System.Serializable]
    public class Fan
    {
        public string id;
        public float x;
        public float y;
        public float powerWatts;
        public float airflowRadius;  // metres — 95% of airflow within this radius
        public float airflowPeakMs;  // peak airflow speed at fan centre (m/s)
        public int idx;              // index in the appliance list

        public Fan(string id, float x, float y, float powerWatts,
                   float airflowRadius, float airflowPeakMs, int idx = 0)
        {
            this.id = id;
            this.x = x;
            this.y = y;
            this.powerWatts = powerWatts;
            this.airflowRadius = airflowRadius;
            this.airflowPeakMs = airflowPeakMs;
            this.idx = idx;
        }
    }

    /// <summary>
    /// A ceiling light appliance.
    /// </summary>
    [System.Serializable]
    public class Light
    {
        public string id;
        public float x;
        public float y;
        public float powerWatts;
        public float lumens;
        public float heightAboveFloor; // metres
        public int idx;

        public Light(string id, float x, float y, float powerWatts,
                     float lumens, float heightAboveFloor, int idx = 0)
        {
            this.id = id;
            this.x = x;
            this.y = y;
            this.powerWatts = powerWatts;
            this.lumens = lumens;
            this.heightAboveFloor = heightAboveFloor;
            this.idx = idx;
        }
    }

    /// <summary>
    /// A ceiling-mounted projector appliance.
    /// </summary>
    [System.Serializable]
    public class Projector
    {
        public string id;
        public float x;
        public float y;
        public float powerWatts;
        public float screenLux;       // lux output on screen area
        public float coverageRadius;  // metres — zones within this radius get lux
        public float heightAboveFloor;
        public int idx;

        public Projector(string id, float x, float y, float powerWatts,
                         float screenLux, float coverageRadius,
                         float heightAboveFloor, int idx = 0)
        {
            this.id = id;
            this.x = x;
            this.y = y;
            this.powerWatts = powerWatts;
            this.screenLux = screenLux;
            this.coverageRadius = coverageRadius;
            this.heightAboveFloor = heightAboveFloor;
            this.idx = idx;
        }
    }

    /// <summary>
    /// Target comfort thresholds.
    /// </summary>
    [System.Serializable]
    public class ComfortParams
    {
        public float targetTempC;
        public float targetLux;
        public float maxCO2Ppm;
        public float minAirflowMs;

        public ComfortParams(float targetTempC = 25f, float targetLux = 300f,
                             float maxCO2Ppm = 1000f, float minAirflowMs = 0.5f)
        {
            this.targetTempC = targetTempC;
            this.targetLux = targetLux;
            this.maxCO2Ppm = maxCO2Ppm;
            this.minAirflowMs = minAirflowMs;
        }
    }

    /// <summary>
    /// Complete room configuration — loaded from JSON, works for any room.
    /// </summary>
    [System.Serializable]
    public class RoomConfig
    {
        public string name;
        public float width;        // metres (x-axis)
        public float depth;        // metres (y-axis / z-axis in Unity)
        public float ceilingHeight; // metres
        public float ambientTemp;  // °C
        public float ambientCO2;   // ppm
        public float ambientLux;   // lux

        public int nZoneCols;
        public int nZoneRows;

        public List<Zone> zones = new List<Zone>();
        public List<Fan> fans = new List<Fan>();
        public List<Light> lights = new List<Light>();
        public List<Projector> projectors = new List<Projector>();
        public ComfortParams comfort;

        // ── Computed Properties ──

        public int NZones => nZoneRows * nZoneCols;
        public int NFans => fans.Count;
        public int NLights => lights.Count;
        public int NProjectors => projectors.Count;
        public int NAppliances => NFans + NLights + NProjectors;

        /// <summary>All appliances in order: fans, then lights, then projectors.</summary>
        public List<object> AllAppliances
        {
            get
            {
                var all = new List<object>();
                all.AddRange(fans);
                all.AddRange(lights);
                all.AddRange(projectors);
                return all;
            }
        }

        /// <summary>Get power consumption of an appliance by its flat index.</summary>
        public float GetApplianceWatts(int index)
        {
            if (index < NFans)
                return fans[index].powerWatts;
            int afterFans = index - NFans;
            if (afterFans < NLights)
                return lights[afterFans].powerWatts;
            return projectors[afterFans - NLights].powerWatts;
        }

        /// <summary>Get ID of an appliance by its flat index.</summary>
        public string GetApplianceId(int index)
        {
            if (index < NFans)
                return fans[index].id;
            int afterFans = index - NFans;
            if (afterFans < NLights)
                return lights[afterFans].id;
            return projectors[afterFans - NLights].id;
        }

        /// <summary>Total max power if everything is on.</summary>
        public float MaxPowerWatts
        {
            get
            {
                float total = 0;
                foreach (var f in fans) total += f.powerWatts;
                foreach (var l in lights) total += l.powerWatts;
                foreach (var p in projectors) total += p.powerWatts;
                return total;
            }
        }

        /// <summary>Get zone by grid coordinates.</summary>
        public Zone ZoneAt(int row, int col) => zones[row * nZoneCols + col];

        /// <summary>Get zone by flat index.</summary>
        public Zone ZoneFlat(int idx) => zones[idx];

        /// <summary>Print summary to Unity console.</summary>
        public void PrintSummary()
        {
            float fanWatts = 0, lightWatts = 0, projWatts = 0;
            foreach (var f in fans) fanWatts += f.powerWatts;
            foreach (var l in lights) lightWatts += l.powerWatts;
            foreach (var p in projectors) projWatts += p.powerWatts;

            Debug.Log($"═══════════════════════════════════════════");
            Debug.Log($"  SRACE Room: {name}");
            Debug.Log($"═══════════════════════════════════════════");
            Debug.Log($"  Dimensions : {width} × {depth} m  (ceiling {ceilingHeight} m)");
            Debug.Log($"  Zone grid  : {nZoneRows} rows × {nZoneCols} cols = {NZones} zones");
            Debug.Log($"  Fans       : {NFans}  (total {fanWatts:F0} W)");
            Debug.Log($"  Lights     : {NLights}  (total {lightWatts:F0} W)");
            Debug.Log($"  Projectors : {NProjectors}  (total {projWatts:F0} W)");
            Debug.Log($"  Max power  : {MaxPowerWatts:F0} W");
            Debug.Log($"  Comfort    : {comfort.targetTempC}°C  {comfort.targetLux} lux" +
                      $"  <{comfort.maxCO2Ppm} ppm CO₂");
            Debug.Log($"═══════════════════════════════════════════");
        }
    }
}
