// RoomConfigLoader.cs — JSON → RoomConfig deserializer
// Port of Python core/room_config.py load_config()
// Uses Unity's JsonUtility-compatible manual parsing for nested JSON.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace SRACE.Core
{
    /// <summary>
    /// Loads room configuration from a JSON string (TextAsset in Unity).
    /// Computes zone grid geometry and validates appliance bounds.
    /// </summary>
    public static class RoomConfigLoader
    {
        // ── JSON wrapper classes for Unity's JsonUtility ──
        // These mirror the JSON structure exactly.

        [Serializable]
        private class JsonRoot
        {
            public JsonRoom room;
            public JsonZones zones;
            public JsonFan[] fans;
            public JsonLight[] lights;
            public JsonProjector[] projectors;
            public JsonComfort comfort;
        }

        [Serializable]
        private class JsonRoom
        {
            public string name = "Unnamed Room";
            public float width;
            public float depth;
            public float ceiling_height = 3f;
            public float ambient_temp = 30f;
            public float ambient_co2 = 400f;
            public float ambient_lux = 50f;
        }

        [Serializable]
        private class JsonZones
        {
            public int cols;
            public int rows;
        }

        [Serializable]
        private class JsonFan
        {
            public string id;
            public float x;
            public float y;
            public float power_watts;
            public float airflow_radius;
            public float airflow_peak_ms;
        }

        [Serializable]
        private class JsonLight
        {
            public string id;
            public float x;
            public float y;
            public float power_watts;
            public float lumens;
            public float height_above_floor = 3f;
        }

        [Serializable]
        private class JsonProjector
        {
            public string id;
            public float x;
            public float y;
            public float power_watts;
            public float screen_lux;
            public float coverage_radius;
            public float height_above_floor = 3f;
        }

        [Serializable]
        private class JsonComfort
        {
            public float target_temp_c = 25f;
            public float target_lux = 300f;
            public float max_co2_ppm = 1000f;
            public float min_airflow_ms = 0.5f;
        }

        /// <summary>
        /// Load a RoomConfig from a JSON string.
        /// This is the single entry point — give it any valid room JSON and it
        /// returns a fully populated RoomConfig ready for physics + optimization.
        /// </summary>
        /// <param name="json">Raw JSON string (from TextAsset.text)</param>
        /// <returns>Fully populated RoomConfig with computed zone geometry.</returns>
        public static RoomConfig LoadFromJson(string json)
        {
            var data = JsonUtility.FromJson<JsonRoot>(json);

            if (data.room == null)
                throw new Exception("JSON missing 'room' section");
            if (data.zones == null)
                throw new Exception("JSON missing 'zones' section");

            float width = data.room.width;
            float depth = data.room.depth;
            float ceiling = data.room.ceiling_height;

            int nCols = data.zones.cols;
            int nRows = data.zones.rows;
            float zoneW = width / nCols;
            float zoneH = depth / nRows;

            // ── Build zone grid ──
            var zones = new List<Zone>();
            for (int r = 0; r < nRows; r++)
            {
                for (int c = 0; c < nCols; c++)
                {
                    float xMin = c * zoneW;
                    float xMax = (c + 1) * zoneW;
                    float yMin = r * zoneH;
                    float yMax = (r + 1) * zoneH;

                    var zone = new Zone(r, c, xMin, xMax, yMin, yMax);
                    zone.SetGridInfo(nCols);
                    zones.Add(zone);
                }
            }

            // ── Build fans ──
            var fans = new List<Fan>();
            if (data.fans != null)
            {
                for (int i = 0; i < data.fans.Length; i++)
                {
                    var fd = data.fans[i];
                    var fan = new Fan(fd.id, fd.x, fd.y, fd.power_watts,
                                     fd.airflow_radius, fd.airflow_peak_ms, i);
                    ValidateBounds(fan.x, fan.y, width, depth, "Fan", fan.id);
                    fans.Add(fan);
                }
            }

            // ── Build lights ──
            var lights = new List<Light>();
            if (data.lights != null)
            {
                for (int i = 0; i < data.lights.Length; i++)
                {
                    var ld = data.lights[i];
                    float h = ld.height_above_floor > 0 ? ld.height_above_floor : ceiling;
                    var light = new Light(ld.id, ld.x, ld.y, ld.power_watts,
                                          ld.lumens, h, i);
                    ValidateBounds(light.x, light.y, width, depth, "Light", light.id);
                    lights.Add(light);
                }
            }

            // ── Build projectors ──
            var projectors = new List<Projector>();
            if (data.projectors != null)
            {
                for (int i = 0; i < data.projectors.Length; i++)
                {
                    var pd = data.projectors[i];
                    float h = pd.height_above_floor > 0 ? pd.height_above_floor : ceiling;
                    var projector = new Projector(pd.id, pd.x, pd.y, pd.power_watts,
                                                  pd.screen_lux, pd.coverage_radius, h, i);
                    ValidateBounds(projector.x, projector.y, width, depth, "Projector", projector.id);
                    projectors.Add(projector);
                }
            }

            // ── Comfort params ──
            ComfortParams comfort;
            if (data.comfort != null)
            {
                comfort = new ComfortParams(
                    data.comfort.target_temp_c,
                    data.comfort.target_lux,
                    data.comfort.max_co2_ppm,
                    data.comfort.min_airflow_ms
                );
            }
            else
            {
                comfort = new ComfortParams(); // defaults
            }

            // ── Assemble config ──
            var config = new RoomConfig
            {
                name = data.room.name,
                width = width,
                depth = depth,
                ceilingHeight = ceiling,
                ambientTemp = data.room.ambient_temp,
                ambientCO2 = data.room.ambient_co2,
                ambientLux = data.room.ambient_lux,
                nZoneCols = nCols,
                nZoneRows = nRows,
                zones = zones,
                fans = fans,
                lights = lights,
                projectors = projectors,
                comfort = comfort,
            };

            return config;
        }

        /// <summary>
        /// Convenience: load from a Unity TextAsset.
        /// </summary>
        public static RoomConfig LoadFromTextAsset(TextAsset asset)
        {
            if (asset == null)
                throw new ArgumentNullException(nameof(asset), "Room config TextAsset is null");
            return LoadFromJson(asset.text);
        }

        private static void ValidateBounds(float x, float y, float width, float depth,
                                            string label, string id)
        {
            if (x < 0 || x > width)
                throw new Exception($"{label} '{id}' x={x} is outside room width [0, {width}]");
            if (y < 0 || y > depth)
                throw new Exception($"{label} '{id}' y={y} is outside room depth [0, {depth}]");
        }
    }
}
