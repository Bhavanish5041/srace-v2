// ZoneHeatmap.cs — Color-coded floor overlay per zone
// Shows occupancy and coverage state with smooth color transitions.
//
// Colors:
//   Empty            → transparent
//   Occupied         → blue
//   Occupied+Covered → green
//   Occupied+NOT covered → red (warning)

using UnityEngine;
using SRACE.Core;

namespace SRACE.Environment
{
    public class ZoneHeatmap : MonoBehaviour
    {
        private GameObject[] zoneQuads;
        private Material[] zoneMats;
        private int nZones;

        // Zone state colors
        private static readonly Color emptyColor = new Color(0, 0, 0, 0);
        private static readonly Color occupiedColor = new Color(0.2f, 0.4f, 0.9f, 0.35f);
        private static readonly Color coveredColor = new Color(0.2f, 0.85f, 0.4f, 0.35f);
        private static readonly Color uncoveredColor = new Color(0.9f, 0.2f, 0.2f, 0.4f);

        private Color[] targetColors;
        private float lerpSpeed = 4f;

        /// <summary>
        /// Initialize the heatmap overlay. Creates one quad per zone.
        /// </summary>
        public void Initialize(RoomConfig cfg)
        {
            nZones = cfg.NZones;
            zoneQuads = new GameObject[nZones];
            zoneMats = new Material[nZones];
            targetColors = new Color[nZones];

            float zoneW = cfg.width / cfg.nZoneCols;
            float zoneH = cfg.depth / cfg.nZoneRows;

            for (int i = 0; i < nZones; i++)
            {
                var zone = cfg.zones[i];

                var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                quad.name = $"HeatmapZone_{i}";
                quad.transform.SetParent(transform);

                // Position just above floor, face up
                float cx = (zone.col + 0.5f) * zoneW;
                float cz = (zone.row + 0.5f) * zoneH;
                quad.transform.localPosition = new Vector3(cx, 0.012f, cz);
                quad.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
                quad.transform.localScale = new Vector3(zoneW * 0.92f, zoneH * 0.92f, 1f);

                // Remove collider
                Object.Destroy(quad.GetComponent<Collider>());

                // Transparent material (URP)
                var mat = CreateTransparentMaterial(emptyColor);
                quad.GetComponent<Renderer>().material = mat;

                zoneQuads[i] = quad;
                zoneMats[i] = mat;
                targetColors[i] = emptyColor;
            }
        }

        /// <summary>
        /// Update the heatmap with current occupancy and coverage data.
        /// </summary>
        /// <param name="occupancy">Per-zone person count (length = nZones).</param>
        /// <param name="coveredZones">Per-zone coverage flag (length = nZones).</param>
        public void UpdateHeatmap(int[] occupancy, bool[] coveredZones)
        {
            for (int i = 0; i < nZones; i++)
            {
                bool occupied = occupancy[i] > 0;
                bool covered = coveredZones != null && coveredZones[i];

                if (!occupied)
                    targetColors[i] = emptyColor;
                else if (covered)
                    targetColors[i] = coveredColor;
                else
                    targetColors[i] = uncoveredColor;
            }
        }

        /// <summary>
        /// Set all zones to occupied (blue) — useful for initial display.
        /// </summary>
        public void ShowOccupancy(int[] occupancy)
        {
            for (int i = 0; i < nZones; i++)
            {
                targetColors[i] = occupancy[i] > 0 ? occupiedColor : emptyColor;
            }
        }

        private void Update()
        {
            if (zoneMats == null) return;

            for (int i = 0; i < nZones; i++)
            {
                Color current = zoneMats[i].color;
                Color target = targetColors[i];
                if (current != target)
                {
                    zoneMats[i].color = Color.Lerp(current, target, Time.deltaTime * lerpSpeed);
                }
            }
        }

        private static Material CreateTransparentMaterial(Color color)
        {
            var shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");

            var mat = new Material(shader);
            mat.color = color;

            // Transparency
            mat.SetFloat("_Surface", 1);
            mat.SetFloat("_Blend", 0);
            mat.SetFloat("_AlphaClip", 0);
            mat.SetOverrideTag("RenderType", "Transparent");
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");

            return mat;
        }
    }
}
