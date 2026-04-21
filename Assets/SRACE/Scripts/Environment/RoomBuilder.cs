// RoomBuilder.cs — Procedural classroom geometry from RoomConfig
// Spawns floor, walls, ceiling, and zone grid lines.
// All dimensions driven by JSON config — swap the config, get a different room.

using UnityEngine;
using SRACE.Core;

namespace SRACE.Environment
{
    public static class RoomBuilder
    {
        // ── Materials (created at runtime for URP) ──

        private static Material CreateMaterial(Color color, bool transparent = false)
        {
            // URP Lit shader
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Standard"); // fallback

            var mat = new Material(shader);
            mat.color = color;

            if (transparent)
            {
                // URP transparency setup
                mat.SetFloat("_Surface", 1); // Transparent
                mat.SetFloat("_Blend", 0);   // Alpha
                mat.SetFloat("_AlphaClip", 0);
                mat.SetOverrideTag("RenderType", "Transparent");
                mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite", 0);
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            }

            return mat;
        }

        /// <summary>
        /// Build the entire room geometry from a RoomConfig.
        /// Returns the root GameObject containing everything.
        /// </summary>
        public static GameObject Build(RoomConfig config)
        {
            var root = new GameObject("Classroom");

            BuildFloor(root.transform, config);
            BuildWalls(root.transform, config);
            BuildCeiling(root.transform, config);
            BuildZoneGrid(root.transform, config);

            return root;
        }

        // ── Floor ──

        private static void BuildFloor(Transform parent, RoomConfig cfg)
        {
            var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "Floor";
            floor.transform.SetParent(parent);

            // Thin slab at y=0
            float thickness = 0.05f;
            floor.transform.localScale = new Vector3(cfg.width, thickness, cfg.depth);
            floor.transform.localPosition = new Vector3(
                cfg.width / 2f, -thickness / 2f, cfg.depth / 2f);

            var mat = CreateMaterial(new Color(0.85f, 0.85f, 0.82f)); // warm light grey
            mat.SetFloat("_Smoothness", 0.3f);
            floor.GetComponent<Renderer>().material = mat;
        }

        // ── Walls ──

        private static void BuildWalls(Transform parent, RoomConfig cfg)
        {
            float w = cfg.width;
            float d = cfg.depth;
            float h = cfg.ceilingHeight;
            float wallThickness = 0.1f;

            var wallMat = CreateMaterial(new Color(0.92f, 0.91f, 0.88f)); // off-white
            wallMat.SetFloat("_Smoothness", 0.15f);

            // Back wall (z=0)
            CreateWall(parent, "Wall_Back", wallMat,
                new Vector3(w, h, wallThickness),
                new Vector3(w / 2f, h / 2f, -wallThickness / 2f));

            // Front wall (z=depth)
            CreateWall(parent, "Wall_Front", wallMat,
                new Vector3(w, h, wallThickness),
                new Vector3(w / 2f, h / 2f, d + wallThickness / 2f));

            // Left wall (x=0)
            CreateWall(parent, "Wall_Left", wallMat,
                new Vector3(wallThickness, h, d),
                new Vector3(-wallThickness / 2f, h / 2f, d / 2f));

            // Right wall (x=width)
            CreateWall(parent, "Wall_Right", wallMat,
                new Vector3(wallThickness, h, d),
                new Vector3(w + wallThickness / 2f, h / 2f, d / 2f));
        }

        private static void CreateWall(Transform parent, string name, Material mat,
                                        Vector3 scale, Vector3 position)
        {
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = name;
            wall.transform.SetParent(parent);
            wall.transform.localScale = scale;
            wall.transform.localPosition = position;
            wall.GetComponent<Renderer>().material = mat;
        }

        // ── Ceiling ──

        private static void BuildCeiling(Transform parent, RoomConfig cfg)
        {
            var ceiling = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ceiling.name = "Ceiling";
            ceiling.transform.SetParent(parent);

            float thickness = 0.05f;
            ceiling.transform.localScale = new Vector3(cfg.width, thickness, cfg.depth);
            ceiling.transform.localPosition = new Vector3(
                cfg.width / 2f, cfg.ceilingHeight + thickness / 2f, cfg.depth / 2f);

            // Slightly transparent so you can see fans/lights from outside
            var mat = CreateMaterial(new Color(0.95f, 0.95f, 0.93f, 0.6f), transparent: true);
            ceiling.GetComponent<Renderer>().material = mat;
        }

        // ── Zone Grid ──

        private static void BuildZoneGrid(Transform parent, RoomConfig cfg)
        {
            var gridRoot = new GameObject("ZoneGrid");
            gridRoot.transform.SetParent(parent);

            float w = cfg.width;
            float d = cfg.depth;
            float zoneW = w / cfg.nZoneCols;
            float zoneH = d / cfg.nZoneRows;
            float lineY = 0.005f; // just above floor
            float lineThickness = 0.03f;

            var lineMat = CreateMaterial(new Color(0.3f, 0.3f, 0.35f, 0.5f), transparent: true);

            // Vertical lines (along Z axis)
            for (int c = 0; c <= cfg.nZoneCols; c++)
            {
                float x = c * zoneW;
                var line = GameObject.CreatePrimitive(PrimitiveType.Cube);
                line.name = $"GridLine_V{c}";
                line.transform.SetParent(gridRoot.transform);
                line.transform.localScale = new Vector3(lineThickness, 0.01f, d);
                line.transform.localPosition = new Vector3(x, lineY, d / 2f);
                line.GetComponent<Renderer>().material = lineMat;
                // Remove collider — grid lines are visual only
                Object.Destroy(line.GetComponent<Collider>());
            }

            // Horizontal lines (along X axis)
            for (int r = 0; r <= cfg.nZoneRows; r++)
            {
                float z = r * zoneH;
                var line = GameObject.CreatePrimitive(PrimitiveType.Cube);
                line.name = $"GridLine_H{r}";
                line.transform.SetParent(gridRoot.transform);
                line.transform.localScale = new Vector3(w, 0.01f, lineThickness);
                line.transform.localPosition = new Vector3(w / 2f, lineY, z);
                line.GetComponent<Renderer>().material = lineMat;
                Object.Destroy(line.GetComponent<Collider>());
            }

            // Zone labels (using 3D Text / TextMesh — works without TMP import)
            for (int r = 0; r < cfg.nZoneRows; r++)
            {
                for (int c = 0; c < cfg.nZoneCols; c++)
                {
                    int idx = r * cfg.nZoneCols + c;
                    float cx = (c + 0.5f) * zoneW;
                    float cz = (r + 0.5f) * zoneH;

                    var labelObj = new GameObject($"ZoneLabel_{idx}");
                    labelObj.transform.SetParent(gridRoot.transform);
                    labelObj.transform.localPosition = new Vector3(cx, 0.02f, cz);
                    // Face upward (readable from above)
                    labelObj.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);

                    var tm = labelObj.AddComponent<TextMesh>();
                    tm.text = $"Z{idx}";
                    tm.fontSize = 48;
                    tm.characterSize = 0.08f;
                    tm.anchor = TextAnchor.MiddleCenter;
                    tm.alignment = TextAlignment.Center;
                    tm.color = new Color(0.4f, 0.4f, 0.45f, 0.7f);
                }
            }
        }
    }
}
