// LightObject.cs — Ceiling light fixture with Unity Light component
// Spawned by SRACEManager at each light's (x, ceiling, y) position.
// Uses a Point Light for actual illumination + a visual panel.

using UnityEngine;

namespace SRACE.Environment
{
    public class LightObject : MonoBehaviour
    {
        [Header("State")]
        public string lightId;
        public bool isActive = false;

        [Header("Config")]
        public float lumens = 3200f;

        private UnityEngine.Light pointLight;
        private Renderer panelRenderer;
        private Material panelMat;

        // Colors
        private static readonly Color warmWhite = new Color(1f, 0.95f, 0.82f, 1f);
        private static readonly Color offColor = new Color(0.3f, 0.3f, 0.3f, 1f);
        private static readonly Color emissiveOn = new Color(1f, 0.92f, 0.7f) * 2f;

        /// <summary>
        /// Build the light fixture geometry. Call once after instantiation.
        /// </summary>
        public void BuildGeometry(string id, float configLumens)
        {
            lightId = id;
            lumens = configLumens;

            // ── Fixture panel (flat rectangular tube light) ──
            var panel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            panel.name = "LightPanel";
            panel.transform.SetParent(transform);
            panel.transform.localPosition = new Vector3(0, -0.02f, 0);
            panel.transform.localScale = new Vector3(0.6f, 0.03f, 0.15f);
            Object.Destroy(panel.GetComponent<Collider>());

            // Create URP material with emission support
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Standard");

            panelMat = new Material(shader);
            panelMat.color = offColor;
            panelMat.SetFloat("_Smoothness", 0.8f);
            panel.GetComponent<Renderer>().material = panelMat;
            panelRenderer = panel.GetComponent<Renderer>();

            // ── Mounting bracket ──
            var bracket = GameObject.CreatePrimitive(PrimitiveType.Cube);
            bracket.name = "Bracket";
            bracket.transform.SetParent(transform);
            bracket.transform.localPosition = Vector3.zero;
            bracket.transform.localScale = new Vector3(0.08f, 0.02f, 0.08f);
            Object.Destroy(bracket.GetComponent<Collider>());

            var bracketShader = Shader.Find("Universal Render Pipeline/Lit");
            if (bracketShader == null)
                bracketShader = Shader.Find("Standard");
            var bracketMat = new Material(bracketShader);
            bracketMat.color = new Color(0.5f, 0.5f, 0.5f);
            bracket.GetComponent<Renderer>().material = bracketMat;

            // ── Unity Point Light (the actual light source) ──
            var lightObj = new GameObject("PointLight");
            lightObj.transform.SetParent(transform);
            lightObj.transform.localPosition = new Vector3(0, -0.05f, 0);

            pointLight = lightObj.AddComponent<UnityEngine.Light>();
            pointLight.type = LightType.Point;
            pointLight.color = warmWhite;
            // Convert lumens → Unity intensity (rough mapping)
            // URP uses physical lights; ~800 lumens ≈ intensity 1 for point lights
            pointLight.intensity = lumens / 800f;
            pointLight.range = 6f; // 6m range covers nearby zones
            pointLight.shadows = LightShadows.Soft;
            pointLight.enabled = false; // start off
        }

        /// <summary>Turn the light on or off.</summary>
        public void SetActive(bool on)
        {
            isActive = on;
            pointLight.enabled = on;

            if (on)
            {
                panelMat.color = warmWhite;
                panelMat.EnableKeyword("_EMISSION");
                panelMat.SetColor("_EmissionColor", emissiveOn);
            }
            else
            {
                panelMat.color = offColor;
                panelMat.DisableKeyword("_EMISSION");
                panelMat.SetColor("_EmissionColor", Color.black);
            }
        }
    }
}
