// FanObject.cs — Code-generated ceiling fan with spinning blades
// Spawned by SRACEManager at each fan's (x, ceiling, y) position.
// Blades are primitive quads that rotate when active.

using UnityEngine;

namespace SRACE.Environment
{
    public class FanObject : MonoBehaviour
    {
        [Header("State")]
        public string fanId;
        public bool isActive = false;

        [Header("Visuals")]
        public float bladeLength = 0.6f;
        public float maxRPM = 180f; // degrees per second when fully spun up

        private float currentSpeed = 0f; // current rotation speed (deg/s)
        private float targetSpeed = 0f;
        private float spinUpRate = 5f; // lerp speed

        private Transform bladesRoot;
        private Renderer[] bladeRenderers;
        private Renderer hubRenderer;

        // Colors
        private static readonly Color activeColor = new Color(0.3f, 0.85f, 0.4f, 1f);  // green
        private static readonly Color inactiveColor = new Color(0.55f, 0.55f, 0.55f, 1f); // grey

        /// <summary>
        /// Build the fan geometry. Call once after instantiation.
        /// </summary>
        public void BuildGeometry(string id, float ceilingHeight)
        {
            fanId = id;

            // ── Downrod (cylinder from ceiling to hub) ──
            float rodLength = 0.3f;
            var rod = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            rod.name = "Downrod";
            rod.transform.SetParent(transform);
            rod.transform.localPosition = new Vector3(0, -rodLength / 2f, 0);
            rod.transform.localScale = new Vector3(0.04f, rodLength / 2f, 0.04f);
            ApplyMaterial(rod, new Color(0.4f, 0.4f, 0.4f));
            Object.Destroy(rod.GetComponent<Collider>());

            // ── Hub (center disc) ──
            var hub = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            hub.name = "Hub";
            hub.transform.SetParent(transform);
            hub.transform.localPosition = new Vector3(0, -rodLength, 0);
            hub.transform.localScale = new Vector3(0.15f, 0.03f, 0.15f);
            ApplyMaterial(hub, inactiveColor);
            hubRenderer = hub.GetComponent<Renderer>();
            Object.Destroy(hub.GetComponent<Collider>());

            // ── Blades root (this is what rotates) ──
            bladesRoot = new GameObject("BladesRoot").transform;
            bladesRoot.SetParent(transform);
            bladesRoot.localPosition = new Vector3(0, -rodLength - 0.02f, 0);

            // 3 blades, evenly spaced at 120°
            bladeRenderers = new Renderer[3];
            for (int i = 0; i < 3; i++)
            {
                float angle = i * 120f;
                var blade = GameObject.CreatePrimitive(PrimitiveType.Cube);
                blade.name = $"Blade_{i}";
                blade.transform.SetParent(bladesRoot);

                // Position blade center at half its length along the radial direction
                float rad = angle * Mathf.Deg2Rad;
                float cx = Mathf.Sin(rad) * bladeLength / 2f;
                float cz = Mathf.Cos(rad) * bladeLength / 2f;
                blade.transform.localPosition = new Vector3(cx, 0, cz);
                blade.transform.localRotation = Quaternion.Euler(0, angle, 0);
                blade.transform.localScale = new Vector3(0.12f, 0.015f, bladeLength);

                ApplyMaterial(blade, inactiveColor);
                bladeRenderers[i] = blade.GetComponent<Renderer>();
                Object.Destroy(blade.GetComponent<Collider>());
            }
        }

        /// <summary>Turn the fan on or off.</summary>
        public void SetActive(bool on)
        {
            isActive = on;
            targetSpeed = on ? maxRPM : 0f;

            var color = on ? activeColor : inactiveColor;
            foreach (var r in bladeRenderers)
                r.material.color = color;
            hubRenderer.material.color = color;
        }

        private void Update()
        {
            // Smooth spin up/down
            currentSpeed = Mathf.Lerp(currentSpeed, targetSpeed, Time.deltaTime * spinUpRate);

            if (currentSpeed > 0.1f && bladesRoot != null)
            {
                bladesRoot.Rotate(Vector3.up, currentSpeed * Time.deltaTime, Space.Self);
            }
        }

        private static void ApplyMaterial(GameObject obj, Color color)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Standard");

            var mat = new Material(shader);
            mat.color = color;
            mat.SetFloat("_Smoothness", 0.5f);
            obj.GetComponent<Renderer>().material = mat;
        }
    }
}
