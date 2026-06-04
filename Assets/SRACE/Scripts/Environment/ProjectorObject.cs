// ProjectorObject.cs — Ceiling-mounted projector with Spot Light
// Spawned by SRACEManager at each projector's (x, ceiling, y) position.
// Uses a Spot Light pointing downward + a visual body to represent the projector.
// Cyan/blue tint distinguishes projectors from regular warm-white lights.

using UnityEngine;

namespace SRACE.Environment
{
    public class ProjectorObject : MonoBehaviour
    {
        [Header("State")]
        public string projectorId;
        public bool isActive = false;

        [Header("Config")]
        public float screenLux = 200f;
        public float coverageRadius = 4f;

        private UnityEngine.Light spotLight;
        private Renderer bodyRenderer;
        private Renderer lensRenderer;
        private Material bodyMat;
        private Material lensMat;

        // Colors — cyan/blue tint to distinguish from warm-white lights
        private static readonly Color projectorBlue = new Color(0.4f, 0.7f, 1f, 1f);
        private static readonly Color offColor = new Color(0.25f, 0.25f, 0.3f, 1f);
        private static readonly Color emissiveOn = new Color(0.3f, 0.6f, 1f) * 2.5f;
        private static readonly Color lensGlow = new Color(0.5f, 0.85f, 1f) * 3f;

        /// <summary>
        /// Build the projector geometry. Call once after instantiation.
        /// </summary>
        public void BuildGeometry(string id, float configScreenLux, float configRadius)
        {
            projectorId = id;
            screenLux = configScreenLux;
            coverageRadius = configRadius;

            // ── Projector body (rectangular box, wider than lights) ──
            var body = GameObject.CreatePrimitive(PrimitiveType.Cube);
            body.name = "ProjectorBody";
            body.transform.SetParent(transform);
            body.transform.localPosition = new Vector3(0, -0.05f, 0);
            body.transform.localScale = new Vector3(0.4f, 0.12f, 0.25f);
            Object.Destroy(body.GetComponent<Collider>());

            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Standard");

            bodyMat = new Material(shader);
            bodyMat.color = offColor;
            bodyMat.SetFloat("_Smoothness", 0.7f);
            bodyMat.SetFloat("_Metallic", 0.3f);
            body.GetComponent<Renderer>().material = bodyMat;
            bodyRenderer = body.GetComponent<Renderer>();

            // ── Lens element (small sphere at the front) ──
            var lens = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            lens.name = "Lens";
            lens.transform.SetParent(transform);
            lens.transform.localPosition = new Vector3(0, -0.12f, 0.1f);
            lens.transform.localScale = new Vector3(0.08f, 0.08f, 0.08f);
            Object.Destroy(lens.GetComponent<Collider>());

            lensMat = new Material(shader);
            lensMat.color = offColor;
            lensMat.SetFloat("_Smoothness", 0.95f);
            lens.GetComponent<Renderer>().material = lensMat;
            lensRenderer = lens.GetComponent<Renderer>();

            // ── Mounting arm (connects to ceiling) ──
            var arm = GameObject.CreatePrimitive(PrimitiveType.Cube);
            arm.name = "MountingArm";
            arm.transform.SetParent(transform);
            arm.transform.localPosition = new Vector3(0, 0.04f, 0);
            arm.transform.localScale = new Vector3(0.06f, 0.06f, 0.06f);
            Object.Destroy(arm.GetComponent<Collider>());

            var armMat = new Material(shader);
            armMat.color = new Color(0.4f, 0.4f, 0.45f);
            arm.GetComponent<Renderer>().material = armMat;

            // ── Unity Spot Light (projects downward in a cone) ──
            var lightObj = new GameObject("SpotLight");
            lightObj.transform.SetParent(transform);
            lightObj.transform.localPosition = new Vector3(0, -0.13f, 0);
            lightObj.transform.localRotation = Quaternion.Euler(90f, 0f, 0f); // point down

            spotLight = lightObj.AddComponent<UnityEngine.Light>();
            spotLight.type = LightType.Spot;
            spotLight.color = projectorBlue;
            // Map screenLux to Unity intensity (rough mapping)
            spotLight.intensity = screenLux / 100f;
            spotLight.range = coverageRadius * 1.5f;
            spotLight.spotAngle = Mathf.Atan2(coverageRadius, 3f) * Mathf.Rad2Deg * 2f;
            spotLight.innerSpotAngle = spotLight.spotAngle * 0.6f;
            spotLight.shadows = LightShadows.Soft;
            spotLight.enabled = false; // start off
        }

        /// <summary>Turn the projector on or off.</summary>
        public void SetActive(bool on)
        {
            isActive = on;
            spotLight.enabled = on;

            if (on)
            {
                bodyMat.color = projectorBlue;
                bodyMat.EnableKeyword("_EMISSION");
                bodyMat.SetColor("_EmissionColor", emissiveOn);

                lensMat.color = new Color(0.6f, 0.9f, 1f);
                lensMat.EnableKeyword("_EMISSION");
                lensMat.SetColor("_EmissionColor", lensGlow);
            }
            else
            {
                bodyMat.color = offColor;
                bodyMat.DisableKeyword("_EMISSION");
                bodyMat.SetColor("_EmissionColor", Color.black);

                lensMat.color = offColor;
                lensMat.DisableKeyword("_EMISSION");
                lensMat.SetColor("_EmissionColor", Color.black);
            }
        }
    }
}
