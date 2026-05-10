// PowerHUD.cs — Live power meter overlay for the Unity simulation.
// Uses OnGUI for zero-dependency rendering (no UI package needed).
// Shows current power, savings %, fan/light counts, and scenario name.
//
// Attach to the same GameObject as SRACEManager, or any GameObject.
// Updates automatically when SRACEManager runs the optimizer or API state.

using UnityEngine;

namespace SRACE.Environment
{
    public class PowerHUD : MonoBehaviour
    {
        // ── State (updated externally) ──
        private float currentPower = 0f;
        private float maxPower = 1150f;
        private float savedPct = 100f;
        private int fansOn = 0;
        private int lightsOn = 0;
        private int totalFans = 10;
        private int totalLights = 10;
        private int totalPeople = 0;
        private string scenarioName = "Ready";

        // ── Animation ──
        private float displayPower = 0f;
        private float displaySaved = 100f;

        // ── Styles (built once) ──
        private GUIStyle panelStyle;
        private GUIStyle titleStyle;
        private GUIStyle powerStyle;
        private GUIStyle savedStyle;
        private GUIStyle detailStyle;
        private GUIStyle barBgStyle;
        private GUIStyle barFillStyle;
        private bool stylesBuilt = false;

        // ── Textures ──
        private Texture2D panelTex;
        private Texture2D barBgTex;
        private Texture2D barFillTex;

        private void Update()
        {
            // Smooth lerp
            displayPower = Mathf.Lerp(displayPower, currentPower, Time.deltaTime * 8f);
            displaySaved = Mathf.Lerp(displaySaved, savedPct, Time.deltaTime * 8f);
        }

        /// <summary>Update the HUD with new power data.</summary>
        public void UpdateStats(
            float power, float maxW, float saved,
            int fans, int lights, int nFans, int nLights,
            int people, string scenario)
        {
            currentPower = power;
            maxPower = maxW;
            savedPct = saved;
            fansOn = fans;
            lightsOn = lights;
            totalFans = nFans;
            totalLights = nLights;
            totalPeople = people;
            scenarioName = scenario;
        }

        private void BuildStyles()
        {
            // Panel background
            panelTex = MakeTex(2, 2, new Color(0.05f, 0.07f, 0.09f, 0.9f));
            panelStyle = new GUIStyle();
            panelStyle.normal.background = panelTex;

            // Title
            titleStyle = new GUIStyle(GUI.skin.label);
            titleStyle.fontSize = 13;
            titleStyle.fontStyle = FontStyle.Bold;
            titleStyle.normal.textColor = new Color(0.35f, 0.65f, 1f);

            // Power value
            powerStyle = new GUIStyle(GUI.skin.label);
            powerStyle.fontSize = 26;
            powerStyle.fontStyle = FontStyle.Bold;
            powerStyle.normal.textColor = Color.white;

            // Saved %
            savedStyle = new GUIStyle(GUI.skin.label);
            savedStyle.fontSize = 16;
            savedStyle.fontStyle = FontStyle.Bold;
            savedStyle.alignment = TextAnchor.MiddleRight;

            // Detail line
            detailStyle = new GUIStyle(GUI.skin.label);
            detailStyle.fontSize = 12;
            detailStyle.normal.textColor = new Color(0.6f, 0.63f, 0.68f);

            // Bar background
            barBgTex = MakeTex(2, 2, new Color(0.15f, 0.17f, 0.2f));
            barBgStyle = new GUIStyle();
            barBgStyle.normal.background = barBgTex;

            // Bar fill
            barFillTex = MakeTex(2, 2, new Color(0.25f, 0.85f, 0.35f));
            barFillStyle = new GUIStyle();
            barFillStyle.normal.background = barFillTex;

            stylesBuilt = true;
        }

        private void OnGUI()
        {
            if (!stylesBuilt) BuildStyles();

            float x = 16f, y = 16f;
            float w = 340f, h = 130f;

            // Panel
            GUI.Box(new Rect(x, y, w, h), GUIContent.none, panelStyle);

            // Scenario label
            GUI.Label(new Rect(x + 12, y + 8, 316, 20), scenarioName, titleStyle);

            // Power value
            GUI.Label(new Rect(x + 12, y + 30, 200, 36),
                $"{displayPower:F0}W / {maxPower:F0}W", powerStyle);

            // Saved %
            float ratio = maxPower > 0 ? displayPower / maxPower : 0f;
            savedStyle.normal.textColor = displaySaved > 50f
                ? new Color(0.25f, 0.85f, 0.35f)
                : new Color(0.97f, 0.55f, 0.13f);
            GUI.Label(new Rect(x + 200, y + 36, 128, 28),
                $"{displaySaved:F1}% saved", savedStyle);

            // Power bar
            float barX = x + 12, barY = y + 72, barW = 316, barH = 10;
            GUI.Box(new Rect(barX, barY, barW, barH), GUIContent.none, barBgStyle);

            // Bar fill color
            Color barColor;
            if (ratio < 0.4f) barColor = new Color(0.25f, 0.85f, 0.35f);
            else if (ratio < 0.7f) barColor = new Color(0.85f, 0.65f, 0.13f);
            else barColor = new Color(0.97f, 0.32f, 0.29f);

            // Update fill texture color
            var pixels = barFillTex.GetPixels();
            for (int i = 0; i < pixels.Length; i++) pixels[i] = barColor;
            barFillTex.SetPixels(pixels);
            barFillTex.Apply();

            float fillW = Mathf.Clamp01(ratio) * barW;
            if (fillW > 1f)
                GUI.Box(new Rect(barX, barY, fillW, barH), GUIContent.none, barFillStyle);

            // Details
            GUI.Label(new Rect(x + 12, y + 90, 316, 28),
                $"Fans: {fansOn}/{totalFans}   Lights: {lightsOn}/{totalLights}   People: {totalPeople}",
                detailStyle);
        }

        private static Texture2D MakeTex(int w, int h, Color col)
        {
            var tex = new Texture2D(w, h);
            var pix = new Color[w * h];
            for (int i = 0; i < pix.Length; i++) pix[i] = col;
            tex.SetPixels(pix);
            tex.Apply();
            return tex;
        }

        private void OnDestroy()
        {
            if (panelTex != null) Destroy(panelTex);
            if (barBgTex != null) Destroy(barBgTex);
            if (barFillTex != null) Destroy(barFillTex);
        }
    }
}
