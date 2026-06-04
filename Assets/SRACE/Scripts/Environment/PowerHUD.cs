// PowerHUD.cs — Live power meter overlay for the Unity simulation.
// Uses OnGUI for zero-dependency rendering (no UI package needed).
// Shows current power, savings %, fan/light/projector counts, temperature, CO₂, and scenario name.
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
        private int projectorsOn = 0;
        private int totalFans = 10;
        private int totalLights = 10;
        private int totalProjectors = 0;
        private int totalPeople = 0;
        private float avgTemp = 30f;
        private float avgCO2 = 400f;
        private string scenarioName = "Ready";

        // ── Animation ──
        private float displayPower = 0f;
        private float displaySaved = 100f;
        private float displayTemp = 30f;
        private float displayCO2 = 400f;

        // ── Styles (built once) ──
        private GUIStyle panelStyle;
        private GUIStyle titleStyle;
        private GUIStyle powerStyle;
        private GUIStyle savedStyle;
        private GUIStyle detailStyle;
        private GUIStyle tempStyle;
        private GUIStyle co2Style;
        private GUIStyle barBgStyle;
        private GUIStyle barFillStyle;
        private bool stylesBuilt = false;

        // ── Textures ──
        private Texture2D panelTex;
        private Texture2D barBgTex;
        private Texture2D barFillTex;
        private Texture2D tempBarTex;
        private Texture2D co2BarTex;

        private void Update()
        {
            // Smooth lerp
            displayPower = Mathf.Lerp(displayPower, currentPower, Time.deltaTime * 8f);
            displaySaved = Mathf.Lerp(displaySaved, savedPct, Time.deltaTime * 8f);
            displayTemp = Mathf.Lerp(displayTemp, avgTemp, Time.deltaTime * 4f);
            displayCO2 = Mathf.Lerp(displayCO2, avgCO2, Time.deltaTime * 4f);
        }

        /// <summary>Update the HUD with new power and environment data.</summary>
        public void UpdateStats(
            float power, float maxW, float saved,
            int fans, int lights, int projectors,
            int nFans, int nLights, int nProjectors,
            int people, float temperature, float co2,
            string scenario)
        {
            currentPower = power;
            maxPower = maxW;
            savedPct = saved;
            fansOn = fans;
            lightsOn = lights;
            projectorsOn = projectors;
            totalFans = nFans;
            totalLights = nLights;
            totalProjectors = nProjectors;
            totalPeople = people;
            avgTemp = temperature;
            avgCO2 = co2;
            scenarioName = scenario;
        }

        private void BuildStyles()
        {
            // Panel background
            panelTex = MakeTex(2, 2, new Color(0.05f, 0.07f, 0.09f, 0.92f));
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

            // Temperature style
            tempStyle = new GUIStyle(GUI.skin.label);
            tempStyle.fontSize = 14;
            tempStyle.fontStyle = FontStyle.Bold;

            // CO₂ style
            co2Style = new GUIStyle(GUI.skin.label);
            co2Style.fontSize = 14;
            co2Style.fontStyle = FontStyle.Bold;

            // Bar background
            barBgTex = MakeTex(2, 2, new Color(0.15f, 0.17f, 0.2f));
            barBgStyle = new GUIStyle();
            barBgStyle.normal.background = barBgTex;

            // Bar fill
            barFillTex = MakeTex(2, 2, new Color(0.25f, 0.85f, 0.35f));
            barFillStyle = new GUIStyle();
            barFillStyle.normal.background = barFillTex;

            // Temp bar (separate texture for different color)
            tempBarTex = MakeTex(2, 2, new Color(0.94f, 0.55f, 0.13f));
            // CO₂ bar
            co2BarTex = MakeTex(2, 2, new Color(0.64f, 0.44f, 0.95f));

            stylesBuilt = true;
        }

        private void OnGUI()
        {
            if (!stylesBuilt) BuildStyles();

            float x = 16f, y = 16f;
            float w = 400f;
            float h = 210f; // taller to fit temp + CO₂
            if (totalProjectors > 0) h += 20f;

            // Panel
            GUI.Box(new Rect(x, y, w, h), GUIContent.none, panelStyle);

            // Scenario label
            GUI.Label(new Rect(x + 12, y + 8, 376, 20), scenarioName, titleStyle);

            // Power value
            GUI.Label(new Rect(x + 12, y + 30, 200, 36),
                $"{displayPower:F0}W / {maxPower:F0}W", powerStyle);

            // Saved %
            float ratio = maxPower > 0 ? displayPower / maxPower : 0f;
            savedStyle.normal.textColor = displaySaved > 50f
                ? new Color(0.25f, 0.85f, 0.35f)
                : new Color(0.97f, 0.55f, 0.13f);
            GUI.Label(new Rect(x + 240, y + 36, 148, 28),
                $"{displaySaved:F1}% saved", savedStyle);

            // Power bar
            float barX = x + 12, barY = y + 72, barW = 376, barH = 8;
            GUI.Box(new Rect(barX, barY, barW, barH), GUIContent.none, barBgStyle);

            // Bar fill color
            Color barColor;
            if (ratio < 0.4f) barColor = new Color(0.25f, 0.85f, 0.35f);
            else if (ratio < 0.7f) barColor = new Color(0.85f, 0.65f, 0.13f);
            else barColor = new Color(0.97f, 0.32f, 0.29f);

            UpdateTexColor(barFillTex, barColor);
            float fillW = Mathf.Clamp01(ratio) * barW;
            if (fillW > 1f)
                GUI.Box(new Rect(barX, barY, fillW, barH), GUIContent.none, barFillStyle);

            // Details line 1: Fans and Lights
            GUI.Label(new Rect(x + 12, y + 88, 376, 20),
                $"Fans: {fansOn}/{totalFans}   Lights: {lightsOn}/{totalLights}   People: {totalPeople}",
                detailStyle);

            float detailY = y + 108;

            // Details line 2: Projectors (only if room has them)
            if (totalProjectors > 0)
            {
                GUI.Label(new Rect(x + 12, detailY, 376, 20),
                    $"Projectors: {projectorsOn}/{totalProjectors}",
                    detailStyle);
                detailY += 20f;
            }

            // ── Temperature display ──
            float tempY = detailY + 4f;

            // Color: green if near target (25°C), yellow 28-32, red above 32
            Color tempColor;
            if (displayTemp < 28f) tempColor = new Color(0.25f, 0.85f, 0.35f);
            else if (displayTemp < 32f) tempColor = new Color(0.94f, 0.65f, 0.13f);
            else tempColor = new Color(0.97f, 0.32f, 0.29f);

            tempStyle.normal.textColor = tempColor;
            GUI.Label(new Rect(x + 12, tempY, 180, 22), $"🌡 {displayTemp:F1}°C", tempStyle);

            // Temp bar (15-45°C range)
            float tempBarY = tempY + 22f;
            GUI.Box(new Rect(barX, tempBarY, barW, 6), GUIContent.none, barBgStyle);
            float tempRatio = Mathf.Clamp01((displayTemp - 15f) / 30f);
            UpdateTexColor(tempBarTex, tempColor);
            float tempFillW = tempRatio * barW;
            if (tempFillW > 1f)
            {
                var tmpStyle = new GUIStyle { normal = { background = tempBarTex } };
                GUI.Box(new Rect(barX, tempBarY, tempFillW, 6), GUIContent.none, tmpStyle);
            }

            // ── CO₂ display ──
            float co2Y = tempBarY + 12f;

            // Color: green < 600, yellow 600-900, red > 900
            Color co2Color;
            if (displayCO2 < 600f) co2Color = new Color(0.25f, 0.85f, 0.35f);
            else if (displayCO2 < 900f) co2Color = new Color(0.64f, 0.44f, 0.95f);
            else co2Color = new Color(0.97f, 0.32f, 0.29f);

            co2Style.normal.textColor = co2Color;
            GUI.Label(new Rect(x + 12, co2Y, 180, 22), $"🌬 {displayCO2:F0} ppm CO₂", co2Style);

            // CO₂ bar (400-2000 ppm range)
            float co2BarY = co2Y + 22f;
            GUI.Box(new Rect(barX, co2BarY, barW, 6), GUIContent.none, barBgStyle);
            float co2Ratio = Mathf.Clamp01((displayCO2 - 400f) / 1600f);
            UpdateTexColor(co2BarTex, co2Color);
            float co2FillW = co2Ratio * barW;
            if (co2FillW > 1f)
            {
                var tmpStyle = new GUIStyle { normal = { background = co2BarTex } };
                GUI.Box(new Rect(barX, co2BarY, co2FillW, 6), GUIContent.none, tmpStyle);
            }
        }

        private static void UpdateTexColor(Texture2D tex, Color col)
        {
            var pixels = tex.GetPixels();
            for (int i = 0; i < pixels.Length; i++) pixels[i] = col;
            tex.SetPixels(pixels);
            tex.Apply();
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
            if (tempBarTex != null) Destroy(tempBarTex);
            if (co2BarTex != null) Destroy(co2BarTex);
        }
    }
}
