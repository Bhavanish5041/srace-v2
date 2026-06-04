// CO2Model.cs — CO₂ mass-balance ODE model using RK4 integration
// Port of Python physics/co2_model.py
//
// Models zone-level CO₂ concentration over time. People exhale CO₂,
// fans increase ventilation which dilutes it.
//
// ODE per zone:
//     dC_z/dt = (n_z · G_person) / V_z  −  λ_vent(z) · (C_z − C_ambient)
//
// where λ_vent is increased by fan airflow impinging on the zone.
//
// Returns a CO₂ reduction matrix [n_fans, n_zones] — ppm reduction per fan
// via marginal analysis (same technique as ThermalModel).

using System;
using UnityEngine;
using SRACE.Core;

namespace SRACE.Physics
{
    public static class CO2Model
    {
        // Constants (must match Python physics/co2_model.py)
        private const float CO2_EXHALE_LS = 0.005f;          // L/s CO₂ per person
        private const float CO2_PPM_PER_LS_M3 = 1e6f;        // conversion: L/s/m³ → ppm/s
        private const float BASE_VENTILATION = 0.0005f;       // 1/s — natural ventilation rate
        private const float FAN_VENTILATION_COEFF = 0.002f;   // extra ventilation per m/s airflow
        private const float FORECAST_SECONDS = 300.0f;        // 5-minute window
        private const float RK4_DT = 1.0f;                    // integration step size (seconds)

        /// <summary>
        /// Compute per-fan CO₂ reduction impact via marginal analysis.
        /// </summary>
        /// <param name="cfg">Room configuration.</param>
        /// <param name="airflowMatrix">(nFans, nZones) airflow in m/s.</param>
        /// <param name="zoneOccupancy">(nZones) people per zone.</param>
        /// <param name="initialCO2">(nZones) starting CO₂ in ppm. Null → ambient.</param>
        /// <returns>float[nFans, nZones] — CO₂ reduction in ppm that each fan provides. Positive = helps.</returns>
        public static float[,] SimulateCO2(
            RoomConfig cfg,
            float[,] airflowMatrix,
            int[] zoneOccupancy,
            float[] initialCO2 = null)
        {
            int nFans = cfg.NFans;
            int nZones = cfg.NZones;

            if (initialCO2 == null)
            {
                initialCO2 = new float[nZones];
                for (int i = 0; i < nZones; i++)
                    initialCO2[i] = cfg.ambientCO2;
            }

            // All fans on
            bool[] allOn = new bool[nFans];
            for (int i = 0; i < nFans; i++) allOn[i] = true;

            float[] cAll = RunODE(cfg, airflowMatrix, zoneOccupancy, initialCO2, allOn);

            // Marginal: remove each fan
            float[,] impact = new float[nFans, nZones];
            for (int fi = 0; fi < nFans; fi++)
            {
                bool[] mask = new bool[nFans];
                Array.Copy(allOn, mask, nFans);
                mask[fi] = false;

                float[] cWithout = RunODE(cfg, airflowMatrix, zoneOccupancy, initialCO2, mask);

                // How much higher CO₂ gets without this fan = fan's ventilation value
                for (int zi = 0; zi < nZones; zi++)
                    impact[fi, zi] = cWithout[zi] - cAll[zi]; // positive = this fan helps
            }

            return impact;
        }

        /// <summary>
        /// Solve the CO₂ ODE system using RK4 and return final concentrations.
        /// </summary>
        public static float[] RunODE(
            RoomConfig cfg,
            float[,] airflowMatrix,
            int[] zoneOccupancy,
            float[] initialCO2,
            bool[] activeMask)
        {
            int nFans = cfg.NFans;
            int nZones = cfg.NZones;
            float ambient = cfg.ambientCO2;

            // Total airflow per zone from active fans
            float[] totalAirflow = new float[nZones];
            for (int fi = 0; fi < nFans; fi++)
            {
                if (!activeMask[fi]) continue;
                for (int zi = 0; zi < nZones; zi++)
                    totalAirflow[zi] += airflowMatrix[fi, zi];
            }

            // Zone volumes
            float[] volumes = new float[nZones];
            for (int zi = 0; zi < nZones; zi++)
                volumes[zi] = cfg.zones[zi].area * cfg.ceilingHeight;

            // CO₂ generation rate per zone (ppm/s)
            float[] generation = new float[nZones];
            for (int zi = 0; zi < nZones; zi++)
            {
                if (volumes[zi] > 0)
                    generation[zi] = (zoneOccupancy[zi] * CO2_EXHALE_LS / volumes[zi]) * CO2_PPM_PER_LS_M3;
            }

            // Ventilation rate per zone (1/s)
            float[] ventilation = new float[nZones];
            for (int zi = 0; zi < nZones; zi++)
                ventilation[zi] = BASE_VENTILATION + FAN_VENTILATION_COEFF * totalAirflow[zi];

            // Copy initial state
            float[] co2 = new float[nZones];
            Array.Copy(initialCO2, co2, nZones);

            // RK4 integration
            int nSteps = Mathf.CeilToInt(FORECAST_SECONDS / RK4_DT);
            float dt = RK4_DT;
            float[] k1 = new float[nZones];
            float[] k2 = new float[nZones];
            float[] k3 = new float[nZones];
            float[] k4 = new float[nZones];
            float[] co2Temp = new float[nZones]; // scratch

            for (int step = 0; step < nSteps; step++)
            {
                // k1
                ComputeDerivative(co2, generation, ventilation, ambient, nZones, k1);

                // k2
                for (int zi = 0; zi < nZones; zi++)
                    co2Temp[zi] = co2[zi] + 0.5f * dt * k1[zi];
                ComputeDerivative(co2Temp, generation, ventilation, ambient, nZones, k2);

                // k3
                for (int zi = 0; zi < nZones; zi++)
                    co2Temp[zi] = co2[zi] + 0.5f * dt * k2[zi];
                ComputeDerivative(co2Temp, generation, ventilation, ambient, nZones, k3);

                // k4
                for (int zi = 0; zi < nZones; zi++)
                    co2Temp[zi] = co2[zi] + dt * k3[zi];
                ComputeDerivative(co2Temp, generation, ventilation, ambient, nZones, k4);

                // Update
                for (int zi = 0; zi < nZones; zi++)
                    co2[zi] += (dt / 6f) * (k1[zi] + 2f * k2[zi] + 2f * k3[zi] + k4[zi]);
            }

            return co2;
        }

        /// <summary>
        /// Compute dC/dt for each zone.
        /// </summary>
        private static void ComputeDerivative(
            float[] co2, float[] generation, float[] ventilation,
            float ambient, int nZones, float[] output)
        {
            for (int zi = 0; zi < nZones; zi++)
                output[zi] = generation[zi] - ventilation[zi] * (co2[zi] - ambient);
        }

        /// <summary>
        /// Quick single-step CO₂ update for real-time HUD display.
        /// Uses the same simplified physics as the PPO gym environment.
        /// </summary>
        /// <param name="currentCO2">Current zone CO₂ levels (modified in place).</param>
        /// <param name="airflowPerZone">Total airflow per zone from active fans.</param>
        /// <param name="zoneOccupancy">People per zone.</param>
        /// <param name="ambientCO2">Ambient CO₂ ppm.</param>
        public static void QuickTick(
            float[] currentCO2,
            float[] airflowPerZone,
            int[] zoneOccupancy,
            float ambientCO2)
        {
            const float co2Generation = 5.0f;       // ppm per person per step
            const float co2DecayFan = 0.01f;         // per-step decay per unit airflow
            const float co2NaturalDecay = 0.005f;    // natural ventilation per step

            for (int zi = 0; zi < currentCO2.Length; zi++)
            {
                // Generation from people
                currentCO2[zi] += co2Generation * zoneOccupancy[zi];
                // Removal from fan ventilation
                currentCO2[zi] -= co2DecayFan * airflowPerZone[zi] * (currentCO2[zi] - ambientCO2);
                // Natural ventilation
                currentCO2[zi] -= co2NaturalDecay * (currentCO2[zi] - ambientCO2);
                // Clamp
                currentCO2[zi] = Mathf.Clamp(currentCO2[zi], ambientCO2, 2000f);
            }
        }
    }
}
