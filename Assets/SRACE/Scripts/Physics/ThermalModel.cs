// ThermalModel.cs — Thermal ODE model using RK4 integration
// Port of Python physics/thermal.py
//
// Models zone-level temperature evolution over a 5-minute forecast horizon.
// Each fan provides convective cooling proportional to its airflow impinging
// on the zone. Occupants contribute ~100W sensible heat each.
//
// ODE per zone:
//     dT_z/dt = −α·v(f,z)·(T_z − T_target) + Q_occ/(ρ·cp·V_z) + k·(T_amb − T_z)
//
// Uses 4th-order Runge-Kutta (RK4) integration since SciPy is unavailable in C#.
// Returns a thermal impact matrix [n_fans, n_zones] — ΔT per fan via marginal analysis.

using System;
using UnityEngine;
using SRACE.Core;

namespace SRACE.Physics
{
    public static class ThermalModel
    {
        // Physical constants (must match Python physics/thermal.py)
        private const float AIR_DENSITY = 1.2f;         // kg/m³
        private const float SPECIFIC_HEAT = 1005.0f;     // J/(kg·K)
        private const float OCCUPANT_HEAT_W = 100.0f;    // sensible heat per person
        private const float CONVECTION_COEFF = 0.15f;    // airflow → cooling coefficient
        private const float WALL_CONDUCTANCE = 0.01f;    // heat leak from ambient (1/s)
        private const float FORECAST_SECONDS = 300.0f;   // 5-minute forecast
        private const float RK4_DT = 1.0f;               // integration step size (seconds)

        /// <summary>
        /// Compute per-fan thermal impact via marginal analysis.
        /// Runs ODE with all fans on, then removes each fan one at a time.
        /// </summary>
        /// <param name="cfg">Room configuration.</param>
        /// <param name="airflowMatrix">(nFans, nZones) airflow contributions in m/s.</param>
        /// <param name="zoneOccupancy">(nZones) people per zone.</param>
        /// <param name="initialTemps">(nZones) starting temperatures. Null → ambient.</param>
        /// <returns>float[nFans, nZones] — ΔT each fan contributes. Positive = cooling.</returns>
        public static float[,] SimulateThermal(
            RoomConfig cfg,
            float[,] airflowMatrix,
            int[] zoneOccupancy,
            float[] initialTemps = null)
        {
            int nFans = cfg.NFans;
            int nZones = cfg.NZones;

            if (initialTemps == null)
            {
                initialTemps = new float[nZones];
                for (int i = 0; i < nZones; i++)
                    initialTemps[i] = cfg.ambientTemp;
            }

            // All fans active
            bool[] allOn = new bool[nFans];
            for (int i = 0; i < nFans; i++) allOn[i] = true;

            float[] tAll = RunODE(cfg, airflowMatrix, zoneOccupancy, initialTemps, allOn);

            // Marginal contribution: remove one fan at a time
            float[,] impact = new float[nFans, nZones];
            for (int fi = 0; fi < nFans; fi++)
            {
                bool[] mask = new bool[nFans];
                Array.Copy(allOn, mask, nFans);
                mask[fi] = false;

                float[] tWithout = RunODE(cfg, airflowMatrix, zoneOccupancy, initialTemps, mask);

                // How much warmer zones get without this fan = fan's cooling value
                for (int zi = 0; zi < nZones; zi++)
                    impact[fi, zi] = tWithout[zi] - tAll[zi]; // positive = this fan helps
            }

            return impact;
        }

        /// <summary>
        /// Solve the thermal ODE system using RK4 and return final temperatures.
        /// </summary>
        public static float[] RunODE(
            RoomConfig cfg,
            float[,] airflowMatrix,
            int[] zoneOccupancy,
            float[] initialTemps,
            bool[] activeMask)
        {
            int nFans = cfg.NFans;
            int nZones = cfg.NZones;
            float targetT = cfg.comfort.targetTempC;
            float ambientT = cfg.ambientTemp;

            // Pre-compute total airflow per zone from active fans
            float[] totalAirflow = new float[nZones];
            for (int fi = 0; fi < nFans; fi++)
            {
                if (!activeMask[fi]) continue;
                for (int zi = 0; zi < nZones; zi++)
                    totalAirflow[zi] += airflowMatrix[fi, zi];
            }

            // Pre-compute zone volumes and thermal mass
            float[] thermalMass = new float[nZones]; // J/K per zone
            for (int zi = 0; zi < nZones; zi++)
                thermalMass[zi] = AIR_DENSITY * SPECIFIC_HEAT * cfg.zones[zi].area * cfg.ceilingHeight;

            // Occupant heat per zone (W)
            float[] qOcc = new float[nZones];
            for (int zi = 0; zi < nZones; zi++)
                qOcc[zi] = zoneOccupancy[zi] * OCCUPANT_HEAT_W;

            // Copy initial state
            float[] temps = new float[nZones];
            Array.Copy(initialTemps, temps, nZones);

            // RK4 integration
            int nSteps = Mathf.CeilToInt(FORECAST_SECONDS / RK4_DT);
            float dt = RK4_DT;
            float[] k1 = new float[nZones];
            float[] k2 = new float[nZones];
            float[] k3 = new float[nZones];
            float[] k4 = new float[nZones];
            float[] temp2 = new float[nZones]; // scratch

            for (int step = 0; step < nSteps; step++)
            {
                // k1
                ComputeDerivative(temps, totalAirflow, qOcc, thermalMass, targetT, ambientT, nZones, k1);

                // k2
                for (int zi = 0; zi < nZones; zi++)
                    temp2[zi] = temps[zi] + 0.5f * dt * k1[zi];
                ComputeDerivative(temp2, totalAirflow, qOcc, thermalMass, targetT, ambientT, nZones, k2);

                // k3
                for (int zi = 0; zi < nZones; zi++)
                    temp2[zi] = temps[zi] + 0.5f * dt * k2[zi];
                ComputeDerivative(temp2, totalAirflow, qOcc, thermalMass, targetT, ambientT, nZones, k3);

                // k4
                for (int zi = 0; zi < nZones; zi++)
                    temp2[zi] = temps[zi] + dt * k3[zi];
                ComputeDerivative(temp2, totalAirflow, qOcc, thermalMass, targetT, ambientT, nZones, k4);

                // Update
                for (int zi = 0; zi < nZones; zi++)
                    temps[zi] += (dt / 6f) * (k1[zi] + 2f * k2[zi] + 2f * k3[zi] + k4[zi]);
            }

            return temps;
        }

        /// <summary>
        /// Compute dT/dt for each zone.
        /// </summary>
        private static void ComputeDerivative(
            float[] temps, float[] totalAirflow, float[] qOcc,
            float[] thermalMass, float targetT, float ambientT,
            int nZones, float[] output)
        {
            for (int zi = 0; zi < nZones; zi++)
            {
                // Convective cooling from fans
                float cooling = CONVECTION_COEFF * totalAirflow[zi] * (temps[zi] - targetT);
                // Heat from occupants
                float heating = qOcc[zi] / thermalMass[zi];
                // Heat leak from walls
                float wallLeak = WALL_CONDUCTANCE * (ambientT - temps[zi]);

                output[zi] = -cooling + heating + wallLeak;
            }
        }

        /// <summary>
        /// Quick single-step temperature update for real-time HUD display.
        /// Uses the same simplified physics as the PPO gym environment.
        /// </summary>
        /// <param name="currentTemps">Current zone temperatures (modified in place).</param>
        /// <param name="airflowPerZone">Total airflow per zone from active fans.</param>
        /// <param name="zoneOccupancy">People per zone.</param>
        /// <param name="targetTemp">Target temperature °C.</param>
        /// <param name="ambientTemp">Ambient temperature °C.</param>
        public static void QuickTick(
            float[] currentTemps,
            float[] airflowPerZone,
            int[] zoneOccupancy,
            float targetTemp,
            float ambientTemp)
        {
            const float thermalDecay = 0.02f;
            const float occupantHeat = 0.05f;
            const float wallLeakRate = 0.005f;

            for (int zi = 0; zi < currentTemps.Length; zi++)
            {
                // Cooling from fans
                currentTemps[zi] -= thermalDecay * airflowPerZone[zi] * (currentTemps[zi] - targetTemp);
                // Heating from occupants
                currentTemps[zi] += occupantHeat * zoneOccupancy[zi];
                // Heat leak from walls
                currentTemps[zi] += wallLeakRate * (ambientTemp - currentTemps[zi]);
                // Clamp
                currentTemps[zi] = Mathf.Clamp(currentTemps[zi], 15f, 45f);
            }
        }
    }
}
