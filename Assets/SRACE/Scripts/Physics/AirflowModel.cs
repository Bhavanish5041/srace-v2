// AirflowModel.cs — Gaussian decay fan airflow model
// Port of Python physics/airflow.py
//
// Each fan produces a peak airflow at its center that decays as a Gaussian
// with distance. σ is set so that 95% of airflow falls within the
// configured airflow_radius.
//
// Formula:
//     v(f, z) = v_peak · exp(−d² / (2σ²))
//     σ = airflow_radius / 2

using System;
using UnityEngine;
using SRACE.Core;

namespace SRACE.Physics
{
    public static class AirflowModel
    {
        /// <summary>
        /// Compute airflow contribution of each fan to each zone.
        /// Returns float[nFans, nZones] — airflow in m/s.
        /// </summary>
        public static float[,] ComputeAirflowMatrix(RoomConfig cfg)
        {
            int nFans = cfg.NFans;
            int nZones = cfg.NZones;
            var matrix = new float[nFans, nZones];

            for (int fi = 0; fi < nFans; fi++)
            {
                var fan = cfg.fans[fi];
                float sigma = fan.airflowRadius / 2f; // 95% within radius
                float twoSigmaSq = 2f * sigma * sigma;

                for (int zi = 0; zi < nZones; zi++)
                {
                    var zone = cfg.zones[zi];
                    float dx = fan.x - zone.cx;
                    float dy = fan.y - zone.cy;
                    float distSq = dx * dx + dy * dy;

                    matrix[fi, zi] = fan.airflowPeakMs * Mathf.Exp(-distSq / twoSigmaSq);
                }
            }

            return matrix;
        }

        /// <summary>
        /// Sum airflow across all active fans for each zone.
        /// </summary>
        /// <param name="airflowMatrix">(nFans, nZones) from ComputeAirflowMatrix.</param>
        /// <param name="activeFans">Boolean array of length nFans.</param>
        /// <returns>float[nZones] — total airflow per zone in m/s.</returns>
        public static float[] TotalAirflowPerZone(float[,] airflowMatrix, bool[] activeFans)
        {
            int nFans = airflowMatrix.GetLength(0);
            int nZones = airflowMatrix.GetLength(1);
            var result = new float[nZones];

            for (int fi = 0; fi < nFans; fi++)
            {
                if (!activeFans[fi]) continue;
                for (int zi = 0; zi < nZones; zi++)
                    result[zi] += airflowMatrix[fi, zi];
            }

            return result;
        }
    }
}
