// LightingModel.cs — Cosine-law illuminance model for ceiling fixtures
// Port of Python physics/lighting.py
//
// Each ceiling light emits downward in a Lambertian (cosine) pattern.
// Illuminance at a point on the floor:
//
//     E(l, z) = (Φ · cos³θ) / (2π · h²)
//
// where:
//     Φ = luminous flux in lumens
//     h = mounting height
//     θ = angle from nadir: tan(θ) = d/h
//     cos θ = h / √(d² + h²)

using System;
using UnityEngine;
using SRACE.Core;

namespace SRACE.Physics
{
    public static class LightingModel
    {
        /// <summary>
        /// Compute illuminance contribution of each light to each zone.
        /// Uses cosine-law for downward-emitting ceiling fixtures.
        /// Returns float[nLights, nZones] — lux values.
        /// </summary>
        public static float[,] ComputeLuxMatrix(RoomConfig cfg)
        {
            int nLights = cfg.NLights;
            int nZones = cfg.NZones;
            var matrix = new float[nLights, nZones];
            float twoPi = 2f * Mathf.PI;

            for (int li = 0; li < nLights; li++)
            {
                var light = cfg.lights[li];
                float h = light.heightAboveFloor;
                float hSq = h * h;

                for (int zi = 0; zi < nZones; zi++)
                {
                    var zone = cfg.zones[zi];
                    float dx = light.x - zone.cx;
                    float dy = light.y - zone.cy;
                    float horizDistSq = dx * dx + dy * dy;
                    float totalDistSq = horizDistSq + hSq;

                    // cos θ = h / r  where r = √(d² + h²)
                    float cosTheta = h / Mathf.Sqrt(totalDistSq);

                    // Cosine-law: E = (Φ · cos³θ) / (2π · h²)
                    float cosTheta3 = cosTheta * cosTheta * cosTheta;
                    matrix[li, zi] = (light.lumens * cosTheta3) / (twoPi * hSq);
                }
            }

            return matrix;
        }

        /// <summary>
        /// Total illuminance per zone from active lights + ambient.
        /// </summary>
        public static float[] TotalLuxPerZone(float[,] luxMatrix, bool[] activeLights,
                                               float ambientLux = 0f)
        {
            int nLights = luxMatrix.GetLength(0);
            int nZones = luxMatrix.GetLength(1);
            var result = new float[nZones];

            for (int li = 0; li < nLights; li++)
            {
                if (!activeLights[li]) continue;
                for (int zi = 0; zi < nZones; zi++)
                    result[zi] += luxMatrix[li, zi];
            }

            for (int zi = 0; zi < nZones; zi++)
                result[zi] += ambientLux;

            return result;
        }
    }
}
