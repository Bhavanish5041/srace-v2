// ClassroomCamera.cs — Orbit camera for inspecting the classroom
// Mouse drag to orbit, scroll to zoom. Starts above the room looking down.

using UnityEngine;

namespace SRACE.Environment
{
    public class ClassroomCamera : MonoBehaviour
    {
        [Header("Target")]
        public Vector3 target = new Vector3(5f, 0f, 4f); // center of 10x8 room

        [Header("Orbit")]
        public float distance = 12f;
        public float minDistance = 4f;
        public float maxDistance = 25f;
        public float orbitSpeed = 4f;
        public float zoomSpeed = 2f;

        [Header("Angles")]
        public float yaw = 45f;
        public float pitch = 45f;
        public float minPitch = 10f;
        public float maxPitch = 85f;

        private bool isDragging = false;

        /// <summary>
        /// Set up the camera to frame a room of given dimensions.
        /// </summary>
        public void FrameRoom(float width, float depth, float height)
        {
            target = new Vector3(width / 2f, height / 3f, depth / 2f);
            distance = Mathf.Max(width, depth) * 1.2f;
        }

        private void LateUpdate()
        {
            HandleInput();
            UpdatePosition();
        }

        private void HandleInput()
        {
            // Right mouse button to orbit
            if (Input.GetMouseButtonDown(1))
                isDragging = true;
            if (Input.GetMouseButtonUp(1))
                isDragging = false;

            if (isDragging)
            {
                yaw += Input.GetAxis("Mouse X") * orbitSpeed;
                pitch -= Input.GetAxis("Mouse Y") * orbitSpeed;
                pitch = Mathf.Clamp(pitch, minPitch, maxPitch);
            }

            // Scroll to zoom
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(scroll) > 0.001f)
            {
                distance -= scroll * zoomSpeed * distance; // proportional zoom
                distance = Mathf.Clamp(distance, minDistance, maxDistance);
            }
        }

        private void UpdatePosition()
        {
            float pitchRad = pitch * Mathf.Deg2Rad;
            float yawRad = yaw * Mathf.Deg2Rad;

            Vector3 offset = new Vector3(
                distance * Mathf.Cos(pitchRad) * Mathf.Sin(yawRad),
                distance * Mathf.Sin(pitchRad),
                distance * Mathf.Cos(pitchRad) * Mathf.Cos(yawRad)
            );

            transform.position = target + offset;
            transform.LookAt(target);
        }
    }
}
