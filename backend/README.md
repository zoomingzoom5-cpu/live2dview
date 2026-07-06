## Monocular localization

The backend now returns a 3D position for every detected person whenever YOLO can
produce a bounding box. It follows the MonoLoco idea of using 2D pose evidence
with camera geometry: the foot point is back-projected into camera coordinates,
and depth is estimated from the apparent human height.

Useful environment variables:

- `CAMERA_FX`, `CAMERA_FY`, `CAMERA_CX`, `CAMERA_CY`: real camera intrinsics in
  pixels. If omitted, a 60 degree horizontal FOV webcam default is used.
- `HUMAN_HEIGHT_M`: assumed pedestrian height, default `1.70`.
- `BEV_HOMOGRAPHY`: optional 3x3 image-to-ground-plane homography matrix. When
  set, the backend uses this calibrated floor mapping instead of height-based
  depth.
- `YOLO_POSE_MODEL`: path to a YOLO pose model. Defaults to
  `backend/yolov8n-pose.pt`.
- `DEBUG_CAMERA_WINDOW`: show the PC-side native OpenCV debug window with the
  camera frame, pose points, and boxes. Defaults to `0` because native OpenCV
  windows often do not appear when the backend is launched as a hidden/background
  process. Set `1` only when running the backend directly in an interactive
  desktop session.
- `DEBUG_BROWSER_WINDOW`: open the PC debug browser window at
  `http://127.0.0.1:8000/debug` when the backend starts. Defaults to `1`; set
  `0` to disable it. Use this view if OpenCV's native window does not appear
  because of the way the backend was launched.

For Android cameras, landscape is the recommended default for mapped spaces
because it gives wider horizontal coverage and usually keeps multiple people in
view. Portrait can work better in a narrow corridor or when the camera is close
and full-body visibility is difficult. In either orientation, keep each person's
head and feet visible; the foot point and apparent body height drive the
geometry estimate.

Run locally:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
