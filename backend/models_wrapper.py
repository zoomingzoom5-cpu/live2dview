import math
import os
from ast import literal_eval
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TransmotionWrapper:
    """YOLO-pose based monocular pedestrian localization.

    The bundled MonoTransmotion code needs trained checkpoints to produce metric
    locations. This wrapper keeps the same API but uses the MonoLoco-style
    geometry fallback: infer depth from the projected human height, then
    back-project the foot point with camera intrinsics.
    """

    def __init__(self, device=None):
        self.device = device or os.getenv("MONOLOCO_DEVICE", None)
        model_path = os.getenv(
            "YOLO_POSE_MODEL",
            os.path.join(CURRENT_DIR, "yolov8n-pose.pt"),
        )
        if not os.path.exists(model_path):
            model_path = "yolov8n-pose.pt"

        self.yolo = YOLO(model_path)
        if self.device:
            self.yolo.to(self.device)

        self.human_height_m = float(os.getenv("HUMAN_HEIGHT_M", "1.70"))
        self.min_keypoint_conf = float(os.getenv("MIN_KEYPOINT_CONF", "0.30"))
        self.min_box_conf = float(os.getenv("MIN_BOX_CONF", "0.25"))
        self.max_depth_m = float(os.getenv("MAX_DEPTH_M", "40.0"))
        self.history_len = int(os.getenv("TRACK_HISTORY_LEN", "12"))
        self.debug_window = os.getenv("DEBUG_CAMERA_WINDOW", "0") == "1"
        self.debug_window_name = os.getenv("DEBUG_CAMERA_WINDOW_NAME", "YOLO Pose Debug")
        self.latest_debug_jpeg = None

        self.fx = _optional_float("CAMERA_FX")
        self.fy = _optional_float("CAMERA_FY")
        self.cx = _optional_float("CAMERA_CX")
        self.cy = _optional_float("CAMERA_CY")
        self.homography = _load_homography(os.getenv("BEV_HOMOGRAPHY"))
        self.position_history = defaultdict(lambda: deque(maxlen=self.history_len))

        print("Using YOLO-pose monocular geometry localization")
        print(
            "Camera intrinsics:",
            {
                "fx": self.fx or "auto",
                "fy": self.fy or "auto",
                "cx": self.cx or "auto",
                "cy": self.cy or "auto",
            },
        )

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        fx, fy, cx, cy = self._camera_params(width, height)

        results = self.yolo.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0],
            conf=self.min_box_conf,
        )
        if not results:
            self._show_debug_window(frame, [])
            return []

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            self._show_debug_window(frame, [])
            return []

        boxes = res.boxes
        keypoints = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        ids = _track_ids(boxes)

        detections = []
        for index, box in enumerate(boxes):
            bbox = box.xyxy[0].cpu().numpy().astype(float)
            if keypoints is None or index >= len(keypoints):
                kp = np.zeros((17, 3), dtype=float)
            else:
                kp = keypoints[index].astype(float)

            person_id = int(ids[index]) if index < len(ids) else index + 1
            pos3d, image_anchor, quality = self._estimate_position(kp, bbox, fx, fy, cx, cy)

            if pos3d is not None:
                pos3d = self._smooth_position(person_id, pos3d)
                trajectory = self._predict_trajectory(person_id)
            else:
                trajectory = None

            detections.append(
                {
                    "id": person_id,
                    "bbox": bbox.tolist(),
                    "pose2d": kp.tolist(),
                    "pos3d": pos3d.tolist() if pos3d is not None else None,
                    "trajectory": trajectory,
                    "image_anchor": image_anchor,
                    "quality": quality,
                }
            )

        self._show_debug_window(frame, detections)
        return detections

    def get_debug_jpeg(self):
        return self.latest_debug_jpeg

    def _camera_params(self, width, height):
        # A 60-degree horizontal FOV is a practical webcam default. Real CAMERA_FX/FY
        # values improve metric accuracy without changing the API.
        auto_fx = width / (2.0 * math.tan(math.radians(60.0) / 2.0))
        auto_fy = auto_fx
        return (
            self.fx or auto_fx,
            self.fy or auto_fy,
            self.cx if self.cx is not None else width / 2.0,
            self.cy if self.cy is not None else height / 2.0,
        )

    def _estimate_position(self, kp, bbox, fx, fy, cx, cy):
        foot = self._foot_point(kp, bbox)
        top_y = self._body_top(kp, bbox)
        pixel_height = max(8.0, foot[1] - top_y)

        if self.homography is not None:
            bev_x, bev_z = _transform_homography(self.homography, foot)
            pos = np.array([bev_x, 0.0, bev_z], dtype=float)
            quality = "homography"
            return pos, [float(foot[0]), float(foot[1])], quality

        depth = fy * self.human_height_m / pixel_height
        if not np.isfinite(depth) or depth <= 0:
            return None, [float(foot[0]), float(foot[1])], "invalid-depth"

        depth = min(depth, self.max_depth_m)
        lateral = (foot[0] - cx) * depth / fx
        vertical = 0.0
        quality = "pose-height" if self._has_reliable_lower_body(kp) else "box-height"
        return np.array([lateral, vertical, depth], dtype=float), [float(foot[0]), float(foot[1])], quality

    def _foot_point(self, kp, bbox):
        ankle_points = _visible_points(kp, (15, 16), self.min_keypoint_conf)
        if ankle_points:
            return np.mean(ankle_points, axis=0)

        knee_points = _visible_points(kp, (13, 14), self.min_keypoint_conf)
        if knee_points:
            knees = np.mean(knee_points, axis=0)
            return np.array([knees[0], bbox[3]], dtype=float)

        x1, _, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, y2], dtype=float)

    def _body_top(self, kp, bbox):
        head_points = _visible_points(kp, (0, 1, 2, 3, 4), self.min_keypoint_conf)
        if head_points:
            return float(min(point[1] for point in head_points))
        return float(bbox[1])

    def _has_reliable_lower_body(self, kp):
        return len(_visible_points(kp, (11, 12, 13, 14, 15, 16), self.min_keypoint_conf)) >= 2

    def _smooth_position(self, person_id, pos):
        history = self.position_history[person_id]
        history.append(pos)
        weights = np.linspace(0.35, 1.0, len(history), dtype=float)
        stacked = np.vstack(history)
        return np.average(stacked, axis=0, weights=weights)

    def _predict_trajectory(self, person_id):
        history = self.position_history[person_id]
        if len(history) < 3:
            return []

        recent = np.vstack(list(history)[-4:])
        velocity = np.diff(recent[:, [0, 2]], axis=0).mean(axis=0)
        current = recent[-1, [0, 2]]
        return [(current + velocity * step).tolist() for step in range(1, 7)]

    def _show_debug_window(self, frame, detections):
        debug = self._update_debug_frame(frame, detections)
        if not self.debug_window:
            return

        try:
            cv2.imshow(self.debug_window_name, debug)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(self.debug_window_name)
                self.debug_window = False
        except cv2.error as exc:
            print(f"OpenCV debug window disabled: {exc}")
            self.debug_window = False

    def _update_debug_frame(self, frame, detections):
        debug = frame.copy()
        for det in detections:
            bbox = det["bbox"]
            pose = det["pose2d"]
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug,
                f"ID {det['id']} {det['quality']}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            for x, y, conf in pose:
                if conf >= self.min_keypoint_conf:
                    cv2.circle(debug, (int(x), int(y)), 3, (0, 0, 255), -1)

            if det["image_anchor"]:
                ax, ay = [int(value) for value in det["image_anchor"]]
                cv2.circle(debug, (ax, ay), 6, (255, 128, 0), -1)

            if det["pos3d"]:
                x, _, z = det["pos3d"]
                cv2.putText(
                    debug,
                    f"X:{x:.2f}m Z:{z:.2f}m",
                    (x1, min(debug.shape[0] - 8, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 128, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            debug,
            f"Detected: {len(detections)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", debug, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self.latest_debug_jpeg = encoded.tobytes()
        return debug


def _optional_float(name):
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        print(f"Ignoring invalid {name}={value!r}")
        return None


def _load_homography(raw):
    if not raw:
        return None
    try:
        matrix = np.array(literal_eval(raw), dtype=np.float32)
    except Exception as exc:
        print(f"Ignoring invalid BEV_HOMOGRAPHY: {exc}")
        return None
    if matrix.shape != (3, 3):
        print("Ignoring BEV_HOMOGRAPHY because it is not a 3x3 matrix")
        return None
    return matrix


def _transform_homography(matrix, point):
    points = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(points, matrix)
    return transformed[0][0].astype(float)


def _visible_points(kp, indices, min_conf):
    points = []
    for idx in indices:
        if idx < len(kp) and kp[idx][2] >= min_conf:
            points.append(np.array([kp[idx][0], kp[idx][1]], dtype=float))
    return points


def _track_ids(boxes):
    if boxes.id is None:
        return np.arange(len(boxes), dtype=int) + 1
    return boxes.id.cpu().numpy().astype(int)
