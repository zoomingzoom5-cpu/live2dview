import cv2
import numpy as np
import json
import time
from ultralytics import YOLO

class BEVMapper:
    def __init__(self, src_points, dst_points):
        """
        src_points: 4 points in the image (x, y) - corners of a rectangle on the floor
        dst_points: Corresponding 4 points in BEV coordinates (X, Y)
        """
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def transform(self, x, y):
        """Transforms image coordinates (x, y) to BEV coordinates (X, Y)."""
        points = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, self.M)
        return transformed[0][0]

def main():
    # Load YOLOv8 model (Nano version for speed)
    model = YOLO("yolov8n.pt")

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Calibration Points (Example) ---
    # These should be adjusted based on the actual camera view.
    # Imagine a 2m x 2m square on the floor.
    # src: [top-left, top-right, bottom-right, bottom-left] in image pixels
    src = [[200, 300], [440, 300], [600, 450], [40, 450]]
    # dst: The same points in world coordinates (e.g., centimeters)
    dst = [[0, 0], [200, 0], [200, 200], [0, 200]]

    mapper = BEVMapper(src, dst)

    print("Starting detection... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame, verbose=False, classes=[0])  # class 0 is 'person'

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()

                    # Bottom-center of the bounding box (representative of feet location)
                    feet_x = (x1 + x2) / 2
                    feet_y = y2

                    # Map to BEV
                    bev_x, bev_y = mapper.transform(feet_x, feet_y)

                    detections.append({
                        "id": int(time.time() * 1000), # Placeholder ID
                        "image_coords": [round(feet_x, 1), round(feet_y, 1)],
                        "bev_coords": [round(bev_x, 1), round(bev_y, 1)],
                        "confidence": round(conf, 2)
                    })

            # Output as JSON to terminal
            if detections:
                print(json.dumps(detections))

            # Optional: Visualize (can be disabled for pure CLI mode)
            for d in detections:
                ix, iy = d["image_coords"]
                cv2.circle(frame, (int(ix), int(iy)), 5, (0, 255, 0), -1)
            
            cv2.imshow("Webcam YOLO BEV", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
