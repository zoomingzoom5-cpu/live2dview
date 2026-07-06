# MonoTransmotion Web App - Setup Instructions

This application integrates the MonoTransmotion framework for real-time 3D human localization and trajectory prediction.

## Prerequisites
- Python 3.9+
- Node.js & npm
- Webcam (or a video stream source)

## Installation

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 2. Checkpoints (CRITICAL)
You must download the pre-trained checkpoints from the [MonoTransmotion Releases](https://github.com/vita-epfl/MonoTransmotion/releases) and place them in the following paths:
- `models/MonoTransmotion/checkpoints/localization.pth`
- `models/MonoTransmotion/checkpoints/traj_pred.pth`

Then update the `configs/*.yaml` files in `models/MonoTransmotion/code/configs/` to point to these checkpoints:
```yaml
MODEL:
    checkpoint: "C:/absolute/path/to/localization.pth"
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the Backend
```bash
cd backend
python main.py
```
The backend will start at `http://localhost:8000`. It will download `yolov8n-pose.pt` on its first run.

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

## Architecture Note
- **Backend:** FastAPI handles WebSocket connections, runs YOLOv8-pose for 2D skeleton extraction, and pipes the results through MonoTransmotion's Localization and Trajectory models.
- **Frontend:** React captures the webcam feed, sends frames to the backend, and visualizes the results on a 2D Bird's Eye View (BEV) canvas.
