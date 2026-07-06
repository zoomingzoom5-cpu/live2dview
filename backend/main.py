import cv2
import base64
import asyncio
import os
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models_wrapper import TransmotionWrapper

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Initialize the model wrapper
# Note: Loading models can be slow, so we do it once at startup
print("Loading models...")
try:
    model_wrapper = TransmotionWrapper()
except Exception as e:
    print(f"Error loading models: {e}")
    model_wrapper = None


def open_debug_window_once():
    if os.getenv("DEBUG_BROWSER_WINDOW", "1") == "0":
        return

    def opener():
        time.sleep(2)
        webbrowser.open_new("http://127.0.0.1:8000/debug")

    threading.Thread(target=opener, daemon=True).start()


def require_local_debug(request: Request):
    host = request.headers.get("host", "").split(":", 1)[0].lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise HTTPException(status_code=404, detail="Not Found")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    open_debug_window_once()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return {
        "message": "Live2DSpaceView Backend is running",
        "debug": "/debug",
        "version": "debug-browser-view",
    }


@app.get("/routes")
async def routes():
    return sorted(route.path for route in app.routes)


@app.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    require_local_debug(request)
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>YOLO Pose Debug</title>
    <style>
      html, body {
        margin: 0;
        width: 100%;
        height: 100%;
        background: #0d1117;
        color: #c9d1d9;
        font-family: Segoe UI, sans-serif;
      }
      body {
        display: grid;
        grid-template-rows: auto 1fr;
      }
      header {
        padding: 10px 14px;
        background: #161b22;
        border-bottom: 1px solid #30363d;
      }
      img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        background: #000;
      }
      main {
        position: relative;
        min-height: 0;
      }
      .hint {
        position: absolute;
        left: 16px;
        bottom: 16px;
        color: #8b949e;
        background: rgba(13, 17, 23, 0.8);
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <header>PC Debug Camera View</header>
    <main>
      <img src="/debug/stream" alt="debug camera stream" />
      <div class="hint">Waiting for frames from the main page...</div>
    </main>
  </body>
</html>
"""


@app.get("/debug/stream")
async def debug_stream(request: Request):
    require_local_debug(request)

    async def frames():
        boundary = b"--frame\r\n"
        while True:
            jpeg = model_wrapper.get_debug_jpeg() if model_wrapper else None
            if jpeg:
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.08)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/process")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    try:
        while True:
            # Receive image from client (as base64 string or binary)
            data = await websocket.receive_text()
            
            # Decode base64 image
            try:
                if "," not in data:
                    continue
                
                parts = data.split(",", 1)
                if len(parts) < 2:
                    continue
                
                encoded = parts[1]
                if not encoded:
                    continue
                    
                image_data = base64.b64decode(encoded)
                if not image_data:
                    continue
                    
                nparr = np.frombuffer(image_data, np.uint8)
                if nparr.size == 0:
                    continue
                    
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process frame
                if model_wrapper:
                    detections = model_wrapper.process_frame(frame)
                else:
                    detections = []

                public_detections = [
                    {
                        "id": detection["id"],
                        "pos3d": detection["pos3d"],
                        "trajectory": detection["trajectory"],
                        "quality": detection["quality"],
                    }
                    for detection in detections
                ]
                
                # Send results back
                await websocket.send_json({
                    "status": "success",
                    "count": len(public_detections),
                    "detections": public_detections
                })
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send_json({"status": "error", "message": str(e)})
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )


@app.get("/{full_path:path}")
async def frontend_app(full_path: str):
    requested_file = FRONTEND_DIST / full_path
    if full_path and requested_file.is_file():
        return FileResponse(requested_file)

    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return HTMLResponse(
        "Frontend is not built. Run `npm run build` in the frontend directory.",
        status_code=503,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
