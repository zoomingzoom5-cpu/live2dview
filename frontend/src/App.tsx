import { useEffect, useRef, useState } from 'react';
import './App.css';

interface Detection {
  id: number;
  pos3d: number[] | null;
  trajectory: number[][] | null;
  quality?: string;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const canvasBevRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [detectedCount, setDetectedCount] = useState(0);
  const [isConnected, setIsConnected] = useState(false);

  const openDebugView = () => {
    const debugUrl = 'http://127.0.0.1:8000/debug';
    window.open(debugUrl, 'monoloco-debug', 'width=960,height=720');
  };

  useEffect(() => {
    const configuredUrl = import.meta.env.VITE_WS_URL;
    const isViteDevServer = window.location.port === '5173';
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = isViteDevServer
      ? `${window.location.hostname}:8000`
      : window.location.host;
    const ws = new WebSocket(configuredUrl || `${wsProtocol}//${wsHost}/ws/process`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to backend');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === 'success') {
        setDetections(data.detections);
        setDetectedCount(data.count ?? data.detections.length);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from backend');
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    // Start Webcam
    async function startWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            aspectRatio: { ideal: 16 / 9 },
          },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing webcam:', err);
      }
    }
    startWebcam();
  }, []);

  useEffect(() => {
    // Send frames to backend
    const interval = setInterval(() => {
      if (wsRef.current && isConnected && videoRef.current && captureCanvasRef.current) {
        const video = videoRef.current;
        const canvas = captureCanvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (ctx && video.readyState === 4 && video.videoWidth > 0) {
          // Sync canvas size with video
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          
          // Draw video to hidden canvas to get base64
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const base64Image = canvas.toDataURL('image/jpeg', 0.5);
          wsRef.current.send(base64Image);
        }
      }
    }, 100); // 10 FPS

    return () => clearInterval(interval);
  }, [isConnected]);

  // Draw Bird's Eye View
  useEffect(() => {
    if (canvasBevRef.current) {
      const canvas = canvasBevRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = 0; i < canvas.width; i += 50) {
          ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke();
        }
        for (let i = 0; i < canvas.height; i += 50) {
          ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke();
        }

        // Center point (representing the camera)
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(canvas.width / 2, canvas.height - 20, 5, 0, 2 * Math.PI); ctx.fill();

        detections.forEach(det => {
          if (det.pos3d) {
            // Map 3D (X, Y, Z) to BEV (X, Z)
            // Backend pos3d is [X, Y, Z] in meters
            // We'll scale it to pixels. X is side, Z is depth.
            const scale = 50; // 1 meter = 50 pixels
            const bevX = canvas.width / 2 + det.pos3d[0] * scale;
            const bevZ = canvas.height - 20 - det.pos3d[2] * scale;
            
            // Draw current position
            ctx.fillStyle = '#00ff00';
            ctx.beginPath(); ctx.arc(bevX, bevZ, 8, 0, 2 * Math.PI); ctx.fill();

            ctx.fillStyle = '#c9d1d9';
            ctx.font = '12px monospace';
            ctx.fillText(`#${det.id}`, bevX + 10, bevZ - 10);
            
            // Draw predicted trajectory
            if (det.trajectory && det.trajectory.length > 0) {
              ctx.strokeStyle = '#00ffff';
              ctx.setLineDash([5, 5]);
              ctx.beginPath();
              ctx.moveTo(bevX, bevZ);
              det.trajectory.forEach(pt => {
                // Assuming trajectory is also 3D or 2D XZ
                // If it's image coords, we'd need another mapping. 
                // Let's assume the wrapper provides world-space trajectory for now.
                const tx = canvas.width / 2 + pt[0] * scale;
                const tz = canvas.height - 20 - pt[1] * scale;
                ctx.lineTo(tx, tz);
              });
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        });
      }
    }
  }, [detections]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Live2DSpaceView</h1>
        <div className="header-actions">
          <button type="button" className="debug-button" onClick={openDebugView}>
            Debug View
          </button>
          <div className={`status ${isConnected ? 'online' : 'offline'}`}>
            {isConnected ? 'Backend Online' : 'Connecting to Backend...'}
          </div>
        </div>
      </header>
      
      <main className="App-main">
        <video ref={videoRef} autoPlay playsInline muted className="privacy-capture" />
        <canvas ref={captureCanvasRef} className="privacy-capture" />

        <section className="summary-grid">
          <div className="metric-panel">
            <span>Detected People</span>
            <strong>{detectedCount}</strong>
          </div>
          <div className="metric-panel">
            <span>Mapped People</span>
            <strong>{detections.filter(det => det.pos3d).length}</strong>
          </div>
        </section>

        <section className="bev-panel">
          <h3>Bird's Eye View</h3>
          <div className="canvas-wrapper">
            <canvas ref={canvasBevRef} width={700} height={620} />
          </div>
        </section>
      </main>

      <div className="detections-info">
        <h3>Live Detections</h3>
        <ul>
          {detections.map(det => (
            <li key={det.id}>
              ID: {det.id} | 
              3D Pos: {det.pos3d ? `${det.pos3d[0].toFixed(2)}, ${det.pos3d[1].toFixed(2)}, ${det.pos3d[2].toFixed(2)} m` : 'N/A'}
              {det.quality ? ` | ${det.quality}` : ''}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
