import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Maximize2, Settings2, WifiOff, RefreshCw, Activity } from 'lucide-react';

interface EdgeCamera {
  id: string;
  name: string;
  ip: string;
}

type TileStatus = 'loading' | 'live' | 'offline';

interface LiveTileProps {
  cam: EdgeCamera;
  index: number;
}

const LiveTile: React.FC<LiveTileProps> = ({ cam, index }) => {
  const [status, setStatus] = useState<TileStatus>('loading');
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const label = `CAM-${String(index + 1).padStart(2, '0')}`;
  // Direct HTTP to port 4000 — img tags have no CORS restriction, proxy not needed
  const src = `http://localhost:4000/api/stream/${encodeURIComponent(cam.id)}`;

  useEffect(() => {
    setStatus('loading');
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setStatus('offline'), 8000);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [src]);

  return (
    <div className="relative bg-gray-900 rounded overflow-hidden group border border-gray-800" style={{ minHeight: '120px' }}>
      {/* MJPEG img */}
      <img
        src={src}
        alt={`Feed ${cam.name}`}
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 ${status === 'live' ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => { if (timerRef.current) clearTimeout(timerRef.current); setStatus('live'); }}
        onError={() => { if (timerRef.current) clearTimeout(timerRef.current); setStatus('offline'); }}
      />

      {/* Placeholder */}
      {status !== 'live' && (
        <div className="absolute inset-0 bg-navy-dark flex flex-col items-center justify-center space-y-1">
          {status === 'loading'
            ? <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
            : <WifiOff className="w-4 h-4 text-red-500" />
          }
          <span className="text-[8px] font-bold text-gray-500 uppercase tracking-wider">
            {status === 'loading' ? 'Connecting…' : 'Offline'}
          </span>
        </div>
      )}

      <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/60 pointer-events-none" />

      <div className="absolute top-2 left-2 pointer-events-none">
        <span className="text-white text-xs font-bold font-mono tracking-wider">{label}</span>
        <p className="text-gray-300 text-[10px]">{cam.name}</p>
      </div>

      <div className="absolute top-2 right-2 pointer-events-none">
        {status === 'live' && (
          <span className="bg-status-success text-white text-[9px] px-1 rounded font-bold tracking-wider">LIVE</span>
        )}
      </div>

      <div className="absolute bottom-2 left-2 pointer-events-none">
        <div className="flex items-center bg-black/50 rounded px-1.5 py-0.5">
          <div className={`h-1.5 w-1.5 rounded-full mr-1 ${status === 'live' ? 'bg-blue-400' : 'bg-gray-500'}`} />
          <span className="text-white text-[9px] font-bold flex items-center">
            {status === 'live' && <Activity className="w-2 h-2 mr-0.5" />}
            AI {status === 'live' ? 'ACTIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>

      <div className="absolute inset-0 bg-navy-dark/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[1px]">
        <button className="bg-white/10 hover:bg-white/20 text-white border border-white/30 rounded px-3 py-1.5 text-xs font-medium">
          View Detail
        </button>
      </div>
    </div>
  );
};

export const LiveCameraGrid: React.FC = () => {
  const [cameras, setCameras] = useState<EdgeCamera[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchCameras = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/edge-api/cameras', {
        headers: { 'Authorization': 'Basic YWRtaW46YWRtaW4=' }  // admin:admin
      });
      if (res.ok) {
        const data: EdgeCamera[] = await res.json();
        setCameras(data.slice(0, 4));
      }
    } catch {
      // best-effort
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchCameras(); }, [fetchCameras]);

  return (
    <div className="card h-full flex flex-col">
      <div className="card-header bg-navy/5 flex justify-between items-center">
        <div className="flex items-center text-navy-dark">
          <span className="relative flex h-2 w-2 mr-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-critical opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-status-critical" />
          </span>
          <span>Live Camera Wall</span>
        </div>
        <div className="flex space-x-2">
          <button onClick={fetchCameras} className="text-text-secondary hover:text-navy transition-colors" title="Refresh">
            <Settings2 className="w-4 h-4" />
          </button>
          <button className="text-text-secondary hover:text-navy transition-colors">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="card-body p-2 flex-1">
        {loading && (
          <div className="h-full flex items-center justify-center">
            <RefreshCw className="w-5 h-5 text-navy animate-spin" />
          </div>
        )}

        {!loading && cameras.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <WifiOff className="w-6 h-6 text-gray-400 mb-2" />
            <p className="text-xs text-text-secondary">No cameras on edge device</p>
            <p className="text-[10px] text-gray-400 mt-1">Start ibvap_rust to enable feeds</p>
          </div>
        )}

        {!loading && cameras.length > 0 && (
          <div className="grid grid-cols-2 grid-rows-2 gap-2 h-full">
            {cameras.map((cam, idx) => (
              <LiveTile key={cam.id} cam={cam} index={idx} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
