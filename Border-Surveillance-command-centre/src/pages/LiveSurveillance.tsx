import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Camera, ShieldAlert, Video, RefreshCw, Maximize2, Grid2x2,
  Grid3x3, Wifi, WifiOff, Activity
} from 'lucide-react';
import { useAuth } from '../auth/AuthProvider';
import { PERMISSIONS } from '../auth/permissions';

// ─── Types ───────────────────────────────────────────────────────────────────

interface EdgeCamera {
  id: string;
  name: string;
  ip: string;
  rtsp: string;
  onvif_uid: string;
}

type StreamStatus = 'loading' | 'live' | 'offline';

// ─── Single camera tile ───────────────────────────────────────────────────────

interface CameraTileProps {
  cam: EdgeCamera;
  index: number;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

const CameraTile: React.FC<CameraTileProps> = ({ cam, index, isSelected, onSelect }) => {
  const [status, setStatus] = useState<StreamStatus>('loading');
  const imgRef = useRef<HTMLImageElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const label = `CAM-${String(index + 1).padStart(2, '0')}`;

  // Direct HTTP URL bypasses Vite proxy — img tags have no CORS restriction.
  // Vite proxy times out on infinite MJPEG streams; direct HTTP does not.
  const streamSrc = `http://localhost:4000/api/stream/${encodeURIComponent(cam.id)}`;

  // Auto-reconnect: on error, wait 3s then retry
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const armTimeout = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setStatus('offline'), 8000);
  }, []);

  const retryStream = useCallback(() => {
    if (imgRef.current) {
      setStatus('loading');
      armTimeout();
      // Force img reload by briefly clearing and resetting src
      const src = imgRef.current.src;
      imgRef.current.src = '';
      setTimeout(() => { if (imgRef.current) imgRef.current.src = src; }, 100);
    }
  }, [armTimeout]);

  useEffect(() => {
    setStatus('loading');
    armTimeout();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [streamSrc, armTimeout]);

  return (
    <div
      onClick={() => onSelect(cam.id)}
      className={`relative overflow-hidden rounded-lg border-2 cursor-pointer transition-all duration-200 bg-black flex flex-col
        ${isSelected
          ? 'border-blue-500 shadow-[0_0_16px_rgba(59,130,246,0.4)] col-span-full row-span-2'
          : 'border-gray-800 hover:border-gray-600'}`}
      style={{ minHeight: isSelected ? '500px' : '180px' }}
    >
      {/* ── MJPEG image — mounted immediately so stream starts ── */}
      <img
        ref={imgRef}
        src={streamSrc}
        alt={`Live feed ${cam.name}`}
        className={`absolute inset-0 w-full h-full transition-opacity duration-500
          ${isSelected ? 'object-contain' : 'object-cover'}
          ${status === 'live' ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => {
          if (timerRef.current) clearTimeout(timerRef.current);
          setStatus('live');
        }}
        onError={() => {
          if (timerRef.current) clearTimeout(timerRef.current);
          setStatus('offline');
          // Auto-retry after 3 seconds
          reconnectTimer.current = setTimeout(() => retryStream(), 3000);
        }}
      />

      {/* ── Placeholder ── */}
      {status !== 'live' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950">
          {status === 'loading' ? (
            <>
              <RefreshCw className="w-7 h-7 text-blue-400 animate-spin mb-2" />
              <span className="text-[10px] font-bold text-gray-400 tracking-widest uppercase">Connecting…</span>
            </>
          ) : (
            <>
              <WifiOff className="w-7 h-7 text-red-500 mb-2" />
              <span className="text-[10px] font-bold text-red-500 tracking-widest uppercase">Stream Offline</span>
              <span className="text-[9px] text-gray-600 mt-1">{cam.ip}</span>
            </>
          )}
        </div>
      )}

      {/* Top gradient */}
      <div className="absolute inset-x-0 top-0 h-12 bg-gradient-to-b from-black/80 to-transparent pointer-events-none" />

      {/* Camera label */}
      <div className="absolute top-2 left-2 z-10">
        <span className="bg-navy/90 text-white text-[9px] font-black px-1.5 py-0.5 rounded tracking-wider border border-white/10">
          {label}
        </span>
      </div>

      {/* Status badge */}
      <div className="absolute top-2 right-2 z-10">
        {status === 'live' ? (
          <span className="flex items-center space-x-1 bg-green-700 text-white text-[9px] font-black px-1.5 py-0.5 rounded">
            <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
            <span>LIVE</span>
          </span>
        ) : status === 'loading' ? (
          <span className="bg-blue-700 text-white text-[9px] font-black px-1.5 py-0.5 rounded">INIT</span>
        ) : (
          <span className="bg-red-700 text-white text-[9px] font-black px-1.5 py-0.5 rounded">OFFLINE</span>
        )}
      </div>

      {/* Bottom gradient */}
      <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/90 to-transparent pointer-events-none" />

      {/* Camera info */}
      <div className="absolute bottom-2 inset-x-2 z-10 flex items-end justify-between">
        <div className="min-w-0">
          <p className="text-white text-[11px] font-bold truncate leading-tight">{cam.name}</p>
          <p className="text-gray-400 text-[9px] mt-0.5 font-mono truncate">{cam.ip}</p>
        </div>
        {status === 'live' && (
          <span className="flex-shrink-0 flex items-center bg-blue-700/80 text-white text-[8px] font-bold px-1.5 py-0.5 rounded ml-2">
            <Activity className="w-2.5 h-2.5 mr-0.5" />AI
          </span>
        )}
      </div>

      {isSelected && (
        <div className="absolute inset-0 border-2 border-blue-500 rounded-lg pointer-events-none" />
      )}
    </div>
  );
};

// ─── Main page ────────────────────────────────────────────────────────────────

type GridMode = '2x2' | '3x3' | '4x2';

export const LiveSurveillance: React.FC = () => {
  const { session, profile, hasPermission: checkPermission } = useAuth();

  const [cameras, setCameras] = useState<EdgeCamera[]>([]);
  const [camError, setCamError]   = useState<string | null>(null);
  const [loading, setLoading]     = useState(true);

  const [gridMode, setGridMode]     = useState<GridMode>('2x2');
  const [selectedCam, setSelectedCam] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const canViewCameras = checkPermission(PERMISSIONS.VIEW_LIVE_CAMERAS);

  // ── Fetch cameras from Rust edge device (no token needed) ──
  const fetchCameras = useCallback(async () => {
    setLoading(true);
    setCamError(null);
    try {
      const res = await fetch('/edge-api/cameras', {
        headers: { 'Authorization': 'Basic YWRtaW46YWRtaW4=' }   // admin:admin
      });
      if (!res.ok) throw new Error(`Edge returned ${res.status} — is ibvap_rust running on port 4000?`);
      const data: EdgeCamera[] = await res.json();
      setCameras(data);
      if (data.length > 0 && !selectedCam) setSelectedCam(data[0].id);
    } catch (err: any) {
      setCamError(err.message);
    } finally {
      setLoading(false);
      setLastRefresh(new Date());
    }
  }, [selectedCam]);

  useEffect(() => {
    if (!canViewCameras) return;
    fetchCameras();
  }, [canViewCameras]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRefresh = () => fetchCameras();

  const gridCols =
    gridMode === '2x2' ? 'grid-cols-2' :
    gridMode === '3x3' ? 'grid-cols-3' :
    'grid-cols-4';

  // ── Access denied ──
  if (!canViewCameras) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-white border border-border rounded-lg">
        <ShieldAlert className="w-16 h-16 text-status-critical mb-4" />
        <h2 className="text-xl font-bold text-text-primary mb-2 uppercase tracking-wide">Access Denied</h2>
        <p className="text-text-secondary">
          Your current role ({profile?.role || 'unknown'}) does not have clearance to view live camera feeds.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-3">

      {/* ── Toolbar ── */}
      <div className="card p-3 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-3">
          <h2 className="text-base font-bold text-navy-dark flex items-center uppercase tracking-wide">
            <Video className="w-4 h-4 mr-2 text-status-warning" />
            Live Surveillance
          </h2>
          {!loading && (
            <span className="text-xs text-text-secondary">
              {cameras.length} camera{cameras.length !== 1 ? 's' : ''} registered
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <span className="flex items-center text-xs font-bold text-status-success bg-status-success/10 px-2 py-1 rounded border border-status-success/20">
            <span className="w-2 h-2 rounded-full bg-status-success mr-1.5 animate-pulse" />
            SYSTEM SECURE
          </span>

          {/* Grid switcher */}
          <div className="flex border border-border rounded overflow-hidden">
            {(['2x2', '3x3', '4x2'] as GridMode[]).map((mode, i) => {
              const Icon = i === 0 ? Grid2x2 : i === 1 ? Grid3x3 : Maximize2;
              return (
                <button
                  key={mode}
                  onClick={() => setGridMode(mode)}
                  className={`p-1.5 ${i === 1 ? 'border-x border-border' : ''} transition-colors
                    ${gridMode === mode ? 'bg-navy text-white' : 'text-text-secondary hover:bg-gray-50'}`}
                  title={`${mode} grid`}
                >
                  <Icon className="w-3.5 h-3.5" />
                </button>
              );
            })}
          </div>

          <button onClick={handleRefresh} className="btn btn-secondary text-xs px-2 py-1 flex items-center space-x-1">
            <RefreshCw className="w-3 h-3" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* ── Error banner ── */}
      {camError && (
        <div className="bg-red-50 border border-red-200 rounded px-3 py-2 text-xs text-red-700 font-medium flex-shrink-0">
          ⚠ {camError}
        </div>
      )}

      {/* ── Content ── */}
      <div className="flex-1 min-h-0 overflow-y-auto">

        {loading && (
          <div className="flex flex-col items-center justify-center h-full">
            <RefreshCw className="w-8 h-8 text-navy animate-spin mb-3" />
            <p className="text-text-secondary text-sm font-medium">Connecting to edge device…</p>
            <p className="text-text-secondary text-xs mt-1 font-mono">http://localhost:4000/api/cameras</p>
          </div>
        )}

        {!loading && cameras.length === 0 && !camError && (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <Camera className="w-12 h-12 text-gray-300 mb-4" />
            <h3 className="font-bold text-text-primary mb-2">No Cameras Found</h3>
            <p className="text-text-secondary text-sm mb-4">
              No cameras registered on the edge device.
              Add cameras using the IBVAP desktop app.
            </p>
            <button onClick={handleRefresh} className="btn btn-primary text-xs">
              <RefreshCw className="w-3 h-3 mr-1" /> Retry
            </button>
          </div>
        )}

        {!loading && cameras.length > 0 && (
          <>
            <div className={`grid ${gridCols} gap-3`}>
              {cameras.map((cam, idx) => (
                <CameraTile
                  key={cam.id}
                  cam={cam}
                  index={idx}
                  isSelected={selectedCam === cam.id}
                  onSelect={setSelectedCam}
                />
              ))}
            </div>

            <div className="mt-3 flex items-center justify-between text-xs text-text-secondary">
              <div className="flex items-center space-x-4">
                <span className="flex items-center space-x-1">
                  <Wifi className="w-3 h-3 text-status-success" />
                  <span>{cameras.length} registered · streaming via port 4000</span>
                </span>
              </div>
              <span>Last synced: {lastRefresh.toLocaleTimeString()}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
