import React from 'react';
import { mockCameras } from '../../mockData';
import { Maximize2, Settings2 } from 'lucide-react';

export const LiveCameraGrid: React.FC = () => {
  // Take first 4 cameras for the 2x2 grid
  const gridCameras = mockCameras.slice(0, 4);

  return (
    <div className="card h-full flex flex-col">
      <div className="card-header bg-navy/5 flex justify-between items-center">
        <div className="flex items-center text-navy-dark">
          <span className="relative flex h-2 w-2 mr-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-critical opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-status-critical"></span>
          </span>
          <span>Live Camera Wall</span>
        </div>
        <div className="flex space-x-2">
          <button className="text-text-secondary hover:text-navy transition-colors"><Settings2 className="w-4 h-4" /></button>
          <button className="text-text-secondary hover:text-navy transition-colors"><Maximize2 className="w-4 h-4" /></button>
        </div>
      </div>
      <div className="card-body p-2 flex-1">
        <div className="grid grid-cols-2 grid-rows-2 gap-2 h-full">
          {gridCameras.map(cam => (
            <div key={cam.id} className="relative bg-gray-900 rounded overflow-hidden group border border-gray-800">
              {/* This would be the actual video stream */}
              <div className="absolute inset-0 bg-navy-dark flex items-center justify-center">
                <span className="text-gray-700 text-sm font-mono tracking-widest">{cam.id}_STREAM</span>
              </div>
              
              {/* Overlays */}
              <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/60 pointer-events-none"></div>
              
              <div className="absolute top-2 left-2 flex flex-col pointer-events-none">
                <span className="text-white text-xs font-bold font-mono tracking-wider">{cam.id}</span>
                <span className="text-gray-300 text-[10px]">{cam.name}</span>
              </div>
              
              <div className="absolute top-2 right-2 pointer-events-none">
                <div className="flex items-center space-x-2">
                  {cam.status === 'ONLINE' && <span className="bg-status-success text-white text-[9px] px-1 rounded font-bold tracking-wider">LIVE</span>}
                  <span className="text-gray-300 text-[10px] font-mono">{cam.fps} FPS</span>
                </div>
              </div>

              <div className="absolute bottom-2 left-2 flex items-center space-x-2 pointer-events-none">
                <div className="flex items-center bg-black/50 rounded px-1.5 py-0.5">
                  <div className={`h-1.5 w-1.5 rounded-full mr-1 ${cam.aiStatus === 'ACTIVE' ? 'bg-status-info' : 'bg-gray-500'}`}></div>
                  <span className="text-white text-[9px] font-bold">AI {cam.aiStatus}</span>
                </div>
                <div className="text-white text-[10px]">
                  👤 {Math.floor(Math.random() * 3)}
                </div>
              </div>
              
              <div className="absolute bottom-2 right-2 text-white text-[10px] font-mono opacity-80 pointer-events-none">
                {new Date().toLocaleTimeString()}
              </div>

              {/* Hover actions */}
              <div className="absolute inset-0 bg-navy-dark/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[1px]">
                <button className="bg-white/10 hover:bg-white/20 text-white border border-white/30 rounded px-3 py-1.5 text-xs font-medium backdrop-blur-sm transition-all">
                  View Detail
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
