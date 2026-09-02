import React, { useState } from 'react';
import { mockCameras } from '../mockData';
import { Filter, Grid, Maximize, Settings } from 'lucide-react';

export const LiveSurveillance: React.FC = () => {
  const [filter, setFilter] = useState('ALL');

  const filteredCameras = mockCameras.filter(c => {
    if (filter === 'OFFLINE') return c.status === 'OFFLINE';
    if (filter === 'ONLINE') return c.status === 'ONLINE';
    return true;
  });

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex justify-between items-center bg-white p-3 rounded-md border border-border shadow-sm">
        <h2 className="text-lg font-bold text-navy-dark">Live Surveillance</h2>
        <div className="flex space-x-4 items-center">
          <div className="flex items-center space-x-2 text-sm border-r border-border pr-4">
            <span className="text-text-secondary">View:</span>
            <button className="p-1 bg-gray-100 rounded text-navy"><Grid className="w-4 h-4" /></button>
            <button className="p-1 hover:bg-gray-100 rounded text-text-secondary"><Maximize className="w-4 h-4" /></button>
          </div>
          <div className="flex items-center space-x-2 text-sm">
            <Filter className="w-4 h-4 text-text-secondary" />
            <select 
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="border border-border rounded px-2 py-1 text-sm bg-white"
            >
              <option value="ALL">All Cameras</option>
              <option value="ONLINE">Online Only</option>
              <option value="OFFLINE">Offline Only</option>
            </select>
          </div>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 auto-rows-max overflow-y-auto custom-scrollbar pr-2">
        {filteredCameras.map(cam => (
          <div key={cam.id} className="card overflow-hidden flex flex-col group">
            <div className="h-48 bg-gray-900 relative flex items-center justify-center border-b border-navy-dark">
              {cam.status === 'OFFLINE' ? (
                <div className="text-center">
                  <Settings className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                  <span className="text-status-critical font-bold text-sm tracking-widest">OFFLINE</span>
                </div>
              ) : (
                <span className="text-gray-700 text-sm font-mono tracking-widest">{cam.id}_STREAM</span>
              )}
              
              {cam.status === 'ONLINE' && (
                <div className="absolute top-2 right-2 bg-status-success text-white text-[9px] px-1.5 py-0.5 rounded font-bold tracking-wider">LIVE</div>
              )}
              <div className="absolute top-2 left-2 text-white text-xs font-mono opacity-75">{cam.id}</div>
            </div>
            <div className="p-3 bg-white">
              <div className="font-semibold text-sm text-navy-dark truncate" title={cam.name}>{cam.name}</div>
              <div className="text-xs text-text-secondary mb-2">{cam.location} • {cam.sector}</div>
              <div className="flex justify-between items-center text-[10px] font-mono text-gray-500">
                <span>{cam.resolution} @ {cam.fps}fps</span>
                <span className={`px-1.5 rounded ${cam.aiStatus === 'ACTIVE' ? 'bg-status-info/10 text-status-info border border-status-info/20' : 'bg-gray-100 text-gray-500 border border-gray-200'}`}>
                  AI {cam.aiStatus}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
