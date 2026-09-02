import React from 'react';
import { Camera, Settings, Wrench, Video } from 'lucide-react';
import { mockCameras } from '../mockData';

export const CameraInfrastructure: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <Camera className="w-5 h-5 mr-2 text-steel" /> Camera Infrastructure
        </h2>
        <button className="btn btn-primary text-xs flex items-center">
          <Wrench className="w-3 h-3 mr-1" /> Provision New Camera
        </button>
      </div>

      <div className="flex-1 card overflow-hidden flex flex-col">
        <div className="overflow-x-auto flex-1 custom-scrollbar">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th>ID</th>
                <th>Name / Location</th>
                <th>Status</th>
                <th>Network</th>
                <th>Resolution</th>
                <th>FPS</th>
                <th>AI Edge Processing</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {mockCameras.map(cam => (
                <tr key={cam.id} className="hover:bg-gray-50">
                  <td className="font-mono text-navy font-semibold">{cam.id}</td>
                  <td>
                    <div className="font-semibold text-text-primary text-sm">{cam.name}</div>
                    <div className="text-xs text-text-secondary">{cam.location} • {cam.sector}</div>
                  </td>
                  <td>
                    <span className={`badge ${cam.status === 'ONLINE' ? 'badge-success' : 'badge-critical'}`}>
                      {cam.status}
                    </span>
                  </td>
                  <td>
                    <div className="text-xs font-mono">{cam.latency}ms latency</div>
                    <div className="text-[10px] text-text-secondary">Last heartbeat: {cam.lastHeartbeat}</div>
                  </td>
                  <td className="font-mono">{cam.resolution}</td>
                  <td className="font-mono">{cam.fps}</td>
                  <td>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold ${cam.aiStatus === 'ACTIVE' ? 'bg-status-info/10 text-status-info border-status-info/20' : 'bg-gray-100 text-gray-500 border-gray-200'}`}>
                      {cam.aiStatus}
                    </span>
                  </td>
                  <td>
                    <div className="flex space-x-2">
                      <button className="text-steel-light hover:text-navy" title="View Stream"><Video className="w-4 h-4" /></button>
                      <button className="text-text-secondary hover:text-navy" title="Settings"><Settings className="w-4 h-4" /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
