import React from 'react';
import { Fence, Plus, Settings2 } from 'lucide-react';

export const ZonesVirtualFence: React.FC = () => {
  const zones = [
    { id: 'Z-01', name: 'Sector A Restricted', type: 'Restricted Area', cameras: 4, severity: 'CRITICAL', status: 'ACTIVE' },
    { id: 'Z-02', name: 'Main Gate Approach', type: 'Patrol Zone', cameras: 2, severity: 'MEDIUM', status: 'ACTIVE' },
    { id: 'VF-01', name: 'Perimeter Line North', type: 'Virtual Fence', cameras: 3, severity: 'HIGH', status: 'ACTIVE' },
    { id: 'Z-03', name: 'Loading Dock', type: 'Entry Zone', cameras: 1, severity: 'LOW', status: 'INACTIVE' },
  ];

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <Fence className="w-5 h-5 mr-2 text-steel" /> Zones & Virtual Fences
        </h2>
        <button className="btn btn-primary text-xs flex items-center">
          <Plus className="w-3 h-3 mr-1" /> Create Zone
        </button>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
        <div className="col-span-12 lg:col-span-4 card flex flex-col min-h-0">
          <div className="card-header bg-navy/5">Managed Areas</div>
          <div className="flex-1 overflow-y-auto custom-scrollbar p-0">
            <ul className="divide-y divide-border">
              {zones.map(zone => (
                <li key={zone.id} className="p-3 hover:bg-gray-50 cursor-pointer">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-bold text-navy text-sm">{zone.name}</span>
                    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${zone.status === 'ACTIVE' ? 'bg-status-success/10 text-status-success border-status-success/20' : 'bg-gray-100 text-gray-500'}`}>{zone.status}</span>
                  </div>
                  <div className="flex justify-between text-xs text-text-secondary mt-2">
                    <span className="font-mono">{zone.id} • {zone.type}</span>
                    <span className={`font-semibold ${zone.severity === 'CRITICAL' ? 'text-status-critical' : zone.severity === 'HIGH' ? 'text-status-warning' : ''}`}>{zone.severity}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-8 flex flex-col space-y-4 min-h-0">
          <div className="flex-1 card flex flex-col overflow-hidden">
            <div className="card-header bg-navy/5 flex justify-between">
              <span>Zone Configuration Map</span>
              <Settings2 className="w-4 h-4 text-text-secondary" />
            </div>
            <div className="flex-1 bg-navy-dark flex items-center justify-center text-gray-500 relative">
              <div className="absolute inset-4 border-2 border-status-critical border-dashed bg-status-critical/10 flex items-center justify-center">
                <span className="text-status-critical font-bold tracking-widest text-lg opacity-50">Z-01 RESTRICTED AREA</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
