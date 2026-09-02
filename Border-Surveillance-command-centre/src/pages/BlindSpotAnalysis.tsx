import React from 'react';
import { EyeOff, MapPin } from 'lucide-react';
import { mockBlindSpots } from '../mockData';

export const BlindSpotAnalysis: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <EyeOff className="w-5 h-5 mr-2 text-steel" /> Blind Spot Analysis
        </h2>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
        <div className="col-span-12 lg:col-span-4 card flex flex-col min-h-0">
          <div className="card-header bg-navy/5">Identified Blind Spots</div>
          <div className="flex-1 overflow-y-auto custom-scrollbar p-0">
            <ul className="divide-y divide-border">
              {mockBlindSpots.map(bs => (
                <li key={bs.id} className="p-3 hover:bg-gray-50 cursor-pointer">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-mono font-bold text-navy">{bs.id}</span>
                    <span className={`badge ${bs.risk === 'CRITICAL' ? 'badge-critical' : bs.risk === 'HIGH' ? 'badge-warning' : 'badge-info'}`}>{bs.risk} RISK</span>
                  </div>
                  <div className="text-xs text-text-secondary mb-2 flex items-center">
                    <MapPin className="w-3 h-3 mr-1" /> {bs.location}
                  </div>
                  <div className="text-[10px] bg-gray-100 p-1.5 rounded text-gray-600">
                    Affected Cameras: <span className="font-mono font-bold text-navy">{bs.affectedCameras.join(', ')}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-8 flex flex-col space-y-4 min-h-0">
          <div className="flex-1 card flex flex-col overflow-hidden">
            <div className="card-header bg-navy/5">Coverage Gap Analysis (BS-03)</div>
            <div className="flex-1 bg-navy-dark flex flex-col items-center justify-center text-gray-500 relative p-8">
              <div className="w-full max-w-lg aspect-video border border-steel relative flex items-center justify-center">
                {/* Simulated coverage visual */}
                <div className="absolute top-0 left-0 w-1/2 h-full bg-status-info/20 border-r border-status-info flex items-center justify-center text-white/50 text-xs font-mono">CAM-07 FOV</div>
                <div className="absolute top-0 right-0 w-1/3 h-full bg-status-info/20 border-l border-status-info flex items-center justify-center text-white/50 text-xs font-mono">CAM-09 FOV</div>
                
                {/* The gap */}
                <div className="absolute top-0 left-1/2 right-1/3 h-full bg-status-warning/20 border-x border-status-warning border-dashed flex flex-col items-center justify-center z-10">
                  <EyeOff className="w-8 h-8 text-status-warning mb-2 opacity-50" />
                  <span className="text-status-warning font-bold text-xs">UNMONITORED</span>
                  <span className="text-status-warning text-[10px] mt-1 font-mono">14 meters</span>
                </div>
              </div>
              <p className="mt-4 text-xs text-gray-400">Recommendation: Install auxiliary camera covering the 14m gap between CAM-07 and CAM-09 to eliminate High Risk blind spot.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
