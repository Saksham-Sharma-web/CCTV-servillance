import React from 'react';
import { BrainCircuit, Activity, Zap } from 'lucide-react';

export const AIAnalytics: React.FC = () => {
  const modules = [
    { name: 'Person Detection', status: 'ACTIVE', detections: 127, conf: '94.2%', fps: 24, latency: '82 ms' },
    { name: 'Vehicle Detection', status: 'ACTIVE', detections: 42, conf: '96.1%', fps: 15, latency: '120 ms' },
    { name: 'Person Tracking', status: 'ACTIVE', detections: 84, conf: '89.5%', fps: 24, latency: '95 ms' },
    { name: 'Face Detection', status: 'ACTIVE', detections: 19, conf: '82.3%', fps: 10, latency: '150 ms' },
    { name: 'ANPR', status: 'ACTIVE', detections: 31, conf: '97.8%', fps: 15, latency: '110 ms' },
    { name: 'Intrusion Detection', status: 'ACTIVE', detections: 7, conf: '91.0%', fps: 30, latency: '65 ms' },
    { name: 'Loitering Detection', status: 'ACTIVE', detections: 5, conf: '88.4%', fps: 10, latency: '200 ms' },
    { name: 'Suspicious Activity', status: 'ACTIVE', detections: 3, conf: '76.2%', fps: 5, latency: '350 ms' },
    { name: 'Night Movement', status: 'STANDBY', detections: 0, conf: '-', fps: 0, latency: '-' },
  ];

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <BrainCircuit className="w-5 h-5 mr-2 text-steel" /> AI Analytics Engine
        </h2>
        <div className="flex space-x-2">
          <div className="bg-status-success/10 text-status-success px-3 py-1 text-xs font-bold rounded flex items-center">
            <Activity className="w-4 h-4 mr-1" /> SYSTEM HEALTHY
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {modules.map((mod, i) => (
          <div key={i} className="card flex flex-col p-0 overflow-hidden">
            <div className="bg-navy/5 px-4 py-2 border-b border-border flex justify-between items-center">
              <span className="font-semibold text-sm text-navy-dark">{mod.name}</span>
              <span className={`badge ${mod.status === 'ACTIVE' ? 'badge-success' : 'badge-neutral'}`}>{mod.status}</span>
            </div>
            <div className="p-4 flex-1">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold mb-1">Detections</div>
                  <div className="text-xl font-bold text-navy-dark">{mod.detections}</div>
                </div>
                <div>
                  <div className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold mb-1">Confidence</div>
                  <div className="text-xl font-mono text-steel-light">{mod.conf}</div>
                </div>
                <div>
                  <div className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold mb-1">Processing</div>
                  <div className="text-sm font-mono text-text-primary">{mod.fps} FPS</div>
                </div>
                <div>
                  <div className="text-[10px] text-text-secondary uppercase tracking-wider font-semibold mb-1">Latency</div>
                  <div className="text-sm font-mono text-text-primary">{mod.latency}</div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
