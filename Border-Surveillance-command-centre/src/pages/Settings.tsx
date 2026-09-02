import React from 'react';
import { Settings as SettingsIcon, Save } from 'lucide-react';

export const Settings: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <SettingsIcon className="w-5 h-5 mr-2 text-steel" /> System Configuration
        </h2>
        <button className="btn btn-primary text-xs flex items-center">
          <Save className="w-3 h-3 mr-1" /> Save Changes
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div className="card p-0 flex flex-col">
          <div className="card-header bg-navy/5">Global Alert Thresholds</div>
          <div className="card-body space-y-4">
            <div>
              <label className="block text-xs font-semibold text-navy mb-1 uppercase tracking-wider">High Risk Confidence Threshold</label>
              <input type="range" min="0" max="100" defaultValue="85" className="w-full" />
              <div className="text-right text-xs font-mono mt-1">85%</div>
            </div>
            <div>
              <label className="block text-xs font-semibold text-navy mb-1 uppercase tracking-wider">Unattended Object Timeout (seconds)</label>
              <input type="number" defaultValue="300" className="border border-border rounded px-2 py-1 w-full text-sm" />
            </div>
            <div className="flex items-center space-x-2">
              <input type="checkbox" id="auto-track" defaultChecked className="rounded border-border" />
              <label htmlFor="auto-track" className="text-sm font-semibold text-navy">Enable Auto PTZ Tracking on Critical Alerts</label>
            </div>
          </div>
        </div>

        <div className="card p-0 flex flex-col">
          <div className="card-header bg-navy/5">Network & Integrations</div>
          <div className="card-body space-y-4">
             <div>
              <label className="block text-xs font-semibold text-navy mb-1 uppercase tracking-wider">Backend API Endpoint</label>
              <input type="text" defaultValue="https://api.internal.gov/svas/v2" className="border border-border rounded px-2 py-1 w-full text-sm font-mono" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-navy mb-1 uppercase tracking-wider">WebSocket Stream URL</label>
              <input type="text" defaultValue="wss://stream.internal.gov/ws" className="border border-border rounded px-2 py-1 w-full text-sm font-mono" />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};
