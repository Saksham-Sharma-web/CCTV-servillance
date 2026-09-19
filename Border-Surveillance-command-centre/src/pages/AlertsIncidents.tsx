import React, { useState } from 'react';
import { mockAlerts } from '../mockData';
import { AlertTriangle, Filter, CheckCircle2, Search } from 'lucide-react';

export const AlertsIncidents: React.FC = () => {
  const [activeTab, setActiveTab] = useState('ALL');

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex justify-between items-center bg-white p-3 rounded-md border border-border shadow-sm">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <AlertTriangle className="w-5 h-5 mr-2 text-status-warning" /> 
          Alerts & Incidents
        </h2>
        <div className="flex space-x-2">
          <div className="relative w-64">
            <input 
              type="text" 
              placeholder="Search Incident ID, Location..." 
              className="w-full pl-8 pr-3 py-1.5 text-sm border border-border rounded"
            />
            <Search className="w-4 h-4 text-gray-400 absolute left-2.5 top-2" />
          </div>
          <button className="btn btn-secondary flex items-center text-xs">
            <Filter className="w-3 h-3 mr-1" /> Filters
          </button>
        </div>
      </div>

      <div className="flex-1 card flex flex-col overflow-hidden">
        <div className="border-b border-border">
          <nav className="flex space-x-1 px-2" aria-label="Tabs">
            {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'RESOLVED'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-2 text-xs font-semibold uppercase tracking-wider border-b-2 transition-colors ${
                  activeTab === tab 
                  ? 'border-navy text-navy' 
                  : 'border-transparent text-text-secondary hover:text-navy hover:border-gray-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex-1 overflow-auto custom-scrollbar">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0 z-10">
              <tr>
                <th>Incident ID</th>
                <th>Severity</th>
                <th>Event Type</th>
                <th>Location</th>
                <th>Camera</th>
                <th>Entity ID</th>
                <th>Timestamp</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {mockAlerts.map(alert => (
                <tr key={alert.id} className="hover:bg-gray-50 transition-colors cursor-pointer">
                  <td className="font-mono text-navy font-semibold">{alert.id}</td>
                  <td>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${
                      alert.severity === 'CRITICAL' ? 'bg-status-critical/10 text-status-critical border-status-critical/20' : 
                      alert.severity === 'HIGH' ? 'bg-status-warning/10 text-status-warning border-status-warning/20' : 
                      'bg-status-info/10 text-status-info border-status-info/20'
                    }`}>
                      {alert.severity}
                    </span>
                  </td>
                  <td className="font-semibold text-text-primary text-xs">{alert.type}</td>
                  <td>{alert.location}</td>
                  <td className="font-mono text-xs">{alert.cameraId}</td>
                  <td className="font-mono">{alert.entityId || '-'}</td>
                  <td className="font-mono text-text-secondary">{new Date(alert.timestamp).toLocaleTimeString()}</td>
                  <td>
                    <span className="badge badge-neutral text-[10px] font-bold uppercase">{alert.status}</span>
                  </td>
                  <td>
                    <div className="flex space-x-2">
                      <button className="text-xs text-steel-light hover:text-navy font-semibold">View</button>
                      <button className="text-xs text-status-success hover:text-green-800 font-semibold flex items-center">
                        <CheckCircle2 className="w-3 h-3 mr-1" /> Resolve
                      </button>
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
