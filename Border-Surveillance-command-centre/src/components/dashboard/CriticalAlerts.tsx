import React from 'react';
import { mockAlerts } from '../../mockData';
import { ShieldAlert, MapPin, Clock } from 'lucide-react';

export const CriticalAlerts: React.FC = () => {
  return (
    <div className="card h-full flex flex-col">
      <div className="card-header bg-navy/5 flex justify-between items-center">
        <div className="flex items-center text-navy-dark">
          <ShieldAlert className="w-4 h-4 mr-2 text-status-critical" />
          <span>Active Alerts</span>
        </div>
        <span className="badge badge-critical text-[10px]">{mockAlerts.length}</span>
      </div>
      <div className="card-body p-0 flex-1 overflow-y-auto custom-scrollbar">
        <ul className="divide-y divide-border">
          {mockAlerts.map(alert => (
            <li key={alert.id} className="p-3 hover:bg-gray-50 transition-colors cursor-pointer group">
              <div className="flex justify-between items-start mb-1.5">
                <span className={`text-xs font-bold ${alert.severity === 'CRITICAL' ? 'text-status-critical' : alert.severity === 'HIGH' ? 'text-status-warning' : alert.severity === 'MEDIUM' ? 'text-yellow-600' : 'text-status-info'}`}>
                  {alert.severity}
                </span>
                <span className="text-[10px] text-text-secondary flex items-center">
                  <Clock className="w-3 h-3 mr-1" />
                  {alert.timeAgo}
                </span>
              </div>
              <div className="text-sm font-semibold text-navy-dark mb-1 leading-tight group-hover:text-steel-light transition-colors">
                {alert.type}
              </div>
              <div className="flex items-center justify-between mt-2 text-xs">
                <div className="flex items-center text-text-secondary">
                  <MapPin className="w-3 h-3 mr-1" />
                  <span className="truncate max-w-[120px]">{alert.location}</span>
                </div>
                <div className="font-medium text-navy bg-gray-100 px-1.5 py-0.5 rounded border border-gray-200">
                  {alert.cameraId}
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
