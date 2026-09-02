import React from 'react';
import { mockPersons } from '../mockData';
import { Search, MapPin, Activity } from 'lucide-react';

export const PeopleTracking: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex justify-between items-center bg-white p-3 rounded-md border border-border shadow-sm">
        <h2 className="text-lg font-bold text-navy-dark">People Tracking</h2>
        <div className="relative w-64">
          <input 
            type="text" 
            placeholder="Search Person ID..." 
            className="w-full pl-8 pr-3 py-1.5 text-sm border border-border rounded"
          />
          <Search className="w-4 h-4 text-gray-400 absolute left-2.5 top-2" />
        </div>
      </div>

      <div className="flex-1 card overflow-hidden flex flex-col">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th>Person ID</th>
                <th>Status</th>
                <th>First Seen</th>
                <th>Last Seen</th>
                <th>Current Location</th>
                <th>Direction</th>
                <th>Speed</th>
                <th>Risk</th>
                <th>Confidence</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {mockPersons.map(person => (
                <tr key={person.id} className="hover:bg-gray-50 transition-colors">
                  <td className="font-mono font-semibold text-navy">{person.id}</td>
                  <td>
                    <span className={`badge ${person.status === 'TRACKING' ? 'badge-success' : 'badge-critical'}`}>
                      {person.status}
                    </span>
                  </td>
                  <td className="font-mono text-text-secondary">{person.firstSeen}</td>
                  <td className="font-mono text-text-secondary">{person.lastSeen}</td>
                  <td>
                    {person.currentCamera ? (
                      <span className="flex items-center text-text-primary">
                        <MapPin className="w-3 h-3 mr-1 text-gray-400" />
                        {person.currentCamera}
                      </span>
                    ) : (
                      <span className="text-text-secondary italic">Unknown</span>
                    )}
                  </td>
                  <td>{person.direction}</td>
                  <td className="font-mono">{person.speed.toFixed(1)} m/s</td>
                  <td>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${person.risk === 'CRITICAL' || person.risk === 'HIGH' ? 'bg-status-critical/10 text-status-critical border-status-critical/20' : 'bg-status-info/10 text-status-info border-status-info/20'}`}>
                      {person.risk}
                    </span>
                  </td>
                  <td>
                    <div className="flex items-center">
                      <div className="w-16 h-1.5 bg-gray-200 rounded-full mr-2 overflow-hidden">
                        <div className="h-full bg-navy" style={{ width: `${person.confidence}%` }}></div>
                      </div>
                      <span className="text-[10px] text-text-secondary font-mono">{person.confidence}%</span>
                    </div>
                  </td>
                  <td>
                    <button className="text-steel-light hover:text-navy text-xs font-semibold flex items-center">
                      <Activity className="w-3 h-3 mr-1" /> Investigate
                    </button>
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
