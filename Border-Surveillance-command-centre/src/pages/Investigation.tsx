import React, { useState } from 'react';
import { Search, History, Activity, ShieldCheck, MapPin } from 'lucide-react';
import { mockPersons } from '../mockData';

export const Investigation: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('P-102');
  const person = mockPersons.find(p => p.id === searchQuery);

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <Search className="w-5 h-5 mr-2 text-steel" /> Investigation Workspace
        </h2>
        <div className="flex space-x-2 w-96">
          <input 
            type="text" 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search Person ID, Vehicle, Plate..." 
            className="flex-1 px-3 py-1.5 text-sm border border-border rounded"
          />
          <button className="btn btn-primary text-xs">Search Database</button>
        </div>
      </div>

      {person ? (
        <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
          
          {/* Identity Column */}
          <div className="col-span-12 lg:col-span-3 card flex flex-col">
            <div className="card-header bg-navy/5">Identity Overview</div>
            <div className="card-body">
              <div className="flex flex-col items-center mb-6 border-b border-border pb-6">
                <div className="w-24 h-24 bg-gray-200 border-2 border-border mb-3 flex items-center justify-center text-gray-500 text-xs text-center p-2">
                  [CCTV Snapshot Placeholder]
                </div>
                <h3 className="text-xl font-mono font-bold text-navy">{person.id}</h3>
                <span className={`mt-1 badge ${person.status === 'TRACKING' ? 'badge-success' : 'badge-critical'}`}>
                  {person.status}
                </span>
              </div>
              
              <div className="space-y-4 text-sm">
                <div>
                  <div className="text-text-secondary text-[10px] uppercase tracking-wider font-semibold">First Seen</div>
                  <div className="font-mono">{person.firstSeen}</div>
                </div>
                <div>
                  <div className="text-text-secondary text-[10px] uppercase tracking-wider font-semibold">Last Seen</div>
                  <div className="font-mono">{person.lastSeen}</div>
                </div>
                <div>
                  <div className="text-text-secondary text-[10px] uppercase tracking-wider font-semibold">Risk Level</div>
                  <div className="font-bold text-status-critical">{person.risk}</div>
                </div>
                <div>
                  <div className="text-text-secondary text-[10px] uppercase tracking-wider font-semibold">Match Confidence</div>
                  <div className="font-mono">{person.confidence}%</div>
                </div>
              </div>
              
              <div className="mt-6 pt-6 border-t border-border">
                <button className="w-full btn btn-secondary text-xs mb-2 flex justify-center items-center">
                  <Activity className="w-3 h-3 mr-2" /> View Full Trajectory
                </button>
                <button className="w-full btn bg-navy hover:bg-navy-dark text-white text-xs flex justify-center items-center">
                  <ShieldCheck className="w-3 h-3 mr-2" /> Add to Watchlist
                </button>
              </div>
            </div>
          </div>

          {/* Timeline and Map Column */}
          <div className="col-span-12 lg:col-span-6 flex flex-col space-y-4 min-h-0">
            <div className="flex-1 card flex flex-col">
              <div className="card-header bg-navy/5">Movement Trajectory Map</div>
              <div className="flex-1 bg-navy-dark flex items-center justify-center text-gray-500">
                [Detailed Trajectory Map View]
              </div>
            </div>
          </div>

          {/* Evidence Column */}
          <div className="col-span-12 lg:col-span-3 card flex flex-col">
            <div className="card-header bg-navy/5 flex items-center justify-between">
              <span>Camera History</span>
              <History className="w-4 h-4 text-text-secondary" />
            </div>
            <div className="card-body overflow-y-auto custom-scrollbar p-0">
              <ul className="divide-y divide-border">
                {[
                  { cam: 'CAM-02', time: '14:21:04', conf: '94%' },
                  { cam: 'CAM-04', time: '14:25:31', conf: '96%' },
                  { cam: 'CAM-05', time: '14:28:15', conf: '95%' },
                  { cam: 'CAM-07', time: '14:32:18', conf: '96%' },
                  { cam: 'BLIND SPOT', time: '14:33:00', conf: '-' },
                ].map((ev, i) => (
                  <li key={i} className="p-3 hover:bg-gray-50">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-mono font-bold text-navy text-sm">{ev.cam}</span>
                      <span className="font-mono text-xs text-text-secondary">{ev.time}</span>
                    </div>
                    <div className="text-xs text-text-secondary flex items-center">
                      <MapPin className="w-3 h-3 mr-1" />
                      Confidence: {ev.conf}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

        </div>
      ) : (
        <div className="flex-1 card flex items-center justify-center text-text-secondary">
          No entity selected. Please search to begin investigation.
        </div>
      )}
    </div>
  );
};
