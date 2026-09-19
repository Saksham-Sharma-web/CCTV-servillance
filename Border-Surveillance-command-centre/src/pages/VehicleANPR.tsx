import React from 'react';
import { Search, CarFront } from 'lucide-react';

export const VehicleANPR: React.FC = () => {
  const mockVehicles = [
    { id: 'V-204', type: 'SUV', color: 'White', plate: 'UP16AB1234', cam: 'CAM-07', time: '14:31', dir: 'North', conf: 96, status: 'NORMAL' },
    { id: 'V-311', type: 'Sedan', color: 'Black', plate: 'DL09XY7788', cam: 'CAM-03', time: '14:28', dir: 'East', conf: 92, status: 'WATCHLIST' },
    { id: 'V-102', type: 'Truck', color: 'Blue', plate: 'UP14CD2201', cam: 'CAM-01', time: '14:15', dir: 'North', conf: 85, status: 'NORMAL' },
  ];

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex justify-between items-center bg-white p-3 rounded-md border border-border shadow-sm">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <CarFront className="w-5 h-5 mr-2 text-steel" /> Vehicle & ANPR Surveillance
        </h2>
        <div className="relative w-64">
          <input 
            type="text" 
            placeholder="Search License Plate..." 
            className="w-full pl-8 pr-3 py-1.5 text-sm border border-border rounded"
          />
          <Search className="w-4 h-4 text-gray-400 absolute left-2.5 top-2" />
        </div>
      </div>

      <div className="flex-1 card overflow-hidden">
        <div className="overflow-x-auto h-full">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th>Vehicle ID</th>
                <th>Type / Color</th>
                <th>License Plate</th>
                <th>Status</th>
                <th>Camera</th>
                <th>Timestamp</th>
                <th>Confidence</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {mockVehicles.map(v => (
                <tr key={v.id} className="hover:bg-gray-50">
                  <td className="font-mono text-navy font-semibold">{v.id}</td>
                  <td>{v.color} {v.type}</td>
                  <td>
                    <span className="font-mono bg-yellow-100 border border-yellow-300 text-yellow-800 px-2 py-0.5 rounded font-bold tracking-widest text-sm shadow-sm">
                      {v.plate}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${v.status === 'WATCHLIST' ? 'badge-critical' : 'badge-neutral'}`}>
                      {v.status}
                    </span>
                  </td>
                  <td className="font-mono">{v.cam}</td>
                  <td className="font-mono text-text-secondary">{v.time}</td>
                  <td className="font-mono">{v.conf}%</td>
                  <td>
                    <button className="text-steel-light hover:text-navy text-xs font-semibold">Track</button>
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
