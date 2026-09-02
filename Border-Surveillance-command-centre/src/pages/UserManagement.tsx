import React from 'react';
import { Users, Shield, Plus } from 'lucide-react';

export const UserManagement: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <Users className="w-5 h-5 mr-2 text-steel" /> Operator & Access Management
        </h2>
        <button className="btn btn-primary text-xs flex items-center">
          <Plus className="w-3 h-3 mr-1" /> Add Operator
        </button>
      </div>

      <div className="flex-1 card overflow-hidden flex flex-col">
        <div className="overflow-x-auto flex-1 custom-scrollbar p-0">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th>Operator ID</th>
                <th>Name</th>
                <th>Clearance Level</th>
                <th>Role</th>
                <th>Current Status</th>
                <th>Last Login</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {[
                { id: 'OP-4012', name: 'S. Sharma', level: 'Level 4', role: 'Command Lead', status: 'ACTIVE', login: '14:02 UTC' },
                { id: 'OP-1105', name: 'A. Kumar', level: 'Level 2', role: 'Surveillance Tech', status: 'ACTIVE', login: '09:15 UTC' },
                { id: 'OP-0992', name: 'R. Singh', level: 'Level 1', role: 'Field Agent', status: 'OFF DUTY', login: 'Yesterday' },
              ].map((user, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="font-mono text-navy font-semibold">{user.id}</td>
                  <td className="font-semibold text-text-primary text-sm">{user.name}</td>
                  <td>
                    <span className="bg-gray-200 text-gray-700 px-1.5 py-0.5 rounded text-[10px] font-bold flex items-center w-fit">
                      <Shield className="w-3 h-3 mr-1" /> {user.level}
                    </span>
                  </td>
                  <td className="text-sm">{user.role}</td>
                  <td>
                    <span className={`badge ${user.status === 'ACTIVE' ? 'badge-success' : 'badge-neutral'}`}>
                      {user.status}
                    </span>
                  </td>
                  <td className="font-mono text-xs text-text-secondary">{user.login}</td>
                  <td>
                    <button className="text-steel-light hover:text-navy text-xs font-semibold">Manage Access</button>
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
