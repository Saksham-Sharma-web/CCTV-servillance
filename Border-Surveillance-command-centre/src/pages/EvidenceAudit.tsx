import React from 'react';
import { Database, Download, Lock, Search } from 'lucide-react';

export const EvidenceAudit: React.FC = () => {
  const auditLogs = [
    { id: 'EV-2023-1102-01', type: 'Intrusion Footage', timestamp: '2023-11-02 14:32:00', requestedBy: 'OP-4012 (S. Sharma)', authLevel: 'Level 4', status: 'LOCKED', chainOfCustody: 'INTACT' },
    { id: 'EV-2023-1102-02', type: 'ANPR Log Export', timestamp: '2023-11-02 12:15:22', requestedBy: 'OP-1105 (A. Kumar)', authLevel: 'Level 2', status: 'AVAILABLE', chainOfCustody: 'INTACT' },
    { id: 'EV-2023-1101-05', type: 'System Config Change', timestamp: '2023-11-01 09:00:11', requestedBy: 'SYSADMIN', authLevel: 'Level 5', status: 'ARCHIVED', chainOfCustody: 'INTACT' },
  ];

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <Database className="w-5 h-5 mr-2 text-steel" /> Evidence & Audit Logs
        </h2>
        <div className="relative w-64">
          <input 
            type="text" 
            placeholder="Search EV-ID..." 
            className="w-full pl-8 pr-3 py-1.5 text-sm border border-border rounded"
          />
          <Search className="w-4 h-4 text-gray-400 absolute left-2.5 top-2" />
        </div>
      </div>

      <div className="flex-1 card overflow-hidden flex flex-col">
        <div className="card-header bg-navy/5">Chain of Custody & Audit Trail</div>
        <div className="overflow-x-auto flex-1 custom-scrollbar p-0">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th>Evidence ID</th>
                <th>Record Type</th>
                <th>Timestamp (UTC)</th>
                <th>Operator</th>
                <th>Auth Level</th>
                <th>Custody Status</th>
                <th>Data Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {auditLogs.map((log, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="font-mono text-navy font-semibold">{log.id}</td>
                  <td>{log.type}</td>
                  <td className="font-mono text-text-secondary">{log.timestamp}</td>
                  <td className="font-semibold text-sm">{log.requestedBy}</td>
                  <td>
                    <span className="bg-gray-200 text-gray-700 px-1.5 py-0.5 rounded text-[10px] font-bold">
                      {log.authLevel}
                    </span>
                  </td>
                  <td>
                    <span className="text-status-success text-xs font-bold flex items-center">
                      <Lock className="w-3 h-3 mr-1" /> {log.chainOfCustody}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${log.status === 'LOCKED' ? 'badge-critical' : log.status === 'AVAILABLE' ? 'badge-info' : 'badge-neutral'}`}>
                      {log.status}
                    </span>
                  </td>
                  <td>
                    <button className="text-steel-light hover:text-navy flex items-center text-xs font-semibold">
                      <Download className="w-3 h-3 mr-1" /> Export
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
