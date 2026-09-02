import React from 'react';
import { mockSystemHealth } from '../mockData';
import { ServerCrash, Database, HardDrive, Wifi, Cpu } from 'lucide-react';

export const SystemHealth: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3">
        <h2 className="text-lg font-bold text-navy-dark">System Infrastructure Health</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div className="card flex flex-col p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-sm text-text-secondary uppercase tracking-wider">Cameras</h3>
            <ServerCrash className="w-5 h-5 text-steel-light" />
          </div>
          <div className="text-3xl font-bold text-navy-dark mb-1">
            {mockSystemHealth.camerasOnline} <span className="text-sm font-normal text-text-secondary">/ {mockSystemHealth.camerasTotal}</span>
          </div>
          <div className="text-xs text-status-success font-semibold">ONLINE</div>
        </div>

        <div className="card flex flex-col p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-sm text-text-secondary uppercase tracking-wider">AI Services</h3>
            <Cpu className="w-5 h-5 text-steel-light" />
          </div>
          <div className="text-3xl font-bold text-navy-dark mb-1">
            {mockSystemHealth.aiServicesActive} <span className="text-sm font-normal text-text-secondary">/ {mockSystemHealth.aiServicesTotal}</span>
          </div>
          <div className="text-xs text-status-success font-semibold">ACTIVE</div>
        </div>

        <div className="card flex flex-col p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-sm text-text-secondary uppercase tracking-wider">Database</h3>
            <Database className="w-5 h-5 text-steel-light" />
          </div>
          <div className="text-xl font-bold text-navy-dark mt-2 mb-1 uppercase">
            {mockSystemHealth.databaseStatus}
          </div>
          <div className="text-xs text-status-success font-semibold">CONNECTED</div>
        </div>

        <div className="card flex flex-col p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-sm text-text-secondary uppercase tracking-wider">Storage</h3>
            <HardDrive className="w-5 h-5 text-steel-light" />
          </div>
          <div className="text-3xl font-bold text-navy-dark mb-2">
            {mockSystemHealth.storageUsed}%
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1.5 mb-1">
            <div className="bg-navy h-1.5 rounded-full" style={{ width: `${mockSystemHealth.storageUsed}%` }}></div>
          </div>
        </div>

      </div>

      <div className="flex-1 card flex flex-col">
        <div className="card-header bg-navy/5">Network Status</div>
        <div className="card-body flex items-center justify-center flex-col text-gray-500">
          <Wifi className="w-16 h-16 mb-4 text-status-success opacity-50" />
          <h3 className="text-lg font-semibold text-navy">Network {mockSystemHealth.networkStatus}</h3>
          <p className="text-sm">All command center connections are secure and stable.</p>
        </div>
      </div>
    </div>
  );
};
