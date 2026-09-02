import React from 'react';
import { Search, Bell, User } from 'lucide-react';
import { mockSystemHealth } from '../../mockData';

export const Header: React.FC = () => {
  return (
    <header className="h-16 bg-white border-b border-border flex items-center justify-between px-6 flex-shrink-0">
      <div className="flex items-center flex-1">
        <div className="relative w-full max-w-xl">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search camera, person ID, vehicle, plate, incident..."
            className="block w-full pl-10 pr-3 py-2 border border-border rounded-md leading-5 bg-background placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:border-steel-light focus:ring-1 focus:ring-steel-light sm:text-sm transition-colors"
          />
        </div>
      </div>

      <div className="flex items-center space-x-6 ml-4">
        {/* System Status */}
        <div className="hidden md:flex flex-col items-end">
          <span className="text-[10px] text-text-secondary font-semibold tracking-wider uppercase">System Status</span>
          <div className="flex items-center">
            <div className={`h-2 w-2 rounded-full mr-1.5 ${mockSystemHealth.databaseStatus === 'OPERATIONAL' ? 'bg-status-success' : 'bg-status-warning'}`}></div>
            <span className="text-sm font-semibold text-text-primary">
              {mockSystemHealth.databaseStatus}
            </span>
          </div>
        </div>

        {/* Notifications */}
        <button className="relative p-1 text-gray-400 hover:text-gray-500 focus:outline-none transition-colors">
          <span className="absolute top-1 right-1 block h-2 w-2 rounded-full bg-status-critical ring-2 ring-white"></span>
          <Bell className="h-6 w-6" />
        </button>

        {/* User Profile */}
        <div className="flex items-center border-l border-border pl-6">
          <div className="flex flex-col items-end mr-3">
            <span className="text-sm font-semibold text-text-primary">Duty Officer</span>
            <span className="text-xs text-text-secondary">Operations Control</span>
          </div>
          <div className="h-9 w-9 rounded-full bg-steel flex items-center justify-center text-white">
            <User className="h-5 w-5" />
          </div>
        </div>
      </div>
    </header>
  );
};
