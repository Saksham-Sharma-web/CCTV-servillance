import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Cctv, 
  Map as MapIcon, 
  UserCircle, 
  CarFront, 
  AlertTriangle, 
  BrainCircuit, 
  Fence, 
  EyeOff, 
  Search, 
  ServerCrash, 
  Activity, 
  ShieldCheck, 
  FileText, 
  Users, 
  Settings 
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const NAV_ITEMS = [
  { name: 'Command Center', path: '/', icon: LayoutDashboard },
  { name: 'Live Surveillance', path: '/surveillance', icon: Cctv },
  { name: 'Situation Map', path: '/map', icon: MapIcon },
  { name: 'People Tracking', path: '/tracking', icon: UserCircle },
  { name: 'Vehicle & ANPR', path: '/vehicles', icon: CarFront },
  { name: 'Alerts & Incidents', path: '/alerts', icon: AlertTriangle },
  { name: 'AI Analytics', path: '/analytics', icon: BrainCircuit },
  { name: 'Zones & Virtual Fence', path: '/zones', icon: Fence },
  { name: 'Blind Spot Analysis', path: '/blind-spots', icon: EyeOff },
  { name: 'Investigation', path: '/investigation', icon: Search },
  { name: 'Camera Infrastructure', path: '/cameras', icon: ServerCrash },
  { name: 'System Health', path: '/health', icon: Activity },
  { name: 'Evidence & Audit', path: '/audit', icon: ShieldCheck },
  { name: 'Reports', path: '/reports', icon: FileText },
  { name: 'User Management', path: '/users', icon: Users },
  { name: 'Settings', path: '/settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  return (
    <div className="w-64 bg-navy-dark text-gray-300 flex flex-col h-full border-r border-navy flex-shrink-0">
      <div className="h-16 flex items-center px-4 border-b border-navy/50 bg-navy">
        <ShieldCheck className="w-6 h-6 text-accent-saffron mr-3" />
        <div>
          <h1 className="text-white font-bold text-sm tracking-wide">NB-SVAS</h1>
          <p className="text-[10px] text-gray-400 uppercase tracking-widest">Command Center</p>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto py-4 custom-scrollbar">
        <nav className="space-y-1 px-2">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    "flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors",
                    isActive 
                      ? "bg-steel-light text-white" 
                      : "text-gray-300 hover:bg-navy hover:text-white"
                  )
                }
              >
                <Icon className={cn("mr-3 flex-shrink-0 h-5 w-5")} />
                {item.name}
              </NavLink>
            );
          })}
        </nav>
      </div>
      
      <div className="p-4 border-t border-navy/50 bg-navy/20">
        <div className="flex items-center">
          <div className="h-2 w-2 rounded-full bg-status-success mr-2 shadow-[0_0_8px_rgba(24,121,78,0.8)]"></div>
          <span className="text-xs font-semibold text-gray-300 tracking-wide">SYSTEM SECURE</span>
        </div>
      </div>
    </div>
  );
};
