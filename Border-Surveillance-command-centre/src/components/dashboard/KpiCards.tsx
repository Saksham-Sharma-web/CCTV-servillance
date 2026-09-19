import React from 'react';
import { Cctv, Users, CarFront, AlertTriangle, ShieldAlert, EyeOff } from 'lucide-react';

export const KpiCards: React.FC = () => {
  const kpis = [
    { label: 'Cameras Online', value: '48 / 50', subtext: '96% operational', icon: Cctv, color: 'text-status-info', bg: 'bg-status-info/10' },
    { label: 'Active People', value: '127', subtext: 'Across 12 sectors', icon: Users, color: 'text-text-primary', bg: 'bg-gray-100' },
    { label: 'Vehicles Detected', value: '42', subtext: '3 watchlisted', icon: CarFront, color: 'text-text-primary', bg: 'bg-gray-100' },
    { label: 'Active Alerts', value: '23', subtext: '+5 in last hour', icon: AlertTriangle, color: 'text-status-warning', bg: 'bg-status-warning/10' },
    { label: 'Critical Alerts', value: '4', subtext: 'Require attention', icon: ShieldAlert, color: 'text-status-critical', bg: 'bg-status-critical/10' },
    { label: 'Blind Spot Events', value: '3', subtext: 'Prediction active', icon: EyeOff, color: 'text-accent-saffron', bg: 'bg-accent-saffron/10' },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
      {kpis.map((kpi, idx) => (
        <div key={idx} className="card p-3 flex flex-col justify-between h-full">
          <div className="flex justify-between items-start mb-2">
            <span className="text-[11px] font-semibold text-text-secondary uppercase tracking-wider leading-tight">{kpi.label}</span>
            <div className={`p-1.5 rounded-sm ${kpi.bg}`}>
              <kpi.icon className={`w-4 h-4 ${kpi.color}`} />
            </div>
          </div>
          <div>
            <div className="text-xl font-bold text-navy-dark leading-none mb-1">{kpi.value}</div>
            <div className="text-[10px] text-text-secondary">{kpi.subtext}</div>
          </div>
        </div>
      ))}
    </div>
  );
};
