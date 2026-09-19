import React from 'react';
import { mockEvents } from '../../mockData';
import { User, Car, AlertTriangle, Info } from 'lucide-react';

export const EventTimeline: React.FC = () => {
  const getIcon = (type: string) => {
    switch (type) {
      case 'PERSON': return <User className="w-3 h-3 text-steel-light" />;
      case 'VEHICLE': return <Car className="w-3 h-3 text-gray-500" />;
      case 'ALERT': return <AlertTriangle className="w-3 h-3 text-status-warning" />;
      default: return <Info className="w-3 h-3 text-status-info" />;
    }
  };

  return (
    <div className="card h-full flex flex-col">
      <div className="card-header bg-navy/5">
        <span className="text-navy-dark">Real-time Event Timeline</span>
      </div>
      <div className="card-body p-0 flex-1 overflow-y-auto custom-scrollbar">
        <div className="relative p-4">
          <div className="absolute top-4 bottom-4 left-[27px] w-px bg-border"></div>
          <div className="space-y-4">
            {mockEvents.map((event) => (
              <div key={event.id} className="relative flex items-start group">
                <div className="absolute left-[-2px] w-2 h-2 rounded-full bg-border group-hover:bg-steel transition-colors mt-1.5"></div>
                <div className="flex-1 ml-6">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="text-xs font-mono text-text-secondary">{event.timestamp}</span>
                    <span className="bg-gray-100 border border-border rounded px-1 text-[9px] font-mono text-navy">
                      {event.cameraId}
                    </span>
                  </div>
                  <div className="text-sm text-text-primary flex items-start">
                    <div className="mt-1 mr-2 bg-gray-100 rounded-sm p-1">
                      {getIcon(event.type)}
                    </div>
                    <span className="leading-snug pt-0.5">{event.description}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
