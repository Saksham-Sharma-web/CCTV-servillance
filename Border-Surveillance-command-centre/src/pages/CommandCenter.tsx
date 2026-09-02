import React from 'react';
import { KpiCards } from '../components/dashboard/KpiCards';
import { CriticalAlerts } from '../components/dashboard/CriticalAlerts';
import { LiveCameraGrid } from '../components/dashboard/LiveCameraGrid';
import { EventTimeline } from '../components/dashboard/EventTimeline';
import { SituationMap } from '../components/map/SituationMap';

export const CommandCenter: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Top row: KPI Cards */}
      <div className="flex-none">
        <KpiCards />
      </div>

      {/* Main content grid */}
      <div className="flex-1 min-h-0 grid grid-cols-12 gap-4">
        
        {/* Left Column (Main map area + cameras) */}
        <div className="col-span-12 lg:col-span-9 flex flex-col space-y-4 min-h-0">
          
          {/* Situation Map - Takes majority of space */}
          <div className="flex-1 card overflow-hidden relative p-0 border-steel">
            <SituationMap />
          </div>
          
          {/* Bottom row in left column: Cameras */}
          <div className="h-64 flex-none">
            <LiveCameraGrid />
          </div>
        </div>

        {/* Right Column (Alerts + Timeline) */}
        <div className="col-span-12 lg:col-span-3 flex flex-col space-y-4 min-h-0">
          
          {/* Critical Alerts */}
          <div className="flex-1 min-h-0">
            <CriticalAlerts />
          </div>

          {/* Event Timeline */}
          <div className="flex-1 min-h-0">
            <EventTimeline />
          </div>

        </div>
      </div>
    </div>
  );
};
