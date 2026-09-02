import React from 'react';
import { useDemoMode } from '../../hooks/useDemoMode';
import { Crosshair } from 'lucide-react';

export const SituationMap: React.FC = () => {
  const demoState = useDemoMode();

  return (
    <div className="w-full h-full bg-navy-dark relative overflow-hidden flex flex-col">
      <div className="absolute top-4 left-4 z-10 bg-navy/80 p-2 rounded border border-steel backdrop-blur-sm">
        <div className="text-white text-xs font-bold flex items-center mb-2">
          <Crosshair className="w-4 h-4 mr-1 text-steel-light" /> BORDER SECTOR ALPHA
        </div>
        <div className="space-y-1">
          <div className="flex items-center text-[10px] text-gray-300">
            <div className="w-3 h-3 bg-blue-500/20 border border-blue-500 mr-2 rounded-sm"></div> Camera FOV
          </div>
          <div className="flex items-center text-[10px] text-gray-300">
            <div className="w-3 h-3 bg-red-500/20 border border-red-500 border-dashed mr-2 rounded-sm"></div> Restricted Zone
          </div>
          <div className="flex items-center text-[10px] text-gray-300">
            <div className="w-3 h-3 bg-yellow-500/20 border border-yellow-500 hatch-pattern mr-2 rounded-sm"></div> Blind Spot
          </div>
        </div>
      </div>

      <div className="absolute top-4 right-4 z-10">
        {demoState.activeCamera === null && (
          <div className="bg-status-critical/20 border border-status-critical text-status-critical px-3 py-1.5 rounded animate-pulse text-xs font-bold flex items-center">
            ⚠ TRACK LOST (P-102) IN BLIND SPOT
          </div>
        )}
        {demoState.activeCamera && (
          <div className="bg-status-success/20 border border-status-success text-status-success px-3 py-1.5 rounded text-xs font-bold flex items-center">
            🟢 TRACKING P-102 ({demoState.activeCamera})
          </div>
        )}
      </div>

      {/* Map SVG Canvas */}
      <div className="flex-1 w-full h-full overflow-hidden relative cursor-crosshair">
        <svg width="100%" height="100%" viewBox="0 0 1000 600" className="absolute inset-0">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1F4E79" strokeWidth="0.5" strokeOpacity="0.3"/>
            </pattern>
            <pattern id="hatch" width="10" height="10" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
              <line x1="0" y1="0" x2="0" y2="10" stroke="#B54708" strokeWidth="1" strokeOpacity="0.5" />
            </pattern>
          </defs>
          
          <rect width="100%" height="100%" fill="url(#grid)" />

          {/* Infrastructure Lines */}
          <path d="M 50 50 L 950 50 L 950 550 L 50 550 Z" fill="none" stroke="#3E6078" strokeWidth="2" strokeDasharray="5,5" />
          <rect x="100" y="100" width="150" height="100" fill="none" stroke="#5F6B76" strokeWidth="1" />
          <text x="175" y="155" fill="#5F6B76" fontSize="12" textAnchor="middle" className="font-mono">BUILDING 1</text>
          
          <rect x="700" y="300" width="200" height="150" fill="none" stroke="#5F6B76" strokeWidth="1" />
          <text x="800" y="380" fill="#5F6B76" fontSize="12" textAnchor="middle" className="font-mono">BUILDING 2</text>

          {/* Restricted Zone */}
          <rect x="400" y="100" width="200" height="200" fill="#B42318" fillOpacity="0.05" stroke="#B42318" strokeWidth="1" strokeDasharray="4,4" />
          <text x="500" y="200" fill="#B42318" fontSize="10" textAnchor="middle" opacity="0.7" className="font-mono">RESTRICTED ZONE</text>

          {/* Blind Spot */}
          <rect x="580" y="220" width="120" height="100" fill="url(#hatch)" stroke="#B54708" strokeWidth="1" strokeOpacity="0.5" />
          <text x="640" y="275" fill="#B54708" fontSize="10" textAnchor="middle" className="font-mono font-bold">BS-03</text>

          {/* Cameras and FOVs */}
          <g>
            {/* CAM-04 */}
            <path d="M 150 150 L 250 50 L 350 150 Z" fill="#175CD3" fillOpacity="0.1" stroke="#175CD3" strokeWidth="1" strokeOpacity="0.5" />
            <circle cx="150" cy="150" r="4" fill="#3E6078" />
            <text x="150" y="140" fill="#fff" fontSize="10" textAnchor="middle" className="font-mono">CAM-04</text>
            
            {/* CAM-05 */}
            <path d="M 300 150 L 400 50 L 450 150 Z" fill="#175CD3" fillOpacity="0.1" stroke="#175CD3" strokeWidth="1" strokeOpacity="0.5" />
            <circle cx="300" cy="150" r="4" fill="#3E6078" />
            <text x="300" y="140" fill="#fff" fontSize="10" textAnchor="middle" className="font-mono">CAM-05</text>
            
            {/* CAM-07 */}
            <path d="M 500 200 L 580 120 L 620 200 Z" fill="#175CD3" fillOpacity="0.1" stroke="#175CD3" strokeWidth="1" strokeOpacity="0.5" />
            <circle cx="500" cy="200" r="4" fill="#3E6078" />
            <text x="500" y="190" fill="#fff" fontSize="10" textAnchor="middle" className="font-mono">CAM-07</text>

            {/* CAM-09 */}
            <path d="M 800 350 L 700 250 L 680 350 Z" fill="#175CD3" fillOpacity="0.1" stroke="#175CD3" strokeWidth="1" strokeOpacity="0.5" />
            <circle cx="800" cy="350" r="4" fill="#3E6078" />
            <text x="800" y="340" fill="#fff" fontSize="10" textAnchor="middle" className="font-mono">CAM-09</text>
          </g>

          {/* Prediction Line */}
          {demoState.prediction.active && (
            <g>
              <path 
                d={`M ${demoState.prediction.start.x} ${demoState.prediction.start.y} Q 750 300 ${demoState.prediction.end.x} ${demoState.prediction.end.y}`} 
                fill="none" 
                stroke="#E87817" 
                strokeWidth="2" 
                strokeDasharray="6,4" 
                className="animate-pulse"
              />
              <text x="750" y="300" fill="#E87817" fontSize="10" className="font-mono font-bold">PREDICTED 81%</text>
            </g>
          )}

          {/* Active Person tracking */}
          {demoState.personPosition && (
            <g className="transition-all duration-1000 ease-in-out" transform={`translate(${demoState.personPosition.x}, ${demoState.personPosition.y})`}>
              <circle cx="0" cy="0" r="15" fill={demoState.activeCamera ? "#18794E" : "#B42318"} fillOpacity="0.2" className="animate-ping" />
              <circle cx="0" cy="0" r="6" fill={demoState.activeCamera ? "#18794E" : "#B42318"} />
              <rect x="10" y="-15" width="45" height="16" fill="#0B1F33" rx="2" />
              <text x="32" y="-4" fill="#fff" fontSize="10" textAnchor="middle" className="font-mono font-bold">P-102</text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
};
