import React, { useEffect, useState } from 'react';
import { Camera, ShieldAlert, Video } from 'lucide-react';
import { useAuth } from '../auth/AuthProvider';
import { hasPermission, PERMISSIONS } from '../auth/permissions';

export const LiveSurveillance: React.FC = () => {
  const { session, profile, hasPermission: checkPermission } = useAuth();
  const [streamToken, setStreamToken] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Only operators and above can view cameras (enforced by UI and backend gatekeeper)
    const fetchToken = async () => {
      try {
        const res = await fetch('/api/cameras/token', {
          headers: {
            'Authorization': `Bearer ${session?.access_token}`
          }
        });
        const result = await res.json();
        
        if (!result.success) throw new Error(result.error?.message || 'Failed to get stream token');
        
        setStreamToken(result.data.token);
      } catch (err: any) {
        setError(err.message);
      }
    };

    fetchToken();
  }, [session]);

  const canViewCameras = checkPermission(PERMISSIONS.VIEW_LIVE_CAMERAS);

  if (!canViewCameras) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-surface border border-border rounded-lg">
        <ShieldAlert className="w-16 h-16 text-status-critical mb-4" />
        <h2 className="text-xl font-bold text-text-primary mb-2 uppercase tracking-wide">Access Denied</h2>
        <p className="text-text-secondary">Your current role ({profile?.role || 'unknown'}) does not have clearance to view live camera feeds.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center uppercase tracking-wide">
          <Video className="w-5 h-5 mr-2 text-status-warning" /> Live Surveillance Feed
        </h2>
        <div className="flex items-center space-x-2">
          <span className="flex items-center text-xs font-bold text-status-success bg-status-success/10 px-2 py-1 rounded">
            <span className="w-2 h-2 rounded-full bg-status-success mr-2 animate-pulse"></span>
            SYSTEM SECURE
          </span>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-2 gap-4">
        {/* Camera 1 */}
        <div className="card overflow-hidden flex flex-col relative group">
          <div className="absolute top-2 left-2 z-10 bg-black/70 px-2 py-1 rounded border border-white/20 backdrop-blur-sm flex items-center space-x-2">
            <Camera className="w-3 h-3 text-white" />
            <span className="text-[10px] font-bold text-white uppercase tracking-wider">CAM_01 / NORTH PERIMETER</span>
          </div>
          
          <div className="flex-1 bg-black flex items-center justify-center relative">
            {!streamToken && !error && (
              <span className="text-steel animate-pulse text-sm font-bold tracking-widest uppercase">Negotiating Secure Stream...</span>
            )}
            
            {error && (
              <span className="text-status-critical text-sm font-bold tracking-widest uppercase">{error}</span>
            )}

            {streamToken && (
              <img 
                src={`https://localhost:3000/api/stream/cam1?token=${streamToken}`} 
                alt="Live Feed CAM_01" 
                className="w-full h-full object-cover"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                  target.parentElement!.innerHTML = '<div class="text-steel-light font-bold text-sm tracking-widest uppercase">Stream Offline (Accept TLS Cert in Browser first!)</div>';
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
