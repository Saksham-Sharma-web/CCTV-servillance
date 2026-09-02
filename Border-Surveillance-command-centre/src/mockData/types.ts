export interface Camera {
  id: string;
  name: string;
  location: string;
  sector: string;
  status: 'ONLINE' | 'OFFLINE' | 'DEGRADED' | 'MAINTENANCE';
  fps: number;
  resolution: string;
  aiStatus: 'ACTIVE' | 'INACTIVE';
  lastHeartbeat: string;
  latency: number;
  coverage: number; // in sq meters
  fov: number; // in degrees
}

export interface Person {
  id: string;
  status: 'TRACKING' | 'TRACK LOST' | 'REACQUIRED';
  firstSeen: string;
  lastSeen: string;
  currentCamera: string | null;
  direction: string;
  speed: number;
  risk: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  confidence: number;
}

export interface Alert {
  id: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'INFO';
  type: string;
  location: string;
  cameraId: string;
  entityId?: string; // Person or Vehicle ID
  timestamp: string;
  status: 'NEW' | 'ACKNOWLEDGED' | 'INVESTIGATING' | 'RESOLVED';
  timeAgo: string; // just for UI ease
}

export interface EventTimelineItem {
  id: string;
  timestamp: string;
  description: string;
  cameraId?: string;
  type: 'PERSON' | 'VEHICLE' | 'ALERT' | 'SYSTEM';
}

export interface BlindSpot {
  id: string;
  location: string;
  risk: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  affectedCameras: string[];
  lastPerson: string;
}

export interface SystemHealth {
  camerasOnline: number;
  camerasTotal: number;
  aiServicesActive: number;
  aiServicesTotal: number;
  databaseStatus: 'OPERATIONAL' | 'DEGRADED';
  storageUsed: number; // percentage
  networkStatus: 'STABLE' | 'UNSTABLE';
}
