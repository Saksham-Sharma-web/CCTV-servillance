import type { Alert, BlindSpot, Camera, EventTimelineItem, Person, SystemHealth } from './types';

export const mockCameras: Camera[] = [
  { id: 'CAM-01', name: 'Main Gate - Entrance 1', location: 'Main Gate', sector: 'Sector A', status: 'ONLINE', fps: 24, resolution: '1080p', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:21', latency: 82, coverage: 150, fov: 90 },
  { id: 'CAM-02', name: 'Main Gate - Entrance 2', location: 'Main Gate', sector: 'Sector A', status: 'ONLINE', fps: 24, resolution: '1080p', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:21', latency: 85, coverage: 140, fov: 85 },
  { id: 'CAM-03', name: 'Loading Bay', location: 'Loading Bay', sector: 'Sector B', status: 'ONLINE', fps: 30, resolution: '1080p', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:19', latency: 60, coverage: 200, fov: 110 },
  { id: 'CAM-04', name: 'North Corridor Alpha', location: 'North Corridor', sector: 'Sector C', status: 'ONLINE', fps: 24, resolution: '720p', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:20', latency: 75, coverage: 120, fov: 75 },
  { id: 'CAM-05', name: 'North Corridor Beta', location: 'North Corridor', sector: 'Sector C', status: 'ONLINE', fps: 24, resolution: '720p', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:20', latency: 78, coverage: 120, fov: 75 },
  { id: 'CAM-07', name: 'Restricted Area Entry', location: 'Restricted Zone', sector: 'Sector A', status: 'ONLINE', fps: 30, resolution: '4K', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:22', latency: 45, coverage: 80, fov: 60 },
  { id: 'CAM-08', name: 'Watch Tower 03', location: 'Perimeter Fence', sector: 'Sector D', status: 'ONLINE', fps: 15, resolution: '1080p', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:18', latency: 120, coverage: 500, fov: 120 },
  { id: 'CAM-09', name: 'Sector B Hub', location: 'Sector B', sector: 'Sector B', status: 'ONLINE', fps: 24, resolution: '1080p', aiStatus: 'ACTIVE', lastHeartbeat: '14:32:21', latency: 80, coverage: 160, fov: 90 },
  { id: 'CAM-17', name: 'South Gate Auxiliary', location: 'South Gate', sector: 'Sector E', status: 'OFFLINE', fps: 0, resolution: '1080p', aiStatus: 'INACTIVE', lastHeartbeat: '14:28:11', latency: 0, coverage: 130, fov: 80 },
];

export const mockPersons: Person[] = [
  { id: 'P-102', status: 'TRACKING', firstSeen: '14:21:04', lastSeen: '14:32:18', currentCamera: 'CAM-07', direction: 'North-East', speed: 1.7, risk: 'HIGH', confidence: 96 },
  { id: 'P-204', status: 'TRACKING', firstSeen: '14:10:12', lastSeen: '14:31:00', currentCamera: 'CAM-02', direction: 'South', speed: 1.2, risk: 'MEDIUM', confidence: 92 },
  { id: 'P-309', status: 'TRACK LOST', firstSeen: '13:55:40', lastSeen: '14:25:30', currentCamera: null, direction: 'West', speed: 1.5, risk: 'LOW', confidence: 88 },
];

export const mockAlerts: Alert[] = [
  { id: 'INC-2026-001', severity: 'CRITICAL', type: 'Unauthorized Access Detected', location: 'Main Gate - Entrance 2', cameraId: 'CAM-07', entityId: 'P-102', timestamp: '2026-09-02T14:30:18', status: 'NEW', timeAgo: '2 min ago' },
  { id: 'INC-2026-002', severity: 'HIGH', type: 'Restricted Zone Entry', location: 'Sector A', cameraId: 'CAM-07', entityId: 'P-204', timestamp: '2026-09-02T14:27:18', status: 'ACKNOWLEDGED', timeAgo: '5 min ago' },
  { id: 'INC-2026-003', severity: 'MEDIUM', type: 'Loitering Detected', location: 'Loading Bay', cameraId: 'CAM-03', timestamp: '2026-09-02T14:22:18', status: 'INVESTIGATING', timeAgo: '10 min ago' },
  { id: 'INC-2026-004', severity: 'INFO', type: 'Vehicle Detected', location: 'Front Desk', cameraId: 'CAM-03', entityId: 'V-204', timestamp: '2026-09-02T14:17:18', status: 'RESOLVED', timeAgo: '15 min ago' },
];

export const mockEvents: EventTimelineItem[] = [
  { id: 'EV-1', timestamp: '14:32:18', description: 'Person P-102 detected', cameraId: 'CAM-07', type: 'PERSON' },
  { id: 'EV-2', timestamp: '14:31:04', description: 'Virtual fence violation', cameraId: 'CAM-07', type: 'ALERT' },
  { id: 'EV-3', timestamp: '14:29:41', description: 'P-102 transitioned from CAM-05 → CAM-07', cameraId: 'CAM-07', type: 'PERSON' },
  { id: 'EV-4', timestamp: '14:27:19', description: 'Vehicle V-204 detected', cameraId: 'CAM-05', type: 'VEHICLE' },
  { id: 'EV-5', timestamp: '14:25:03', description: 'Loitering detected', cameraId: 'CAM-03', type: 'ALERT' },
];

export const mockBlindSpots: BlindSpot[] = [
  { id: 'BS-01', location: 'North Corridor', risk: 'HIGH', affectedCameras: ['CAM-04', 'CAM-05'], lastPerson: 'P-309' },
  { id: 'BS-02', location: 'Storage Area', risk: 'MEDIUM', affectedCameras: ['CAM-03'], lastPerson: 'None' },
  { id: 'BS-03', location: 'Sector A Transition', risk: 'CRITICAL', affectedCameras: ['CAM-07', 'CAM-09'], lastPerson: 'P-102' },
];

export const mockSystemHealth: SystemHealth = {
  camerasOnline: 48,
  camerasTotal: 50,
  aiServicesActive: 8,
  aiServicesTotal: 8,
  databaseStatus: 'OPERATIONAL',
  storageUsed: 72,
  networkStatus: 'STABLE',
};
