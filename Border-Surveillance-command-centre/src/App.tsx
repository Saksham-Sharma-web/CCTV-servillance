import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { CommandCenter } from './pages/CommandCenter';
import { LiveSurveillance } from './pages/LiveSurveillance';
import { PeopleTracking } from './pages/PeopleTracking';
import { AlertsIncidents } from './pages/AlertsIncidents';
import { Investigation } from './pages/Investigation';
import { VehicleANPR } from './pages/VehicleANPR';
import { SystemHealth } from './pages/SystemHealth';
import { AIAnalytics } from './pages/AIAnalytics';
import { ZonesVirtualFence } from './pages/ZonesVirtualFence';
import { BlindSpotAnalysis } from './pages/BlindSpotAnalysis';
import { CameraInfrastructure } from './pages/CameraInfrastructure';
import { EvidenceAudit } from './pages/EvidenceAudit';
import { Reports } from './pages/Reports';
import { UserManagement } from './pages/UserManagement';
import { Settings } from './pages/Settings';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<CommandCenter />} />
          <Route path="surveillance" element={<LiveSurveillance />} />
          <Route path="map" element={<CommandCenter />} /> {/* Redirect Situation Map to Command Center since it's embedded there */}
          <Route path="tracking" element={<PeopleTracking />} />
          <Route path="vehicles" element={<VehicleANPR />} />
          <Route path="alerts" element={<AlertsIncidents />} />
          <Route path="analytics" element={<AIAnalytics />} />
          <Route path="zones" element={<ZonesVirtualFence />} />
          <Route path="blind-spots" element={<BlindSpotAnalysis />} />
          <Route path="investigation" element={<Investigation />} />
          <Route path="cameras" element={<CameraInfrastructure />} />
          <Route path="health" element={<SystemHealth />} />
          <Route path="audit" element={<EvidenceAudit />} />
          <Route path="reports" element={<Reports />} />
          <Route path="users" element={<UserManagement />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
