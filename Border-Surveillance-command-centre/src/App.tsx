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
import { Login } from './pages/Login';
import { AuthProvider } from './auth/AuthProvider';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { PERMISSIONS } from './auth/permissions';

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }>
            <Route index element={<CommandCenter />} />
            <Route path="surveillance" element={<LiveSurveillance />} />
            <Route path="map" element={<CommandCenter />} /> {/* Redirect Situation Map to Command Center since it's embedded there */}
            <Route path="tracking" element={<PeopleTracking />} />
            <Route path="vehicles" element={<VehicleANPR />} />
            <Route path="alerts" element={<AlertsIncidents />} />
            <Route path="analytics" element={<AIAnalytics />} />
            <Route path="zones" element={
              <ProtectedRoute requiredPermission={PERMISSIONS.MANAGE_ZONES}>
                <ZonesVirtualFence />
              </ProtectedRoute>
            } />
            <Route path="blind-spots" element={<BlindSpotAnalysis />} />
            <Route path="investigation" element={
              <ProtectedRoute requiredPermission={PERMISSIONS.INVESTIGATE_INCIDENTS}>
                <Investigation />
              </ProtectedRoute>
            } />
            <Route path="cameras" element={
              <ProtectedRoute requiredPermission={PERMISSIONS.MANAGE_CAMERAS}>
                <CameraInfrastructure />
              </ProtectedRoute>
            } />
            <Route path="health" element={
              <ProtectedRoute requiredPermission={PERMISSIONS.VIEW_CAMERA_HEALTH}>
                <SystemHealth />
              </ProtectedRoute>
            } />
            <Route path="audit" element={<EvidenceAudit />} />
            <Route path="reports" element={
              <ProtectedRoute requiredPermission={PERMISSIONS.GENERATE_REPORTS}>
                <Reports />
              </ProtectedRoute>
            } />
            <Route path="users" element={
              <ProtectedRoute requiredPermission={PERMISSIONS.MANAGE_USERS}>
                <UserManagement />
              </ProtectedRoute>
            } />
            <Route path="settings" element={
              <ProtectedRoute requiredPermission={PERMISSIONS.MANAGE_SYSTEM_SETTINGS}>
                <Settings />
              </ProtectedRoute>
            } />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
