import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './index.css';

// ─── TYPES ────────────────────────────────────────────────────────────────────
interface Camera {
  id: string;
  name: string;
  ip: string;
  rtsp: string;
  is_restricted: boolean;
  onvif_uid: string;
}

interface AppEvent {
  id: string;
  camera_id: string;
  camera_name: string;
  event_type: string;
  confidence: number;
  timestamp: string;
  media_path: string;
}

interface Toast {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

type ModalType = null | 'add-camera' | 'edit-camera' | 'settings' | 'confirm-reset';
type SettingsTab = 'password' | 'onvif' | 'ai-ref';

const API = 'http://127.0.0.1:3000';

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function isAlert(type: string) {
  return /FENCE|INTRUSION|BLACKLIST|SUSPICIOUS|UNATTENDED|CROSSING|TRESPASS/i.test(type);
}
function eventClass(type: string) {
  if (isAlert(type)) return 'alert-type';
  if (/PERSON|FACE|RECOGNITION|KNOWN/i.test(type)) return 'warn-type';
  return 'info-type';
}
function pct(confidence: number) { return Math.round(confidence * 100); }
function fmt(s: string) { return s.replace(/_/g, ' '); }

// ─── APP ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [cameras,     setCameras]     = useState<Camera[]>([]);
  const [selectedCam, setSelectedCam] = useState<string | null>(null);
  const [events,      setEvents]      = useState<AppEvent[]>([]);
  const [modal,       setModal]       = useState<ModalType>(null);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('password');
  const [editCam,     setEditCam]     = useState<Camera | null>(null);
  const [toasts,      setToasts]      = useState<Toast[]>([]);
  const [streamToken, setStreamToken] = useState('');
  const [loading,     setLoading]     = useState(false);
  const [camSearch,   setCamSearch]   = useState('');
  const [discovering, setDiscovering] = useState(false);
  const [aiFilePath,  setAiFilePath]  = useState('');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const toastId = useRef(0);

  // ─── TOAST ───
  const toast = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = ++toastId.current;
    setToasts(p => [...p, { id, message, type }]);
    setTimeout(() => setToasts(p => p.filter(t => t.id !== id)), 3500);
  }, []);

  // ─── DATA LOADERS (via Tauri invoke — no HTTPS needed) ───
  const loadCameras = useCallback(async () => {
    try {
      const cams = await invoke<Camera[]>('get_cameras');
      setCameras(cams);
      if (cams.length > 0 && !selectedCam) setSelectedCam(cams[0].id);
    } catch (e) { console.error('get_cameras', e); }
  }, [selectedCam]);

  const loadEvents = useCallback(async () => {
    try {
      const evs = await invoke<AppEvent[]>('get_events', { limit: 100 });
      setEvents(evs);
    } catch (e) { console.error('get_events', e); }
  }, []);

  const refreshToken = useCallback(async () => {
    try {
      const tok = await invoke<string>('get_stream_token');
      setStreamToken(tok);
    } catch {}
  }, []);

  // ─── BOOT ───
  useEffect(() => {
    loadCameras();
    loadEvents();
    refreshToken();
    const ci = setInterval(loadCameras, 8000);
    const ei = setInterval(loadEvents,  3000);
    const ti = setInterval(refreshToken, 50 * 60 * 1000); // refresh token every 50 min

    // WS for real-time event push
    let ws: WebSocket;
    const connect = () => {
      ws = new WebSocket('ws://127.0.0.1:3000/ws/events');
      ws.onmessage = () => loadEvents();
      ws.onclose   = () => setTimeout(connect, 5000);
    };
    connect();

    return () => {
      clearInterval(ci); clearInterval(ei); clearInterval(ti);
      ws?.close();
    };
  }, []); // eslint-disable-line

  // ─── DERIVED ───
  const filteredCameras = cameras.filter(c =>
    !camSearch || c.name.toLowerCase().includes(camSearch.toLowerCase()) || c.ip.includes(camSearch)
  );
  const selectedCamera = cameras.find(c => c.id === selectedCam);
  const alertCount     = events.filter(e => isAlert(e.event_type)).length;
  const streamUrl      = selectedCam && streamToken
    ? `${API}/api/stream/${selectedCam}?token=${streamToken}`
    : '';

  // ─── HANDLERS ─────────────────────────────────────────────────────────────

  async function handleAddCamera(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd   = new FormData(e.currentTarget);
    const rtsp = (fd.get('rtsp') as string).trim();
    const name = (fd.get('name') as string).trim();
    const user = (fd.get('user') as string).trim();
    const pass = (fd.get('pass') as string).trim();
    if (!rtsp) { toast('RTSP URL is required', 'error'); return; }
    setLoading(true);
    try {
      const newId = await invoke<string>('add_camera', { rtsp, name: name || 'Camera', user, pass });
      toast(`Camera "${name || 'Camera'}" added!`, 'success');
      setModal(null);
      await loadCameras();
      setSelectedCam(newId);
    } catch (err: unknown) {
      toast(`Failed to add camera: ${err}`, 'error');
    } finally { setLoading(false); }
  }

  async function handleEditCamera(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!editCam) return;
    const fd       = new FormData(e.currentTarget);
    const new_name = (fd.get('name') as string).trim();
    if (!new_name) { toast('Name cannot be empty', 'error'); return; }
    setLoading(true);
    try {
      await invoke('rename_camera', { id: editCam.id, newName: new_name });
      toast('Camera renamed', 'success');
      setModal(null); setEditCam(null);
      loadCameras();
    } catch (err: unknown) {
      toast(`Failed: ${err}`, 'error');
    } finally { setLoading(false); }
  }

  async function handleRemoveCamera(id: string, name: string) {
    if (!confirm(`Remove "${name}"?`)) return;
    try {
      await invoke('remove_camera', { id });
      if (selectedCam === id) setSelectedCam(null);
      toast('Camera removed', 'info');
      loadCameras();
    } catch (err: unknown) {
      toast(`Failed: ${err}`, 'error');
    }
  }

  async function handleFactoryReset() {
    setLoading(true);
    try {
      await invoke('factory_reset');
      toast('Factory reset complete', 'success');
      setModal(null);
      setCameras([]); setEvents([]); setSelectedCam(null);
    } catch (err: unknown) {
      toast(`Failed: ${err}`, 'error');
    } finally { setLoading(false); }
  }

  async function handleDiscoverCameras() {
    setDiscovering(true);
    toast('Scanning network for ONVIF cameras…', 'info');
    try {
      const msg = await invoke<string>('discover_cameras');
      await loadCameras();
      toast(msg, 'success');
    } catch (err: unknown) { 
      toast(`Discovery failed: ${err}`, 'error'); 
    }
    finally { setDiscovering(false); }
  }

  async function handleChangePassword(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd          = new FormData(e.currentTarget);
    const username    = (fd.get('username') as string).trim();
    const current_pass = (fd.get('current_pass') as string);
    const new_pass    = (fd.get('new_pass') as string);
    const confirm     = (fd.get('confirm') as string);
    if (!username) { toast('Enter your username', 'error'); return; }
    if (new_pass !== confirm) { toast('New passwords do not match', 'error'); return; }
    if (new_pass.length < 4) { toast('Password must be at least 4 characters', 'error'); return; }
    setLoading(true);
    try {
      await invoke('reset_password', { username, currentPass: current_pass, newPass: new_pass });
      toast('Password changed successfully', 'success');
      (e.target as HTMLFormElement).reset();
    } catch (err: unknown) {
      toast(`Failed: ${err}`, 'error');
    } finally { setLoading(false); }
  }

  async function handleSaveOnvif(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd       = new FormData(e.currentTarget);
    const username = (fd.get('onvif_user') as string).trim();
    const password = (fd.get('onvif_pass') as string);
    setLoading(true);
    try {
      await invoke('save_onvif_settings', { username, password });
      toast('ONVIF settings saved', 'success');
    } catch (err: unknown) {
      toast(`Failed: ${err}`, 'error');
    } finally { setLoading(false); }
  }

  async function handlePickAiFile() {
    try {
      const path = await invoke<string>('select_ai_reference');
      setAiFilePath(path);
    } catch { /* cancelled */ }
  }

  async function handleRegisterAiRef(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd  = new FormData(e.currentTarget);
    const tag = (fd.get('tag') as string).trim();
    if (!tag)         { toast('Enter a person name', 'error'); return; }
    if (!aiFilePath)  { toast('Select a reference image first', 'error'); return; }
    setLoading(true);
    try {
      await invoke('register_ai_reference', { tag, path: aiFilePath });
      toast(`"${tag}" registered for AI recognition`, 'success');
      setAiFilePath('');
      (e.target as HTMLFormElement).reset();
    } catch (err: unknown) {
      toast(`Failed: ${err}`, 'error');
    } finally { setLoading(false); }
  }

  function openSettings(tab: SettingsTab) {
    setSettingsTab(tab);
    setModal('settings');
  }

  function closeModal() { setModal(null); setEditCam(null); }

  // ─── RENDER ────────────────────────────────────────────────────────────────
  return (
    <>
      {/* TOASTS */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>
            {t.type === 'success' ? '✓' : t.type === 'error' ? '✗' : 'ℹ'} {t.message}
          </div>
        ))}
      </div>

      <div className="app-root">

        {/* ═══ SIDEBAR ═══ */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-logo-icon">📹</div>
            <div className="sidebar-logo-text">
              <h1>IBVAP Edge</h1>
              <span>Command Center</span>
            </div>
          </div>

          {/* Camera search */}
          <div className="cam-search-row">
            <input
              className="cam-search-input"
              placeholder="Search cameras…"
              value={camSearch}
              onChange={e => setCamSearch(e.target.value)}
            />
          </div>

          <div className="sidebar-content">
            <div className="section-label">
              Cameras ({filteredCameras.length}{cameras.length !== filteredCameras.length ? ` / ${cameras.length}` : ''})
            </div>

            {filteredCameras.length === 0 && (
              <div className="empty-state">
                <p>{cameras.length === 0 ? 'No cameras registered' : 'No matches'}</p>
              </div>
            )}

            {filteredCameras.map(cam => (
              <div
                key={cam.id}
                className={`camera-item ${selectedCam === cam.id ? 'active' : ''}`}
                onClick={() => setSelectedCam(cam.id)}
              >
                <div className="status-dot online" />
                <div className="camera-info">
                  <h3>{cam.name || cam.id}</h3>
                  <p>{cam.ip || cam.rtsp.replace(/rtsp:\/\/[^@]*@/, '').split(':')[0] || cam.id.slice(0, 22)}</p>
                </div>
                <div className="cam-actions" onClick={e => e.stopPropagation()}>
                  <button
                    className="icon-btn"
                    title="Rename"
                    onClick={() => { setEditCam(cam); setModal('edit-camera'); }}
                  >✏</button>
                  <button
                    className="icon-btn danger"
                    title="Remove"
                    onClick={() => handleRemoveCamera(cam.id, cam.name)}
                  >✕</button>
                </div>
              </div>
            ))}
          </div>

          {/* Sidebar actions */}
          <div className="sidebar-actions">
            <button className="sidebar-action-btn primary" onClick={() => setModal('add-camera')}>
              ＋ Add Camera
            </button>
            <button
              className={`sidebar-action-btn ${discovering ? 'loading' : ''}`}
              onClick={handleDiscoverCameras}
              disabled={discovering}
            >
              {discovering ? '⟳ Scanning…' : '🔍 Discover'}
            </button>
          </div>

          <div className="sidebar-footer">
            <button className="footer-btn" onClick={() => openSettings('password')}>🔑 Password</button>
            <button className="footer-btn" onClick={() => openSettings('onvif')}>📡 ONVIF</button>
            <button className="footer-btn" onClick={() => openSettings('ai-ref')}>🤖 AI Ref</button>
            <button className="footer-btn danger" onClick={() => setModal('confirm-reset')}>🔄</button>
          </div>
        </aside>

        {/* ═══ MAIN AREA ═══ */}
        <div className="main-area">
          {/* Top bar */}
          <div className="topbar">
            <div className="topbar-left">
              {selectedCamera ? (
                <>
                  <div className="live-badge"><div className="live-badge-dot" />LIVE</div>
                  <div>
                    <div className="topbar-cam-name">{selectedCamera.name}</div>
                    <div className="topbar-cam-meta">{selectedCamera.ip || 'RTSP'} · {selectedCamera.id.slice(0, 22)}…</div>
                  </div>
                </>
              ) : (
                <div className="topbar-cam-meta">No camera selected — select one from the sidebar</div>
              )}
            </div>
            <div className="topbar-right">
              <span className="topbar-stat">{cameras.length} cam{cameras.length !== 1 ? 's' : ''}</span>
              <span className="topbar-stat alert-stat">{alertCount} alert{alertCount !== 1 ? 's' : ''}</span>
            </div>
          </div>

          {/* Video */}
          <div className="video-stage">
            {streamUrl ? (
              <img
                key={streamUrl}
                src={streamUrl}
                alt="Live feed"
                onError={() => {}}
              />
            ) : (
              <div className="no-cam-placeholder">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                  <circle cx="12" cy="13" r="4"/>
                </svg>
                <p>{cameras.length === 0 ? 'Add a camera to get started' : 'Select a camera to view the live feed'}</p>
              </div>
            )}
          </div>
        </div>

        {/* ═══ EVENTS PANEL ═══ */}
        <div className="events-panel">
          <div className="events-header">
            <span>AI Events</span>
            {alertCount > 0 && <div className="alert-badge">{alertCount} alerts</div>}
          </div>

          <div className="events-list">
            {events.length === 0 && (
              <div className="empty-state"><p>No events yet</p></div>
            )}
            {events.slice(0, 80).map((ev, i) => {
              let typeLabel = (ev.event_type || '').trim();
              if (!typeLabel) typeLabel = 'EVENT';
              
              const cls = typeLabel.toLowerCase();
              const fmt = (t: string) => t.replace(/_/g, ' ').toUpperCase();
              const confidence = pct(ev.confidence);
              return (
                <div 
                  key={`${ev.id}-${i}`} 
                  className={`event-card ${cls}`}
                  onClick={() => setExpandedEvent(expandedEvent === ev.id ? null : ev.id)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div className={`event-type-label ${cls}`}>{fmt(typeLabel)}</div>
                    <div className="event-time-label" style={{ marginTop: 0 }}>{ev.timestamp.replace('T', ' ').slice(0, 19)}</div>
                  </div>
                  {(ev.camera_name || ev.camera_id) && (
                    <div className="event-cam-label">{ev.camera_name || ev.camera_id}</div>
                  )}
                  <div className="conf-bar">
                    <div className={`conf-fill ${cls}`} style={{ width: `${confidence}%` }} />
                  </div>
                  {expandedEvent === ev.id && (
                    <div className="event-snapshot visible" style={{ display: 'block', minHeight: '100px', backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <img
                        src={`${API}/api/snapshots/${ev.id}?auth=ignored`}
                        alt="Snapshot missing"
                        style={{ width: '100%' }}
                        onError={e => { 
                          const img = e.target as HTMLImageElement;
                          img.style.display = 'none';
                          img.parentElement!.innerHTML = '<span style="color: #666; font-size: 0.8rem;">Snapshot not available</span>';
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ═══ MODAL: ADD CAMERA ═══ */}
      {modal === 'add-camera' && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>📹 Add Camera</h2>
              <button type="button" className="modal-close" onClick={closeModal}>✕</button>
            </div>
            <form onSubmit={handleAddCamera}>
              <div className="form-group">
                <label className="form-label">Camera Name</label>
                <input className="form-input" name="name" placeholder="e.g. Front Door" autoFocus />
              </div>
              <div className="form-group">
                <label className="form-label">RTSP URL *</label>
                <input
                  className="form-input"
                  name="rtsp"
                  placeholder="rtsp://user:pass@192.168.1.100:554/stream1"
                  required
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">RTSP Username</label>
                  <input className="form-input" name="user" placeholder="admin" />
                </div>
                <div className="form-group">
                  <label className="form-label">RTSP Password</label>
                  <input className="form-input" name="pass" type="password" placeholder="••••••••" />
                </div>
              </div>
              <div className="form-actions">
                <button type="button" className="btn-cancel" onClick={closeModal}>Cancel</button>
                <button type="submit" className="btn-primary" disabled={loading}>
                  {loading ? 'Adding…' : 'Add Camera'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ═══ MODAL: RENAME CAMERA ═══ */}
      {modal === 'edit-camera' && editCam && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>✏ Rename Camera</h2>
              <button type="button" className="modal-close" onClick={closeModal}>✕</button>
            </div>
            <form onSubmit={handleEditCamera}>
              <div className="form-group">
                <label className="form-label">Camera ID</label>
                <input className="form-input" value={editCam.id} disabled style={{ color: 'var(--muted)', fontSize: '0.76rem' }} />
              </div>
              <div className="form-group">
                <label className="form-label">New Name</label>
                <input className="form-input" name="name" defaultValue={editCam.name} required autoFocus />
              </div>
              <div className="form-actions">
                <button type="button" className="btn-cancel" onClick={closeModal}>Cancel</button>
                <button type="submit" className="btn-primary" disabled={loading}>
                  {loading ? 'Saving…' : 'Save'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ═══ MODAL: CONFIRM RESET ═══ */}
      {modal === 'confirm-reset' && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>⚠ Factory Reset</h2>
              <button type="button" className="modal-close" onClick={closeModal}>✕</button>
            </div>
            <p style={{ color: 'var(--subtext)', fontSize: '0.88rem', lineHeight: 1.65, marginBottom: 24 }}>
              This will permanently delete <strong style={{ color: 'var(--text)' }}>all cameras, events, and settings</strong>.
              This cannot be undone.
            </p>
            <div className="form-actions">
              <button type="button" className="btn-cancel" onClick={closeModal}>Cancel</button>
              <button type="button" className="btn-danger-solid" onClick={handleFactoryReset} disabled={loading}>
                {loading ? 'Resetting…' : 'Yes, Delete Everything'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ═══ MODAL: SETTINGS (Password / ONVIF / AI Ref) ═══ */}
      {modal === 'settings' && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal modal-lg" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>⚙ Settings</h2>
              <button type="button" className="modal-close" onClick={closeModal}>✕</button>
            </div>

            <div className="settings-tabs">
              <button className={`settings-tab ${settingsTab === 'password' ? 'active' : ''}`} onClick={() => setSettingsTab('password')}>
                🔑 Change Password
              </button>
              <button className={`settings-tab ${settingsTab === 'onvif' ? 'active' : ''}`} onClick={() => setSettingsTab('onvif')}>
                📡 ONVIF
              </button>
              <button className={`settings-tab ${settingsTab === 'ai-ref' ? 'active' : ''}`} onClick={() => setSettingsTab('ai-ref')}>
                🤖 AI Reference
              </button>
            </div>

            {/* ── PASSWORD TAB ── */}
            {settingsTab === 'password' && (
              <form onSubmit={handleChangePassword}>
                <p className="settings-desc">Change the login password for any user account.</p>
                <div className="form-group">
                  <label className="form-label">Username</label>
                  <input className="form-input" name="username" defaultValue="admin" placeholder="admin" autoFocus />
                </div>
                <div className="form-group">
                  <label className="form-label">Current Password</label>
                  <input className="form-input" name="current_pass" type="password" placeholder="Current password" required />
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label className="form-label">New Password</label>
                    <input className="form-input" name="new_pass" type="password" placeholder="New password" required />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Confirm Password</label>
                    <input className="form-input" name="confirm" type="password" placeholder="Repeat new password" required />
                  </div>
                </div>
                <div className="form-actions">
                  <button type="button" className="btn-cancel" onClick={closeModal}>Cancel</button>
                  <button type="submit" className="btn-primary" disabled={loading}>
                    {loading ? 'Changing…' : 'Change Password'}
                  </button>
                </div>
              </form>
            )}

            {/* ── ONVIF TAB ── */}
            {settingsTab === 'onvif' && (
              <form onSubmit={handleSaveOnvif}>
                <p className="settings-desc">
                  Default ONVIF credentials used when auto-discovering cameras on the network.
                  These are saved to the local database and used on next discovery scan.
                </p>
                <div className="form-row">
                  <div className="form-group">
                    <label className="form-label">ONVIF Username</label>
                    <input className="form-input" name="onvif_user" placeholder="admin" autoFocus />
                  </div>
                  <div className="form-group">
                    <label className="form-label">ONVIF Password</label>
                    <input className="form-input" name="onvif_pass" type="password" placeholder="••••••••" />
                  </div>
                </div>
                <div className="form-actions">
                  <button type="button" className="btn-cancel" onClick={closeModal}>Cancel</button>
                  <button type="submit" className="btn-primary" disabled={loading}>
                    {loading ? 'Saving…' : 'Save ONVIF Settings'}
                  </button>
                </div>
              </form>
            )}

            {/* ── AI REFERENCE TAB ── */}
            {settingsTab === 'ai-ref' && (
              <form onSubmit={handleRegisterAiRef}>
                <p className="settings-desc">
                  Register a known face photo so the AI can identify and alert on that person.
                  Use a clear, front-facing photo (PNG/JPG/WEBP).
                </p>
                <div className="form-group">
                  <label className="form-label">Person Name / Label *</label>
                  <input className="form-input" name="tag" placeholder="e.g. John Doe" required autoFocus />
                </div>
                <div className="form-group">
                  <label className="form-label">Reference Image *</label>
                  <div className="file-pick-row">
                    <input
                      className="form-input"
                      placeholder="No image selected…"
                      value={aiFilePath ? aiFilePath.split(/[\\/]/).pop()! : ''}
                      readOnly
                    />
                    <button type="button" className="pick-btn" onClick={handlePickAiFile}>
                      📁 Browse…
                    </button>
                  </div>
                </div>
                {aiFilePath && (
                  <div className="ai-preview-row">
                    <img src={`data:image/*;base64,`} alt="" style={{ display: 'none' }} />
                    <span className="ai-file-label">✓ {aiFilePath.split(/[\\/]/).pop()}</span>
                  </div>
                )}
                <div className="form-actions">
                  <button type="button" className="btn-cancel" onClick={closeModal}>Cancel</button>
                  <button type="submit" className="btn-primary" disabled={loading || !aiFilePath}>
                    {loading ? 'Registering…' : 'Register Face'}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </>
  );
}
