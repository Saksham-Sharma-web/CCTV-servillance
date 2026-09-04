import React, { useState, useEffect, useCallback } from 'react';
import { Users, Shield, Plus, Search, MoreVertical, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { useAuth } from '../auth/AuthProvider';
import { supabase } from '../lib/supabase';
import { UserModal } from '../components/users/UserModal';

interface User {
  id: string;
  full_name: string;
  email: string;
  role: string;
  department: string;
  status: 'ACTIVE' | 'DISABLED';
  last_sign_in_at?: string;
  created_at: string;
}

export const UserManagement: React.FC = () => {
  const { session, user: currentUser } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUsers = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`/api/admin/users?search=${encodeURIComponent(searchTerm)}`, {
        headers: {
          'Authorization': `Bearer ${session?.access_token}`
        }
      });
      const result = await res.json();
      if (!result.success) throw new Error(result.error?.message || 'Failed to fetch users');
      setUsers(result.data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [session, searchTerm]);

  useEffect(() => {
    fetchUsers();

    // Subscribe to realtime profile changes
    const subscription = supabase
      .channel('public:profiles')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'profiles' }, () => {
        // Optimistically refresh the entire list to keep auth/profile sync easy
        fetchUsers();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(subscription);
    };
  }, [fetchUsers]);

  const handleAction = async (action: string, id: string, payload?: any) => {
    if (!confirm(`Are you sure you want to perform this action?`)) return;
    
    try {
      let method = 'POST';
      let url = `/api/admin/users/${id}/${action}`;
      let body = undefined;

      if (action === 'delete') {
        method = 'DELETE';
        url = `/api/admin/users/${id}`;
      } else if (action === 'role') {
        method = 'PATCH';
        body = JSON.stringify({ role: payload });
      }

      const res = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session?.access_token}`
        },
        body
      });
      
      const result = await res.json();
      if (!result.success) throw new Error(result.error?.message);
      
      fetchUsers();
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const activeCount = users.filter(u => u.status === 'ACTIVE').length;
  const operatorCount = users.filter(u => u.role === 'operator').length;
  const adminCount = users.filter(u => u.role === 'admin').length;

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Stats Header */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-surface border border-border p-4 rounded-lg flex flex-col justify-between">
          <span className="text-xs text-text-secondary font-bold uppercase tracking-wider">Total Users</span>
          <span className="text-2xl font-bold text-text-primary">{users.length}</span>
        </div>
        <div className="bg-surface border border-border p-4 rounded-lg flex flex-col justify-between">
          <span className="text-xs text-text-secondary font-bold uppercase tracking-wider">Active</span>
          <span className="text-2xl font-bold text-status-success">{activeCount}</span>
        </div>
        <div className="bg-surface border border-border p-4 rounded-lg flex flex-col justify-between">
          <span className="text-xs text-text-secondary font-bold uppercase tracking-wider">Operators</span>
          <span className="text-2xl font-bold text-steel-light">{operatorCount}</span>
        </div>
        <div className="bg-surface border border-border p-4 rounded-lg flex flex-col justify-between">
          <span className="text-xs text-text-secondary font-bold uppercase tracking-wider">Admins</span>
          <span className="text-2xl font-bold text-status-critical">{adminCount}</span>
        </div>
      </div>

      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <Users className="w-5 h-5 mr-2 text-steel" /> Operator & Access Management
        </h2>
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-gray-400" />
            <input 
              type="text" 
              placeholder="Search users..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9 pr-3 py-1.5 border border-border rounded-md text-sm bg-background focus:ring-1 focus:ring-steel"
            />
          </div>
          <button 
            onClick={() => setIsAddModalOpen(true)}
            className="btn btn-primary text-xs flex items-center bg-steel hover:bg-steel-light text-white px-3 py-2 rounded uppercase font-bold tracking-wider"
          >
            <Plus className="w-4 h-4 mr-1" /> Add Operator
          </button>
        </div>
      </div>

      <div className="flex-1 card overflow-hidden flex flex-col relative">
        {loading && users.length === 0 && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface/50 backdrop-blur-sm">
            <span className="font-bold text-steel animate-pulse uppercase tracking-widest">Loading Records...</span>
          </div>
        )}
        
        {error && (
          <div className="p-4 bg-status-critical/10 border-b border-status-critical/30 text-status-critical text-sm font-semibold">
            {error}
          </div>
        )}

        <div className="overflow-x-auto flex-1 custom-scrollbar p-0">
          <table className="w-full text-left">
            <thead className="bg-background sticky top-0 border-b border-border shadow-sm z-10">
              <tr>
                <th className="px-4 py-3 text-xs font-bold text-text-secondary uppercase tracking-wider">Operator</th>
                <th className="px-4 py-3 text-xs font-bold text-text-secondary uppercase tracking-wider">Email</th>
                <th className="px-4 py-3 text-xs font-bold text-text-secondary uppercase tracking-wider">Role</th>
                <th className="px-4 py-3 text-xs font-bold text-text-secondary uppercase tracking-wider">Status</th>
                <th className="px-4 py-3 text-xs font-bold text-text-secondary uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {users.map((user) => (
                <tr key={user.id} className="hover:bg-gray-50/50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="font-semibold text-text-primary text-sm">{user.full_name || 'N/A'}</div>
                    <div className="text-xs text-text-secondary">{user.department || 'Unassigned'}</div>
                  </td>
                  <td className="px-4 py-3 text-sm text-text-secondary">{user.email}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-[10px] font-bold flex items-center w-fit uppercase tracking-wider ${
                      user.role === 'admin' ? 'bg-status-critical/20 text-status-critical' : 
                      user.role === 'supervisor' ? 'bg-steel/20 text-steel' : 'bg-gray-200 text-gray-700'
                    }`}>
                      <Shield className="w-3 h-3 mr-1" /> {user.role}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${
                      user.status === 'ACTIVE' ? 'bg-status-success/20 text-status-success' : 'bg-status-warning/20 text-status-warning'
                    }`}>
                      {user.status === 'ACTIVE' ? <CheckCircle2 className="inline w-3 h-3 mr-1"/> : <ShieldAlert className="inline w-3 h-3 mr-1"/>}
                      {user.status}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <select 
                        value={user.role} 
                        onChange={(e) => handleAction('role', user.id, e.target.value)}
                        disabled={user.id === currentUser?.id}
                        className="text-xs border border-border rounded bg-background px-2 py-1"
                      >
                        <option value="operator">Operator</option>
                        <option value="supervisor">Supervisor</option>
                        <option value="admin">Admin</option>
                      </select>
                      
                      {user.status === 'ACTIVE' ? (
                        <button onClick={() => handleAction('disable', user.id)} disabled={user.id === currentUser?.id} className="text-xs font-semibold text-status-warning hover:underline disabled:opacity-50">Disable</button>
                      ) : (
                        <button onClick={() => handleAction('enable', user.id)} className="text-xs font-semibold text-status-success hover:underline">Enable</button>
                      )}
                      
                      <button onClick={() => handleAction('delete', user.id)} disabled={user.id === currentUser?.id} className="text-xs font-semibold text-status-critical hover:underline disabled:opacity-50">Delete</button>
                    </div>
                  </td>
                </tr>
              ))}
              {users.length === 0 && !loading && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-text-secondary">No operators found.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <UserModal 
        isOpen={isAddModalOpen} 
        onClose={() => setIsAddModalOpen(false)} 
        onSuccess={() => fetchUsers()} 
      />
    </div>
  );
};
