import { Request, Response } from 'express';
import { supabaseAdmin } from '../lib/supabaseAdmin';

const logAudit = async (adminId: string, action: string, entityId: string | null, details: any = {}) => {
  try {
    await supabaseAdmin.from('audit_logs').insert({
      user_id: adminId,
      action,
      entity_type: 'user',
      entity_id: entityId,
      details,
    });
  } catch (error) {
    console.error('Audit log failed (non-fatal):', error);
  }
};

export const userController = {
  // GET /api/admin/users
  async getUsers(req: Request, res: Response) {
    try {
      const { search, role, status } = req.query;

      let query = supabaseAdmin
        .from('profiles')
        .select(`
          id,
          full_name,
          role,
          department,
          created_at
        `);

      if (role) query = query.eq('role', role);

      const { data: profiles, error: profileError } = await query;
      
      if (profileError) throw profileError;

      // We also need auth status (banned, email, etc.)
      const { data: { users }, error: usersError } = await supabaseAdmin.auth.admin.listUsers();
      
      if (usersError) throw usersError;

      // Merge data
      let merged = profiles.map(p => {
        const authUser = users.find(u => u.id === p.id);
        return {
          ...p,
          email: authUser?.email,
          status: authUser?.banned_until ? 'DISABLED' : 'ACTIVE',
          last_sign_in_at: authUser?.last_sign_in_at,
        };
      });

      if (search && typeof search === 'string') {
        const s = search.toLowerCase();
        merged = merged.filter(u => 
          u.full_name?.toLowerCase().includes(s) || 
          u.email?.toLowerCase().includes(s) || 
          u.department?.toLowerCase().includes(s)
        );
      }

      if (status) {
        merged = merged.filter(u => u.status === status);
      }

      return res.json({ success: true, data: merged });
    } catch (error: any) {
      console.error('Error fetching users:', error);
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  },

  // POST /api/admin/users
  async createUser(req: Request, res: Response) {
    const { email, password, full_name, role = 'operator', department } = req.body;

    if (!email || !password || !full_name) {
      return res.status(400).json({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Email, password, and full name are required.' } });
    }

    if (!['operator', 'supervisor', 'admin'].includes(role)) {
      return res.status(400).json({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Invalid role.' } });
    }

    try {
      // 1. Create auth user
      const { data: authData, error: authError } = await supabaseAdmin.auth.admin.createUser({
        email,
        password,
        email_confirm: true,
      });

      if (authError) {
        return res.status(400).json({ success: false, error: { code: 'AUTH_ERROR', message: authError.message } });
      }

      const userId = authData.user.id;

      // 2. Create profile
      const { error: profileError } = await supabaseAdmin
        .from('profiles')
        .insert({
          id: userId,
          full_name,
          role,
          department,
        });

      if (profileError) {
        // Rollback: delete auth user if profile creation fails
        await supabaseAdmin.auth.admin.deleteUser(userId);
        return res.status(400).json({ success: false, error: { code: 'PROFILE_ERROR', message: 'Failed to create profile.' } });
      }

      // 3. Log Audit
      await logAudit(req.user.id, 'USER_CREATED', userId, { target_email: email, role });

      return res.json({ success: true, data: { id: userId, email, full_name, role, department } });
    } catch (error: any) {
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  },

  // PATCH /api/admin/users/:id/role
  async updateRole(req: Request, res: Response) {
    const { id } = req.params;
    const { role } = req.body;

    if (id === req.user.id) {
      return res.status(403).json({ success: false, error: { code: 'FORBIDDEN', message: 'Your administrator role cannot be removed from your current account.' } });
    }

    if (!['operator', 'supervisor', 'admin'].includes(role)) {
      return res.status(400).json({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Invalid role.' } });
    }

    try {
      // Get old role for logging
      const { data: oldProfile } = await supabaseAdmin.from('profiles').select('role').eq('id', id).single();

      const { data, error } = await supabaseAdmin
        .from('profiles')
        .update({ role })
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;

      await logAudit(req.user.id as string, 'ROLE_CHANGED', id as string, { old_role: oldProfile?.role, new_role: role });

      return res.json({ success: true, data });
    } catch (error: any) {
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  },

  // PATCH /api/admin/users/:id
  async updateUser(req: Request, res: Response) {
    const { id } = req.params;
    const { full_name, department } = req.body;

    const updates: any = {};
    if (full_name !== undefined) updates.full_name = full_name;
    if (department !== undefined) updates.department = department;

    try {
      const { data, error } = await supabaseAdmin
        .from('profiles')
        .update(updates)
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;

      await logAudit(req.user.id as string, 'USER_UPDATED', id as string, updates);

      return res.json({ success: true, data });
    } catch (error: any) {
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  },

  // POST /api/admin/users/:id/disable
  async disableUser(req: Request, res: Response) {
    const { id } = req.params;

    if (id === req.user.id) {
      return res.status(403).json({ success: false, error: { code: 'FORBIDDEN', message: 'You cannot disable your own administrator account.' } });
    }

    try {
      // Ban for 100 years
      const { data, error } = await supabaseAdmin.auth.admin.updateUserById(id as string, { ban_duration: '876000h' });
      if (error) throw error;

      await logAudit(req.user.id as string, 'USER_DISABLED', id as string);
      return res.json({ success: true, data: { status: 'DISABLED' } });
    } catch (error: any) {
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  },

  // POST /api/admin/users/:id/enable
  async enableUser(req: Request, res: Response) {
    const { id } = req.params;

    try {
      const { data, error } = await supabaseAdmin.auth.admin.updateUserById(id as string, { ban_duration: 'none' });
      if (error) throw error;

      await logAudit(req.user.id as string, 'USER_ENABLED', id as string);
      return res.json({ success: true, data: { status: 'ACTIVE' } });
    } catch (error: any) {
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  },

  // DELETE /api/admin/users/:id
  async deleteUser(req: Request, res: Response) {
    const { id } = req.params;

    if (id === req.user.id) {
      return res.status(403).json({ success: false, error: { code: 'FORBIDDEN', message: 'You cannot delete your own administrator account.' } });
    }

    try {
      const { error } = await supabaseAdmin.auth.admin.deleteUser(id as string);
      if (error) throw error;

      await logAudit(req.user.id as string, 'USER_DELETED', id as string);
      return res.json({ success: true });
    } catch (error: any) {
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  },

  // POST /api/admin/users/:id/password-reset
  async resetPassword(req: Request, res: Response) {
    const { id } = req.params;

    try {
      const { data: user, error: fetchError } = await supabaseAdmin.auth.admin.getUserById(id as string);
      if (fetchError || !user.user.email) throw fetchError || new Error('User email not found');

      const { error } = await supabaseAdmin.auth.resetPasswordForEmail(user.user.email);
      if (error) throw error;

      await logAudit(req.user.id as string, 'PASSWORD_RESET_REQUESTED', id as string);
      return res.json({ success: true });
    } catch (error: any) {
      return res.status(500).json({ success: false, error: { code: 'SERVER_ERROR', message: error.message } });
    }
  }
};
