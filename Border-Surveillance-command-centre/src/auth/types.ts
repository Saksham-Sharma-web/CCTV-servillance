import type { User, Session } from '@supabase/supabase-js';

export type Role = 'operator' | 'supervisor' | 'admin';

export interface Profile {
  id: string;
  full_name: string | null;
  role: Role;
  department: string | null;
  created_at: string;
}

export interface AuthContextType {
  user: User | null;
  profile: Profile | null;
  session: Session | null;
  loading: boolean;
  isAuthenticated: boolean;
  hasPermission: (permission: string) => boolean;
  signIn: () => void;
  signOut: () => Promise<void>;
}
