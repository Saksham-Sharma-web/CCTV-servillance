import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from './AuthProvider';
import type { Permission } from './permissions';
import { Unauthorized } from '../components/auth/Unauthorized';
import { AuthLoading } from '../components/auth/AuthLoading';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermission?: Permission;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  requiredPermission 
}) => {
  const { isAuthenticated, loading, hasPermission, profile } = useAuth();
  const location = useLocation();

  if (loading) {
    return <AuthLoading />;
  }

  if (!isAuthenticated) {
    // Redirect them to the /login page, but save the current location they were
    // trying to go to when they were redirected. This allows us to send them
    // along to that page after they login, which is a nicer user experience
    // than dropping them off on the home page.
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  
  if (!profile) {
      // Authenticated but no profile found in public.profiles
      return (
        <div className="min-h-screen bg-background flex flex-col justify-center items-center p-4 text-center">
           <h2 className="text-xl font-bold text-status-critical mb-2">Profile Missing</h2>
           <p className="text-text-secondary">Your account is authenticated but has no authorized operator profile. Contact the system administrator.</p>
        </div>
      );
  }

  if (requiredPermission && !hasPermission(requiredPermission)) {
    return <Unauthorized />;
  }

  return <>{children}</>;
};
