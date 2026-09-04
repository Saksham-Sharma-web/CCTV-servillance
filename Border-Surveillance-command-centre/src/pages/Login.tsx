import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { Shield, Eye, EyeOff, Lock, User, AlertCircle } from 'lucide-react';

export const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const navigate = useNavigate();
  const location = useLocation();

  // Redirect to the page they tried to visit before logging in, or default to home
  const from = location.state?.from?.pathname || '/';

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        if (error.message.includes('Invalid login credentials')) {
          setError('Invalid email or password.');
        } else if (error.message.includes('Failed to fetch') || error.message.includes('Network Error')) {
          setError('Unable to connect to the authentication service. Please try again.');
        } else {
          setError('An unexpected error occurred during authentication.');
        }
      } else {
        // AuthProvider will handle the redirect if signed in, but we can also push directly
        navigate(from, { replace: true });
      }
    } catch (err) {
      setError('An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col justify-center py-12 sm:px-6 lg:px-8 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-steel-light/10 via-background to-background">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
        <Shield className="mx-auto h-16 w-16 text-steel" />
        <h2 className="mt-6 text-3xl font-bold text-text-primary uppercase tracking-widest">
          Intelligent Border Surveillance
        </h2>
        <p className="mt-2 text-sm text-text-secondary uppercase tracking-widest font-medium">
          AI-Powered Video Analytics & Command Center
        </p>
      </div>

      <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-surface border border-border py-8 px-4 shadow-[0_0_15px_rgba(0,0,0,0.5)] sm:rounded-lg sm:px-10 relative overflow-hidden">
          {/* Subtle top border highlight */}
          <div className="absolute top-0 left-0 w-full h-1 bg-steel"></div>
          
          <h3 className="text-xl font-semibold text-text-primary uppercase tracking-wider mb-6 text-center border-b border-border pb-4">
            Secure Operator Login
          </h3>

          <form className="space-y-6" onSubmit={handleLogin}>
            {error && (
              <div className="bg-status-critical/10 border border-status-critical/50 rounded-md p-3 flex items-start">
                <AlertCircle className="h-5 w-5 text-status-critical mr-2 mt-0.5 flex-shrink-0" />
                <span className="text-sm text-status-critical font-medium">{error}</span>
              </div>
            )}

            <div>
              <label htmlFor="email" className="block text-xs font-semibold text-text-secondary uppercase tracking-wider mb-1">
                Official Email
              </label>
              <div className="mt-1 relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-500" />
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-border rounded-md leading-5 bg-background text-text-primary placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-steel focus:border-steel sm:text-sm transition-colors"
                  placeholder="operator@command.gov"
                  disabled={loading}
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-xs font-semibold text-text-secondary uppercase tracking-wider mb-1">
                Password
              </label>
              <div className="mt-1 relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-500" />
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-10 py-2.5 border border-border rounded-md leading-5 bg-background text-text-primary placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-steel focus:border-steel sm:text-sm transition-colors"
                  placeholder="••••••••••••"
                  disabled={loading}
                />
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="text-gray-500 hover:text-gray-300 focus:outline-none"
                    disabled={loading}
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5" />
                    ) : (
                      <Eye className="h-5 w-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={loading}
                className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-bold uppercase tracking-wider text-white bg-steel hover:bg-steel-light focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-steel focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? 'Authenticating...' : 'Sign In'}
              </button>
            </div>
          </form>
          
          <div className="mt-8 pt-4 border-t border-border flex flex-col items-center">
            <span className="text-[10px] text-text-secondary uppercase tracking-widest font-semibold mb-1">
              Secure Authentication
            </span>
            <span className="text-[10px] text-status-critical uppercase tracking-widest font-bold">
              Authorized Personnel Only
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
