import React from 'react';
import { ShieldAlert } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const Unauthorized: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
        <ShieldAlert className="mx-auto h-16 w-16 text-status-critical" />
        <h2 className="mt-6 text-3xl font-extrabold text-text-primary uppercase tracking-wider">
          Access Restricted
        </h2>
        <p className="mt-2 text-sm text-text-secondary">
          Your account does not have permission to access this section.
        </p>
        
        <div className="mt-8">
          <button
            onClick={() => navigate('/')}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-steel hover:bg-steel-light focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-steel transition-colors"
          >
            Return to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};
