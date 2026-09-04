import React from 'react';
import { ShieldCheck } from 'lucide-react';

export const AuthLoading: React.FC = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col justify-center items-center py-12 sm:px-6 lg:px-8">
      <div className="animate-pulse flex flex-col items-center">
        <ShieldCheck className="h-16 w-16 text-steel-light mb-6 opacity-75" />
        <h2 className="text-xl font-bold text-text-primary uppercase tracking-widest">
          Initializing Secure Session...
        </h2>
        <div className="mt-6 flex space-x-2">
          <div className="w-2 h-2 bg-steel-light rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-steel-light rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-steel-light rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
};
