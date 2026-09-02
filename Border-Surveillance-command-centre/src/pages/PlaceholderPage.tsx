import React from 'react';

export const PlaceholderPage: React.FC<{ title: string }> = ({ title }) => {
  return (
    <div className="flex flex-col h-full w-full items-center justify-center text-text-secondary opacity-50">
      <h2 className="text-2xl font-bold mb-2 uppercase tracking-widest">{title}</h2>
      <p className="text-sm">This module is currently under development.</p>
    </div>
  );
};
