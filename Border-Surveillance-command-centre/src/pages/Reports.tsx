import React from 'react';
import { FileBarChart, Download } from 'lucide-react';

export const Reports: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="card p-3 flex items-center justify-between">
        <h2 className="text-lg font-bold text-navy-dark flex items-center">
          <FileBarChart className="w-5 h-5 mr-2 text-steel" /> Operational Reports
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {['Daily Shift Summary', 'Incident Analysis (Weekly)', 'Infrastructure Health Report', 'AI Detection Accuracy', 'Border Crossing Stats'].map((report, i) => (
          <div key={i} className="card p-4 flex flex-col items-center justify-center text-center group hover:border-navy transition-colors cursor-pointer">
            <FileBarChart className="w-8 h-8 text-steel-light mb-3 group-hover:text-navy transition-colors" />
            <h3 className="font-semibold text-navy-dark mb-1">{report}</h3>
            <p className="text-xs text-text-secondary mb-4">Generate automated PDF/CSV report</p>
            <button className="btn btn-secondary w-full text-xs">
              <Download className="w-3 h-3 mr-2" /> Generate
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
