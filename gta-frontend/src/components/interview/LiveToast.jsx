import React from 'react';
import { AlertTriangle } from 'lucide-react';

/**
 * 전역으로 표시되는 라이브 경고 토스트
 */
export const LiveToast = ({ warnings }) => {
  return (
    <div className="fixed top-8 left-1/2 z-50 flex flex-col items-center pointer-events-none gap-2">
      {warnings.map((toast) => (
        <div key={toast.id} className="slide-down flex items-center bg-warning text-white px-6 py-4 rounded-2xl shadow-xl border border-orange-400/50">
          <AlertTriangle className="w-5 h-5 mr-3" />
          <span className="font-bold text-sm tracking-wide">제스처 경고: {toast.message}</span>
        </div>
      ))}
    </div>
  );
};
