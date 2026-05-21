import React from 'react';
import { X, AlertTriangle } from 'lucide-react';
import { Button } from './Button';

/**
 * 기능 안내 등을 위한 Modal 컴포넌트
 */
export const Modal = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/40 z-50 flex items-center justify-center backdrop-blur-sm px-4 modal-overlay">
      <div className="bg-white rounded-3xl p-8 max-w-sm w-full shadow-2xl relative fade-in">
        <button onClick={onClose} className="absolute top-4 right-4 text-slate-400 hover:text-slate-700">
          <X className="w-6 h-6" />
        </button>
        <div className="w-16 h-16 bg-blue-50 rounded-2xl flex items-center justify-center mx-auto mb-6 text-primary">
          <AlertTriangle className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-center text-slate-800 mb-2">{title}</h3>
        <div className="text-center text-slate-500 font-medium mb-8">
          {children}
        </div>
        <Button variant="dark" className="w-full" onClick={onClose}>
          확인
        </Button>
      </div>
    </div>
  );
};
