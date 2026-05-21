import React from 'react';

/**
 * 상태 표시를 위한 Badge 컴포넌트
 */
export const Badge = ({ children, variant = 'warning', className = "" }) => {
  const variants = {
    warning: "bg-orange-50 text-warning border border-orange-100",
    default: "bg-slate-100 text-slate-500 border border-slate-200",
    primary: "bg-blue-50 text-primary border border-blue-100",
  };

  return (
    <span className={`px-3 py-1.5 rounded-xl text-xs font-bold inline-flex items-center ${variants[variant]} ${className}`}>
      {children}
    </span>
  );
};
