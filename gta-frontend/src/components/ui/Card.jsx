import React from 'react';

/**
 * 범용 Card 컴포넌트
 * 라운딩 처리와 부드러운 그림자를 통해 SaaS 룩앤필 제공
 */
export const Card = ({ children, className = "" }) => {
  return (
    <div className={`bg-white rounded-3xl shadow-soft border border-slate-100 ${className}`}>
      {children}
    </div>
  );
};
