import React from 'react';

/**
 * 범용 Button 컴포넌트
 * primary: 메인 액션 (파란색, floating 그림자)
 * outline: 서브 액션 (테두리)
 * ghost: 텍스트 형태 액션
 * dark: 완료 등 강조 액션
 */
export const Button = ({ children, variant = 'primary', className = "", ...props }) => {
  const baseStyle = "font-bold rounded-2xl transition-all active:scale-[0.98] flex items-center justify-center";
  const variants = {
    primary: "bg-primary hover:bg-blue-700 text-white shadow-floating px-8 py-4",
    outline: "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 shadow-sm px-8 py-4",
    ghost: "text-slate-400 hover:text-slate-800 hover:bg-slate-50 px-4 py-2",
    dark: "bg-slate-900 hover:bg-black text-white shadow-xl px-10 py-4"
  };

  return (
    <button className={`${baseStyle} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
};
