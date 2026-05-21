import React from 'react';
import { Sparkles } from 'lucide-react';

export const Header = () => {
  return (
    <header className="w-full bg-white border-b border-slate-100 py-4 px-8 flex items-center justify-between sticky top-0 z-40 shadow-sm">
      <div className="flex items-center gap-2">
        <Sparkles className="w-6 h-6 text-primary" />
        <span className="font-black text-xl text-slate-800 tracking-tight">G.T.A</span>
      </div>
      <div className="text-sm font-bold text-slate-500">
        멀티에이전트 표현 습관 분석 시스템
      </div>
    </header>
  );
};
