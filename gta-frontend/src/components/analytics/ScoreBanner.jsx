import React from 'react';
import { TrendingUp } from 'lucide-react';

export const ScoreBanner = ({ score }) => {
  return (
    <div className="score-gradient rounded-3xl p-10 text-center text-white shadow-floating relative overflow-hidden flex flex-col justify-center min-h-[300px]">
      <div className="relative z-10">
        <div className="text-blue-100 font-bold mb-4 text-sm tracking-wide">종합 평가 점수</div>
        <div className="text-[6rem] font-black mb-6 tracking-tighter leading-none">{score}</div>
        <div className="inline-flex items-center bg-white/20 px-5 py-2.5 rounded-full text-sm font-bold backdrop-blur-md border border-white/20">
          <TrendingUp className="w-4 h-4 mr-2" />
          우수한 성과입니다!
        </div>
      </div>
    </div>
  );
};
