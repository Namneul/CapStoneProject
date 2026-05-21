import React from 'react';
import { Home, Video, BarChart2, Settings } from 'lucide-react';

export const Sidebar = () => {
  const menuItems = [
    { icon: Home, label: '홈', active: true },
    { icon: Video, label: '면접 연습', active: false },
    { icon: BarChart2, label: '통계 리포트', active: false },
    { icon: Settings, label: '설정', active: false },
  ];

  return (
    <aside className="w-64 bg-white border-r border-slate-100 h-screen hidden lg:flex flex-col p-6 sticky top-0">
      <div className="font-black text-2xl text-primary mb-10 tracking-tight">G.T.A</div>
      <nav className="flex-1 space-y-2">
        {menuItems.map((item, idx) => (
          <button 
            key={idx}
            className={`w-full flex items-center px-4 py-3 rounded-2xl font-bold transition-colors ${
              item.active 
                ? 'bg-blue-50 text-primary' 
                : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800'
            }`}
          >
            <item.icon className="w-5 h-5 mr-3" />
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  );
};
