import React from 'react';
import { MessageSquare } from 'lucide-react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';

export const HistoryList = ({ records }) => {
  return (
    <Card className="p-8">
      <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center">
        <MessageSquare className="w-5 h-5 mr-2 text-primary" />
        질문별 답변 기록
      </h3>
      <div className="space-y-4">
        {records.map((item) => (
          <div key={item.id} className="flex flex-col sm:flex-row sm:items-center justify-between p-5 rounded-2xl bg-background border border-slate-200 hover:border-primary/50 transition-all group">
            <div className="flex items-start mb-3 sm:mb-0">
              <div className="w-8 h-8 rounded-lg bg-blue-50 text-primary font-bold flex items-center justify-center flex-shrink-0 mr-4 mt-0.5">
                Q{item.id}
              </div>
              <div>
                <div className="font-bold text-slate-800 text-sm leading-relaxed mb-1">{item.q}</div>
                <div className="text-xs font-semibold text-slate-400">소요 시간: {item.time}</div>
              </div>
            </div>
            <div className="flex items-center self-start sm:self-auto ml-12 sm:ml-0">
              <Badge variant={item.gesture > 0 ? 'warning' : 'default'}>
                경고 {item.gesture}회
              </Badge>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};
