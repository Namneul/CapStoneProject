import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Clock, Activity, MessageSquare, Hand, Home, RotateCcw } from 'lucide-react';
import { useInterview } from '../context/InterviewContext';
import { Layout } from '../components/layout/Layout';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { ScoreBanner } from '../components/analytics/ScoreBanner';
import { HistoryList } from '../components/analytics/HistoryList';

export const ResultDashboardPage = () => {
  const navigate = useNavigate();
  const { sessionData } = useInterview();

  if (sessionData.score === 0) {
    navigate('/');
    return null;
  }

  const rd = sessionData.resultData || {};
  const verbal = rd.verbal || {};
  const nonverbal = rd.nonverbal || {};

  const formatTime = (s) => {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}분 ${secs}초`;
  };

  const questionRecords = [
    { 
      id: 1, 
      q: rd.question || "알 수 없는 질문", 
      time: formatTime(verbal.duration || sessionData.totalTime), 
      gesture: nonverbal.clusters?.length || sessionData.totalGestures 
    }
  ];

  const statItems = [
    { icon: Clock, label: '답변 시간', value: formatTime(verbal.duration || sessionData.totalTime), color: 'text-primary', bg: 'bg-blue-50' },
    { icon: Activity, label: '말 속도(단어/초)', value: verbal.speech_rate || '0', color: 'text-violet-500', bg: 'bg-violet-50' },
    { icon: MessageSquare, label: '추임새 횟수', value: `${verbal.filler_count || 0}회`, color: 'text-green-500', bg: 'bg-green-50' },
    { icon: Hand, label: '자세 패턴 수', value: `${nonverbal.clusters?.length || 0}개`, color: 'text-warning', bg: 'bg-orange-50' }
  ];

  return (
    <Layout withHeader>
      <div className="fade-in w-full max-w-5xl py-8 flex flex-col gap-6">
        <div className="text-center mb-4">
          <h2 className="text-2xl font-bold text-slate-800 mb-2">면접 분석 리포트</h2>
          <p className="text-slate-500 font-medium text-sm">G.T.A 에이전트가 분석한 상세 피드백입니다.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <ScoreBanner score={sessionData.score} />
          </div>

          <div className="lg:col-span-2 grid grid-cols-2 gap-4">
            {statItems.map((stat, i) => (
              <Card key={i} className="p-6 border border-slate-100 flex flex-col justify-center items-center text-center">
                <div className={`w-14 h-14 rounded-2xl ${stat.bg} ${stat.color} flex items-center justify-center mb-4`}>
                  <stat.icon className="w-6 h-6" />
                </div>
                <div className="text-2xl font-black text-slate-800 mb-1">{stat.value}</div>
                <div className="text-sm text-slate-500 font-bold">{stat.label}</div>
              </Card>
            ))}
          </div>
        </div>

        {/* 질문 내역 */}
        <HistoryList records={questionRecords} />

        {/* 상세 피드백 영역 (백엔드에서 받은 피드백) */}
        {rd.content_evaluation && (
          <Card className="p-6 border border-slate-100">
            <h3 className="text-xl font-bold text-slate-800 mb-4">답변 내용 평가</h3>
            <div className="text-slate-600 leading-relaxed whitespace-pre-wrap bg-slate-50 p-4 rounded-xl">
              {rd.content_evaluation}
            </div>
          </Card>
        )}

        {rd.delivery_feedback && (
          <Card className="p-6 border border-slate-100">
            <h3 className="text-xl font-bold text-slate-800 mb-4">전달 방식 평가</h3>
            <div className="text-slate-600 leading-relaxed whitespace-pre-wrap bg-slate-50 p-4 rounded-xl">
              {rd.delivery_feedback}
            </div>
          </Card>
        )}
        
        {rd.improved_answer && (
          <Card className="p-6 border border-slate-100">
            <h3 className="text-xl font-bold text-slate-800 mb-4">개선된 답변 예시</h3>
            <div className="text-slate-600 leading-relaxed whitespace-pre-wrap bg-slate-50 p-4 rounded-xl">
              {rd.improved_answer}
            </div>
          </Card>
        )}
        
        {rd.followup && (
          <Card className="p-6 border border-slate-100 border-l-4 border-l-primary">
            <h3 className="text-lg font-bold text-slate-800 mb-2 flex items-center">
              <MessageSquare className="w-5 h-5 mr-2 text-primary" />
              추가 꼬리 질문
            </h3>
            <div className="text-slate-600 leading-relaxed bg-blue-50/50 p-4 rounded-xl">
              {rd.followup}
            </div>
          </Card>
        )}

        <div className="flex justify-center gap-4 mt-4">
          <Button variant="outline" onClick={() => navigate('/')}>
            <Home className="w-5 h-5 mr-2" /> 처음으로
          </Button>
          <Button onClick={() => navigate('/setup')}>
            <RotateCcw className="w-5 h-5 mr-2" /> 다시 연습하기
          </Button>
        </div>
      </div>
    </Layout>
  );
};
