import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft } from 'lucide-react';
import { useInterview } from '../context/InterviewContext';
import { Layout } from '../components/layout/Layout';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

const recommendedTopics = [
  '신입 프론트엔드 개발자 면접',
  'React 프로젝트 경험 면접',
  '팀 프로젝트 발표 연습',
  '서비스 기획 아이디어 피칭',
];

export const TopicSetupPage = () => {
  const navigate = useNavigate();
  const { topic, setTopic } = useInterview();

  const handleStart = () => {
    if (!topic.trim()) setTopic('일반 면접 연습');
    navigate('/interview');
  };

  return (
    <Layout>
      <div className="fade-in w-full max-w-2xl">
        <Card className="p-10 border border-slate-100">
          <button
            onClick={() => navigate('/')}
            className="flex items-center text-slate-400 hover:text-slate-800 transition-colors mb-8 font-medium text-sm border border-slate-200 rounded-xl px-4 py-2 hover:bg-slate-50"
          >
            <ChevronLeft className="w-4 h-4 mr-1" /> 돌아가기
          </button>

          <div className="mb-8">
            <h2 className="text-2xl font-bold text-slate-800 mb-2">연습 주제 설정</h2>
            <p className="text-slate-500 font-medium text-sm">
              면접, 발표, 피칭처럼 연습하고 싶은 상황을 적어주세요.
            </p>
          </div>

          <div className="mb-6 relative">
            <textarea
              className="w-full h-40 bg-background border border-slate-200 rounded-2xl p-6 outline-none focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all resize-none text-slate-700 font-medium leading-relaxed"
              placeholder="예: 신입 프론트엔드 개발자 면접을 연습하고 싶어요."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              maxLength={500}
            />
            <div className="absolute bottom-5 right-6 text-sm font-medium text-slate-400">
              {topic.length} / 500
            </div>
          </div>

          <div className="mb-10">
            <h4 className="text-sm font-bold text-slate-400 mb-4 px-1">추천 주제</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {recommendedTopics.map((item) => (
                <button
                  key={item}
                  onClick={() => setTopic(item)}
                  className="text-left px-5 py-4 bg-white border border-slate-200 rounded-2xl text-slate-600 font-medium text-sm hover:border-primary hover:text-primary hover:bg-blue-50/50 transition-all shadow-sm hover:shadow"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <Button onClick={handleStart} className="w-full text-lg">
            연습 시작하기
          </Button>
        </Card>
      </div>
    </Layout>
  );
};
