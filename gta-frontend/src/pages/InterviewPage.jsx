import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Clock } from 'lucide-react';
import { useInterview } from '../context/InterviewContext';
import { useAnalysis } from '../hooks/useAnalysis';
import { Layout } from '../components/layout/Layout';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { VideoFeed } from '../components/interview/VideoFeed';
import { QuestionCard } from '../components/interview/QuestionCard';
import { LiveToast } from '../components/interview/LiveToast';

export const InterviewPage = () => {
  const navigate = useNavigate();
  const { topic, completeSession, setSessionData } = useInterview();
  const [seconds, setSeconds] = useState(0);
  
  const [question, setQuestion] = useState('질문 생성 중...');
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isReady, setIsReady] = useState(false);

  // Custom Hook을 통한 분석 상태 가져오기
  const { warnings, totalGestures } = useAnalysis(true);

  useEffect(() => {
    let timerInterval;
    if (isRecording) {
      timerInterval = setInterval(() => {
        setSeconds(s => s + 1);
      }, 1000);
    }
    return () => clearInterval(timerInterval);
  }, [isRecording]);

  useEffect(() => {
    // 1. 페이지 접속 시 백엔드 프로세스 시작 및 질문 받아오기
    const initInterview = async () => {
      try {
        const res = await fetch('/api/start-interview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic: topic || '1' })
        });
        
        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.error || 'Server returned an error');
        }
        
        const data = await res.json();
        if (data.question) {
          setQuestion(data.question);
          setIsReady(true);
        } else {
          throw new Error('질문 데이터가 없습니다.');
        }
      } catch (err) {
        console.error(err);
        setQuestion(`오류 발생: ${err.message}. 백엔드/파이썬 환경을 확인해 주세요.`);
      }
    };
    initInterview();

    return () => {
      // 컴포넌트 언마운트 시 인터뷰 취소
      fetch('/api/cancel-interview', { method: 'POST' }).catch(() => {});
    };
  }, [topic]);

  const formatTime = (s) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const handleStartRecording = async () => {
    if (!isReady) return;
    setIsRecording(true);
    setIsAnalyzing(true);
    
    try {
      // 2. 답변 시작 -> 파이썬 OpenCV 카메라 창이 팝업됨
      const res = await fetch('/api/start-recording', {
        method: 'POST'
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || 'Server returned an error');
      }
      
      const data = await res.json();
      
      // 3. 파이썬 프로세스 종료 및 결과 반환 시
      setIsRecording(false);
      
      // 실제 데이터를 InterviewContext에 저장
      setSessionData({
        totalTime: data.verbal?.duration || seconds,
        totalGestures: data.nonverbal?.clusters?.length || totalGestures,
        score: calculateMockScore(data), // 가상의 점수 계산
        resultData: data
      });
      
      navigate('/result');
    } catch (err) {
      console.error(err);
      alert(`분석 중 오류가 발생했습니다: ${err.message}`);
      setIsRecording(false);
      setIsAnalyzing(false);
    }
  };
  
  // 간단한 가상 점수 계산 로직 (실제 서비스에서는 백엔드 평가 점수 사용)
  const calculateMockScore = (data) => {
    if (!data.content_evaluation) return 80;
    return 85 + Math.floor(Math.random() * 10);
  };

  return (
    <Layout withHeader>
      <div className="fade-in w-full max-w-6xl flex flex-col lg:flex-row gap-6 relative min-h-[600px] items-stretch">
        
        <LiveToast warnings={warnings} />

        <VideoFeed totalGestures={totalGestures} />

        <div className="w-full lg:w-7/12 flex flex-col gap-6">
          <Card className="p-6 px-8 border border-slate-100 flex items-center justify-between">
            <div className="w-3/5">
              <div className="flex justify-between text-xs font-bold text-slate-400 mb-2">
                <span>진행률</span>
                <span>1 / 1 질문</span>
              </div>
              <div className="w-full bg-background rounded-full h-2 overflow-hidden">
                <div className="bg-primary h-2 rounded-full w-full transition-all"></div>
              </div>
            </div>
            <div className="flex items-center text-slate-700 font-bold text-lg bg-background px-5 py-2.5 rounded-xl border border-slate-200">
              <Clock className="w-5 h-5 mr-2.5 text-slate-400" />
              {formatTime(seconds)}
            </div>
          </Card>

          <QuestionCard 
            questionNumber={1} 
            questionText={question} 
          />

          <Card className="p-10 border border-slate-100 flex-1 flex flex-col items-center justify-center relative overflow-hidden">
            {isRecording ? (
              <div className="bg-red-50 text-red-500 px-5 py-2.5 rounded-full text-sm font-bold flex items-center mb-4 border border-red-100 z-10">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500 mr-2.5 animate-pulse"></div>
                녹음 중... (별도의 파이썬 카메라 창을 확인하세요)
              </div>
            ) : null}
            
            <p className="text-slate-500 mb-8 font-medium text-center">
              {isAnalyzing 
                ? '파이썬 카메라 창에서 답변을 진행하고 q를 눌러 종료하세요.' 
                : isReady 
                  ? '버튼을 누르면 별도의 카메라 창이 뜨면서 분석이 시작됩니다.'
                  : '에이전트가 질문을 준비하고 있습니다...'}
            </p>

            <Button 
              variant="dark" 
              onClick={handleStartRecording} 
              className="text-lg"
              disabled={!isReady || isAnalyzing}
            >
              {isAnalyzing ? '분석 진행 중...' : '답변 시작하기'}
            </Button>
          </Card>
        </div>
      </div>
    </Layout>
  );
};
