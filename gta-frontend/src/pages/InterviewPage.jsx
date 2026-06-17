import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2, Clock, Loader2 } from 'lucide-react';
import { useInterview } from '../context/InterviewContext';
import { useAnalysis } from '../hooks/useAnalysis';
import { Layout } from '../components/layout/Layout';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { VideoFeed } from '../components/interview/VideoFeed';
import { QuestionCard } from '../components/interview/QuestionCard';
import { LiveToast } from '../components/interview/LiveToast';

const FALLBACK_QUESTIONS = [
  '본인의 경험 중 이번 연습 상황과 가장 관련 있는 사례를 소개해 주세요.',
  '그 경험에서 본인이 맡은 역할과 문제를 해결한 과정을 설명해 주세요.',
  '비슷한 상황을 다시 만난다면 무엇을 더 개선해서 해보고 싶나요?',
];

const pickMimeType = () => {
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
};

export const InterviewPage = () => {
  const navigate = useNavigate();
  const { topic, setSessionData } = useInterview();
  const [questions, setQuestions] = useState(FALLBACK_QUESTIONS);
  const [questionSource, setQuestionSource] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [answers, setAnswers] = useState([]);
  const [mediaStream, setMediaStream] = useState(null);
  const [mediaError, setMediaError] = useState('');
  const [analysisError, setAnalysisError] = useState('');
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const startedRef = useRef(false);
  const { warnings, totalGestures } = useAnalysis(true);

  const currentQuestion = questions[currentIndex];
  const isLastQuestion = currentIndex === questions.length - 1;
  const progress = ((currentIndex + 1) / questions.length) * 100;
  const practiceTopic = useMemo(() => topic?.trim() || '일반 면접 연습', [topic]);

  useEffect(() => {
    let active = true;
    fetch('/api/generate-questions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic: practiceTopic }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        if (Array.isArray(data.questions) && data.questions.length >= 3) {
          setQuestions(data.questions.slice(0, 3));
          setQuestionSource(data.source || '');
        }
      })
      .catch(() => {
        if (active) setQuestions(FALLBACK_QUESTIONS);
      });
    return () => {
      active = false;
    };
  }, [practiceTopic]);

  useEffect(() => {
    let stream;
    if (!window.isSecureContext) {
      setMediaError(
        '원격 HTTP 주소에서는 브라우저가 카메라/마이크를 막습니다. 노트북 Chrome/Edge에서 chrome://flags/#unsafely-treat-insecure-origin-as-secure 를 열고 http://192.168.0.2:5173 을 추가한 뒤 브라우저를 재시작해 주세요.'
      );
      return () => {};
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setMediaError('이 브라우저에서는 카메라/마이크 녹화를 사용할 수 없습니다. Chrome 또는 Edge로 다시 열어주세요.');
      return () => {};
    }
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: true })
      .then((value) => {
        stream = value;
        setMediaStream(value);
      })
      .catch(() => {
        setMediaError('카메라 또는 마이크 권한을 가져오지 못했습니다. 주소창 왼쪽 권한 설정에서 카메라와 마이크를 허용해 주세요.');
      });

    return () => {
      const target = stream || mediaStream;
      target?.getTracks?.().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    let timerInterval;
    if (isRecording) {
      timerInterval = setInterval(() => {
        setSeconds((value) => value + 1);
      }, 1000);
    }
    return () => clearInterval(timerInterval);
  }, [isRecording]);

  const formatTime = (value) => {
    const mins = Math.floor(value / 60);
    const secs = value % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const ensureRecorder = () => {
    if (!mediaStream) {
      throw new Error('카메라와 마이크가 준비되지 않았습니다.');
    }
    if (recorderRef.current) return recorderRef.current;

    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
    recorder.ondataavailable = (event) => {
      if (event.data?.size > 0) chunksRef.current.push(event.data);
    };
    recorderRef.current = recorder;
    return recorder;
  };

  const handleStart = () => {
    setAnalysisError('');
    try {
      const recorder = ensureRecorder();
      if (!startedRef.current) {
        chunksRef.current = [];
        recorder.start(1000);
        startedRef.current = true;
      } else if (recorder.state === 'paused') {
        recorder.resume();
      }
      setSeconds(0);
      setIsRecording(true);
    } catch (error) {
      setMediaError(error.message);
    }
  };

  const handleStop = () => {
    const recorder = recorderRef.current;
    if (recorder?.state === 'recording') {
      recorder.requestData();
      recorder.pause();
    }
    const answerRecord = {
      question: currentQuestion,
      duration: seconds,
      completedAt: new Date().toISOString(),
    };
    setAnswers((prev) => [...prev, answerRecord]);
    setIsRecording(false);
  };

  const stopRecorderAndBuildBlob = () => new Promise((resolve, reject) => {
    const recorder = recorderRef.current;
    if (!recorder || !startedRef.current) {
      reject(new Error('녹화된 답변이 없습니다.'));
      return;
    }
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'video/webm' });
      resolve(blob);
    };
    if (recorder.state === 'recording' || recorder.state === 'paused') {
      recorder.requestData();
      recorder.stop();
    } else {
      recorder.onstop();
    }
  });

  const analyzeCurrentSession = async (finalAnswers) => {
    setIsAnalyzing(true);
    setAnalysisError('');
    try {
      const blob = await stopRecorderAndBuildBlob();
      const questionText = finalAnswers
        .map((answer, index) => `Q${index + 1}. ${answer.question}`)
        .join('\n');
      const params = new URLSearchParams({
        topic: practiceTopic,
        question: questionText,
        session: JSON.stringify({
          topic: practiceTopic,
          answers: finalAnswers,
        }),
      });
      const response = await fetch(`/api/analyze-current-session?${params.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: blob,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || '현재 세션 분석에 실패했습니다.');
      }

      setSessionData((prev) => ({
        ...prev,
        totalTime: finalAnswers.reduce((sum, item) => sum + (item.duration || 0), 0),
        totalGestures,
        score: 85,
        resultData: data,
        multiQuestionSession: {
          enabled: true,
          topic: practiceTopic,
          questions,
          answers: finalAnswers,
        },
      }));
      navigate('/result-current');
    } catch (error) {
      setAnalysisError(error.message);
      setIsAnalyzing(false);
    }
  };

  const handleNext = () => {
    const finalAnswers = answers;
    if (!isLastQuestion) {
      setCurrentIndex((index) => index + 1);
      setSeconds(0);
      return;
    }
    analyzeCurrentSession(finalAnswers);
  };

  return (
    <Layout withHeader>
      <div className="fade-in w-full max-w-6xl flex flex-col lg:flex-row gap-6 relative min-h-[600px] items-stretch">
        <LiveToast warnings={warnings} />
        <VideoFeed totalGestures={totalGestures} stream={mediaStream} isRecording={isRecording} />

        <div className="w-full lg:w-7/12 flex flex-col gap-6">
          <Card className="p-6 px-8 border border-slate-100 flex items-center justify-between">
            <div className="w-3/5">
              <div className="flex justify-between text-xs font-bold text-slate-400 mb-2">
                <span>진행률</span>
                <span>
                  {currentIndex + 1} / {questions.length} 질문
                </span>
              </div>
              <div className="w-full bg-background rounded-full h-2 overflow-hidden">
                <div
                  className="bg-primary h-2 rounded-full transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            <div className="flex items-center text-slate-700 font-bold text-lg bg-background px-5 py-2.5 rounded-xl border border-slate-200">
              <Clock className="w-5 h-5 mr-2.5 text-slate-400" />
              {formatTime(seconds)}
            </div>
          </Card>

          <QuestionCard
            questionNumber={currentIndex + 1}
            questionText={currentQuestion}
          />

          <Card className="p-10 border border-slate-100 flex-1 flex flex-col items-center justify-center relative overflow-hidden">
            {isAnalyzing ? (
              <div className="flex flex-col items-center text-center">
                <Loader2 className="w-9 h-9 text-primary animate-spin mb-4" />
                <div className="text-lg font-black text-slate-900">현재 세션을 분석하는 중입니다</div>
                <p className="text-sm text-slate-500 mt-2 leading-6">
                  녹화본을 실제 분석 모델에 넣고 있습니다. 영상 길이에 따라 시간이 조금 걸릴 수 있습니다.
                </p>
              </div>
            ) : (
              <>
                {isRecording ? (
                  <div className="bg-red-50 text-red-500 px-5 py-2.5 rounded-full text-sm font-bold flex items-center mb-4 border border-red-100 z-10">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500 mr-2.5 animate-pulse" />
                    답변 녹화 중
                  </div>
                ) : answers[currentIndex] ? (
                  <div className="bg-green-50 text-green-600 px-5 py-2.5 rounded-full text-sm font-bold flex items-center mb-4 border border-green-100 z-10">
                    <CheckCircle2 className="w-4 h-4 mr-2" />
                    이 질문 답변 완료
                  </div>
                ) : null}

                <p className="text-slate-500 mb-3 font-medium text-center">
                  {isRecording
                    ? '답변을 마치면 완료 버튼을 눌러 다음 질문으로 이동하세요.'
                    : '준비가 되면 답변을 시작하세요. 모든 질문을 마치면 현재 녹화본으로 리포트를 생성합니다.'}
                </p>
                {questionSource === 'llm' && (
                  <p className="text-xs font-bold text-primary mb-5">로컬 LLM이 생성한 질문입니다.</p>
                )}
                {(mediaError || analysisError) && (
                  <div className="mb-5 rounded-xl border border-red-100 bg-red-50 px-4 py-3 text-sm font-bold text-red-600">
                    {mediaError || analysisError}
                  </div>
                )}

                {isRecording ? (
                  <Button
                    variant="dark"
                    onClick={handleStop}
                    className="text-lg bg-red-600 hover:bg-red-700 text-white border-none"
                  >
                    답변 완료
                  </Button>
                ) : answers.length > currentIndex ? (
                  <Button variant="dark" onClick={handleNext} className="text-lg">
                    {isLastQuestion ? '현재 세션 분석하기' : '다음 질문'}
                  </Button>
                ) : (
                  <Button variant="dark" onClick={handleStart} className="text-lg" disabled={!mediaStream}>
                    답변 시작하기
                  </Button>
                )}
              </>
            )}
          </Card>
        </div>
      </div>
    </Layout>
  );
};
