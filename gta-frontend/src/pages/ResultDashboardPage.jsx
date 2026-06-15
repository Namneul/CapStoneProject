import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Activity,
  AlertTriangle,
  Clock,
  Home,
  MessageSquare,
  RotateCcw,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import { useInterview } from '../context/InterviewContext';
import { Layout } from '../components/layout/Layout';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { ScoreBanner } from '../components/analytics/ScoreBanner';
import { mockSessionResult } from '../data/mockSessionResult';

const highlightClass = {
  high: 'border-red-200 bg-red-50 text-red-950',
  medium: 'border-amber-200 bg-amber-50 text-amber-950',
  none: 'border-slate-200 bg-white text-slate-700',
};

const formatTime = (value = 0) => {
  const safe = Number.isFinite(Number(value)) ? Number(value) : 0;
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return mins > 0 ? `${mins}분 ${secs}초` : `${secs}초`;
};

const formatRange = (start, end) => `${formatTime(start)} - ${formatTime(end)}`;

export const ResultDashboardPage = ({ demo = false, resultData = null }) => {
  const navigate = useNavigate();
  const { sessionData } = useInterview();
  const [openSegments, setOpenSegments] = useState({});

  const activeSession = resultData
    ? {
        totalTime: resultData.verbal?.duration || 0,
        totalGestures: resultData.nonverbal?.clusters?.length || 0,
        score: 85,
        resultData,
      }
    : demo
      ? mockSessionResult
      : sessionData;

  if (activeSession.score === 0) {
    navigate('/');
    return null;
  }

  const rd = activeSession.resultData || {};
  const verbal = rd.verbal || {};
  const insights = rd.session_insights || {};
  const history = rd.session_history || {};
  const segments = insights.transcript_segments || [];
  const highlightedSegments = segments.filter((item) => item.highlight !== 'none');
  const changePoints = insights.change_points || [];
  const focusWindows = insights.focus_windows || [];
  const habitPatterns = history.habit_summary?.patterns || [];
  const highlightedCount = highlightedSegments.length;

  const toggleSegment = (id) => {
    setOpenSegments((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const statItems = [
    {
      icon: Clock,
      label: '답변 시간',
      value: formatTime(verbal.duration || activeSession.totalTime),
    },
    {
      icon: Activity,
      label: '말 속도',
      value: `${verbal.speech_rate || 0} 단어/초`,
    },
    {
      icon: MessageSquare,
      label: '추임새',
      value: `${verbal.filler_count || 0}회`,
    },
    {
      icon: AlertTriangle,
      label: '피드백 구간',
      value: `${highlightedCount}구간`,
    },
  ];

  return (
    <Layout withHeader>
      <div className="fade-in w-full max-w-6xl py-8 flex flex-col gap-6">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 mb-2">세션 흐름 분석</h2>
          <p className="text-sm text-slate-500">
            발화 내용, 비언어 변화, 반복 습관 후보를 같은 시간축에서 확인합니다.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ScoreBanner score={activeSession.score} />

          <div className="lg:col-span-2 grid grid-cols-2 gap-4">
            {statItems.map((stat) => (
              <Card key={stat.label} className="p-5 border border-slate-100">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-slate-100 text-slate-700 flex items-center justify-center">
                    <stat.icon className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-xs font-bold text-slate-400">{stat.label}</div>
                    <div className="text-lg font-black text-slate-900">{stat.value}</div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>

        <Card className="p-6 border border-slate-100">
          <div className="flex items-center gap-2 mb-5">
            <MessageSquare className="w-5 h-5 text-primary" />
            <h3 className="text-lg font-bold text-slate-900">발화 내용 하이라이트</h3>
          </div>

          {highlightedSegments.length > 0 ? (
            <div className="space-y-3">
              {highlightedSegments.map((segment) => {
                const isOpen = Boolean(openSegments[segment.id]);
                const details = segment.feedback_details || [];
                return (
                  <div
                    key={segment.id}
                    className={`rounded-2xl border ${highlightClass[segment.highlight] || highlightClass.none}`}
                  >
                    <button
                      type="button"
                      onClick={() => toggleSegment(segment.id)}
                      className="w-full p-4 text-left"
                    >
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <p className="text-sm leading-relaxed font-medium">{segment.text}</p>
                        <span className="shrink-0 text-xs font-bold text-slate-500">
                          {formatRange(segment.start, segment.end)}
                        </span>
                      </div>
                      {segment.reasons?.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-3">
                          {segment.reasons.map((reason) => (
                            <span
                              key={reason}
                              className="rounded-full bg-white/70 border border-current/10 px-3 py-1 text-xs font-bold"
                            >
                              {reason}
                            </span>
                          ))}
                        </div>
                      )}
                    </button>

                    {isOpen && (
                      <div className="border-t border-current/10 bg-white/70 px-4 py-4">
                        {details.length > 0 ? (
                          <div className="space-y-3">
                            {details.map((detail, index) => (
                              <div key={`${segment.id}-${index}`} className="rounded-xl border border-slate-200 bg-white p-4">
                                <div className="text-sm font-black text-slate-900">{detail.title}</div>
                                <div className="mt-2 text-sm text-slate-600">
                                  <span className="font-bold text-slate-800">근거: </span>
                                  {detail.evidence}
                                </div>
                                <div className="mt-2 text-sm text-slate-600">
                                  <span className="font-bold text-slate-800">피드백: </span>
                                  {detail.feedback}
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-sm text-slate-500">
                            이 구간은 참고용으로 표시되었고, 별도 세부 피드백은 없습니다.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="rounded-2xl bg-slate-50 border border-slate-100 p-5 text-sm text-slate-500">
              피드백이 필요한 발화 구간이 감지되지 않았습니다.
            </div>
          )}
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="p-6 border border-slate-100">
            <div className="flex items-center gap-2 mb-5">
              <TrendingUp className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-bold text-slate-900">상태 변화점</h3>
            </div>
            {changePoints.length > 0 ? (
              <div className="space-y-4">
                {changePoints.slice(0, 6).map((point, index) => (
                  <div key={`${point.kind}-${index}`} className="border-l-4 border-primary pl-4 py-1">
                    <div className="text-sm font-black text-slate-900">
                      {formatTime(point.time)} · {point.label}
                    </div>
                    {point.segment_text && (
                      <div className="text-sm text-slate-500 mt-1">{point.segment_text}</div>
                    )}
                    {point.evidence && (
                      <div className="text-xs text-slate-500 mt-2">
                        <span className="font-bold text-slate-700">근거: </span>
                        {point.evidence}
                      </div>
                    )}
                    {point.feedback && (
                      <div className="text-xs text-slate-500 mt-1">
                        <span className="font-bold text-slate-700">피드백: </span>
                        {point.feedback}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-500">큰 상태 변화점은 감지되지 않았습니다.</p>
            )}
          </Card>

          <Card className="p-6 border border-slate-100">
            <div className="flex items-center gap-2 mb-5">
              <Sparkles className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-bold text-slate-900">반복 습관 후보</h3>
            </div>
            <p className="text-sm text-slate-500 mb-4">
              {history.habit_summary?.summary || '세션 기록이 쌓이면 반복 패턴을 보여줍니다.'}
            </p>
            {habitPatterns.length > 0 ? (
              <div className="space-y-3">
                {habitPatterns.map((pattern) => (
                  <div key={`${pattern.type}-${pattern.label}`} className="rounded-2xl bg-slate-50 border border-slate-100 p-4">
                    <div className="text-sm font-black text-slate-900">{pattern.label}</div>
                    <div className="text-sm text-slate-600 mt-1">{pattern.description}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-2xl bg-slate-50 border border-slate-100 p-4 text-sm text-slate-500">
                현재 누적 세션 수: {history.count || 0}
              </div>
            )}
          </Card>
        </div>

        {focusWindows.length > 0 && (
          <Card className="p-6 border border-slate-100">
            <h3 className="text-lg font-bold text-slate-900 mb-4">피드백 우선 구간</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {focusWindows.slice(0, 4).map((window) => (
                <div key={`${window.start}-${window.end}`} className="rounded-2xl border border-slate-100 bg-slate-50 p-4">
                  <div className="text-sm font-black text-slate-900">{formatRange(window.start, window.end)}</div>
                  <div className="text-xs text-slate-500 mt-1">
                    nervous {window.scores?.nervous ?? 0} · confidence {window.scores?.confidence ?? 0}
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {window.reasons?.map((reason) => (
                      <span key={reason} className="rounded-full bg-white border border-slate-200 px-3 py-1 text-xs font-bold text-slate-600">
                        {reason}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {rd.delivery_feedback && (
          <Card className="p-6 border border-slate-100">
            <h3 className="text-lg font-bold text-slate-900 mb-2">전체 답변 피드백</h3>
            <p className="text-sm text-slate-500 mb-5">
              사용자가 말한 답변 전체를 기준으로 내용 구성, 흐름, 전달력을 종합해서 정리합니다.
            </p>

            <div className="rounded-2xl bg-white border border-slate-200 p-5 mb-4 leading-8 text-slate-800">
              <div className="text-xs font-bold text-slate-400 mb-2">사용자 전체 답변</div>
              <div className="text-sm leading-7 text-slate-700">{verbal.text || '발화 내용이 없습니다.'}</div>
            </div>

            <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap rounded-2xl bg-slate-50 p-4 border border-slate-100">
              {rd.delivery_feedback}
            </div>
          </Card>
        )}

        <div className="flex justify-center gap-4 mt-2">
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
