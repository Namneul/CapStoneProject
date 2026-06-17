import React, { useEffect, useState } from 'react';
import { useInterview } from '../context/InterviewContext';
import { ResultDashboardPage } from './ResultDashboardPage';

export const LatestResultPage = ({ mode = 'latest' }) => {
  const { sessionData } = useInterview();
  const [state, setState] = useState({ loading: true, data: null, error: '' });
  const isCurrent = mode === 'current';

  useEffect(() => {
    if (isCurrent && sessionData.resultData) {
      setState({ loading: false, data: sessionData.resultData, error: '' });
      return;
    }

    fetch(isCurrent ? '/api/current-result' : '/api/latest-result')
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error || '분석 결과를 불러오지 못했습니다.');
        }
        return res.json();
      })
      .then((data) => setState({ loading: false, data, error: '' }))
      .catch((err) => setState({ loading: false, data: null, error: err.message }));
  }, [isCurrent, sessionData.resultData]);

  if (state.loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-sm font-bold text-slate-500">
        분석 결과를 불러오는 중...
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-6">
        <div className="max-w-md rounded-2xl border border-slate-200 bg-white p-6 text-center shadow-soft">
          <h1 className="text-lg font-black text-slate-900 mb-2">결과를 찾을 수 없습니다</h1>
          <p className="text-sm text-slate-500">{state.error}</p>
        </div>
      </div>
    );
  }

  return <ResultDashboardPage resultData={state.data} />;
};
