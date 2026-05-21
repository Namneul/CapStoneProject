import React, { createContext, useContext, useState } from 'react';

// 면접 진행 상태와 데이터를 전역에서 관리하는 Context
const InterviewContext = createContext();

export const useInterview = () => useContext(InterviewContext);

export const InterviewProvider = ({ children }) => {
  const [topic, setTopic] = useState('');
  const [sessionData, setSessionData] = useState({
    sessionID: null,
    totalTime: 0,
    totalGestures: 0,
    score: 0,
  });

  // 세션 종료 시 데이터 누적 저장
  const completeSession = (data) => {
    setSessionData(prev => ({ ...prev, ...data }));
  };

  return (
    <InterviewContext.Provider value={{ topic, setTopic, sessionData, setSessionData, completeSession }}>
      {children}
    </InterviewContext.Provider>
  );
};
