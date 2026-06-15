import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { InterviewProvider } from './context/InterviewContext';

import { LoginPage } from './pages/LoginPage';
import { TopicSetupPage } from './pages/TopicSetupPage';
import { InterviewPage } from './pages/InterviewPage';
import { ResultDashboardPage } from './pages/ResultDashboardPage';
import { LatestResultPage } from './pages/LatestResultPage';

function App() {
  return (
    <InterviewProvider>
      <Router>
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/setup" element={<TopicSetupPage />} />
          <Route path="/interview" element={<InterviewPage />} />
          <Route path="/result" element={<ResultDashboardPage />} />
          <Route path="/result-demo" element={<ResultDashboardPage demo />} />
          <Route path="/result-latest" element={<LatestResultPage />} />
        </Routes>
      </Router>
    </InterviewProvider>
  );
}

export default App;
