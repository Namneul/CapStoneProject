import { useState, useEffect } from 'react';

/**
 * 백엔드 통신 및 분석 상태를 시뮬레이션하는 커스텀 훅
 * 실제 서비스 시 이 훅 내부만 WebSocket이나 Polling 로직으로 교체하면 됩니다.
 */
export const useAnalysis = (isActive) => {
  const [warnings, setWarnings] = useState([]);
  const [totalGestures, setTotalGestures] = useState(0);

  useEffect(() => {
    if (!isActive) return;

    // 시뮬레이션: 3초마다 체크하여 랜덤하게 제스처 경고 발생
    const interval = setInterval(() => {
      // 25% 확률로 경고 생성
      if (Math.random() < 0.25) { 
        const newWarning = {
          id: Date.now(),
          type: Math.random() > 0.5 ? 'gaze' : 'hand',
          message: '시선 이탈 및 부적절한 제스처 감지'
        };
        
        setWarnings(prev => [...prev, newWarning]);
        setTotalGestures(prev => prev + 1);
        
        // 3초 뒤 경고 큐에서 제거 (토스트 애니메이션 용도)
        setTimeout(() => {
          setWarnings(prev => prev.filter(w => w.id !== newWarning.id));
        }, 3000);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isActive]);

  return { warnings, totalGestures };
};
