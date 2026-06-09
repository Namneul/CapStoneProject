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
    // 추후 실시간 WebSocket 통신 등 실제 백엔드 연동 로직으로 교체
  }, [isActive]);

  return { warnings, totalGestures };
};
