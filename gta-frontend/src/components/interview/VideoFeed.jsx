import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, VideoOff } from 'lucide-react';
import { Card } from '../ui/Card';

export const VideoFeed = ({ totalGestures = 0, stream = null, isRecording = false }) => {
  const videoRef = useRef(null);
  const prevGestureCountRef = useRef(0);
  const [popAnimation, setPopAnimation] = useState(false);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  useEffect(() => {
    if (totalGestures > prevGestureCountRef.current) {
      setPopAnimation(true);
      setTimeout(() => setPopAnimation(false), 300);
    }
    prevGestureCountRef.current = totalGestures;
  }, [totalGestures]);

  return (
    <Card className="w-full lg:w-5/12 p-5 flex flex-col h-full">
      <div className="relative bg-slate-900 rounded-2xl flex-1 flex flex-col items-center justify-center overflow-hidden min-h-[450px]">
        <div className="absolute top-5 left-5 bg-red-500/20 text-red-500 px-4 py-1.5 rounded-full text-xs font-bold flex items-center border border-red-500/30 backdrop-blur-md z-10">
          <div className={`w-2 h-2 rounded-full bg-red-500 mr-2.5 ${isRecording ? 'animate-pulse' : ''}`} />
          {isRecording ? 'REC' : 'READY'}
        </div>

        {stream ? (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="absolute inset-0 w-full h-full object-cover rounded-2xl opacity-90"
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500">
            <VideoOff className="w-16 h-16 text-slate-700/80 mb-4" />
            <span className="text-sm font-bold">카메라 권한을 기다리는 중</span>
          </div>
        )}

        <div className="absolute bottom-5 left-5 right-5 bg-slate-800/80 backdrop-blur-md rounded-xl p-4 flex items-center text-white border border-slate-700/50">
          <AlertTriangle className={`w-5 h-5 mr-3 flex-shrink-0 ${totalGestures > 0 ? 'text-warning' : 'text-slate-400'}`} />
          <span className="font-bold text-sm tracking-wide">
            누적 경고:
            <span className={`inline-block ml-2 text-lg ${popAnimation ? 'count-up-pop text-warning' : (totalGestures > 0 ? 'text-warning' : '')}`}>
              {totalGestures}회
            </span>
          </span>
        </div>
      </div>
    </Card>
  );
};
