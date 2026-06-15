export const mockSessionResult = {
  totalTime: 92,
  totalGestures: 4,
  score: 87,
  resultData: {
    question: '최근 프로젝트에서 예상치 못한 문제가 생겼을 때 어떻게 해결했나요?',
    verbal: {
      text: '처음에는 원인을 정확히 몰라서 조금 당황했습니다. 음 그래서 로그를 먼저 확인했고, 문제가 특정 API 응답 지연에서 시작된다는 걸 찾았습니다. 이후에는 팀원과 역할을 나눠서 임시 대응과 근본 원인 수정을 병행했습니다.',
      duration: 92,
      speech_rate: 2.8,
      filler_count: 1,
      repeat_count: 0,
      word_error_count: 0,
    },
    nonverbal: {
      clusters: [{ cluster_id: 0 }, { cluster_id: 1 }],
    },
    session_insights: {
      question: '최근 프로젝트에서 예상치 못한 문제가 생겼을 때 어떻게 해결했나요?',
      question_analysis: {
        label: '18.0초 부근부터 긴장 상승이 관찰됨',
        summary:
          "18초 전후 '처음에는 원인을 정확히 몰라서 조금 당황했습니다' 구간에서 긴장 신호가 올라갔고, 이후 해결 과정 설명으로 들어가며 자신감 점수가 회복되었습니다.",
        first_nervous_time: 18,
        answer_duration: 92,
      },
      transcript_segments: [
        {
          id: 0,
          start: 0,
          end: 12,
          text: '처음에는 원인을 정확히 몰라서 조금 당황했습니다.',
          highlight: 'high',
          reasons: ['긴장 점수 상승', '자신감 점수 하락'],
          state_scores: { focused: 0.48, engagement: 0.5, nervous: 0.76, confidence: 0.38 },
        },
        {
          id: 1,
          start: 12,
          end: 26,
          text: '음 그래서 로그를 먼저 확인했고, 문제가 특정 API 응답 지연에서 시작된다는 걸 찾았습니다.',
          highlight: 'medium',
          reasons: ['추임새 포함'],
          state_scores: { focused: 0.62, engagement: 0.58, nervous: 0.54, confidence: 0.56 },
        },
        {
          id: 2,
          start: 26,
          end: 55,
          text: '이후에는 팀원과 역할을 나눠서 임시 대응과 근본 원인 수정을 병행했습니다.',
          highlight: 'none',
          reasons: [],
          state_scores: { focused: 0.74, engagement: 0.72, nervous: 0.32, confidence: 0.78 },
        },
      ],
      change_points: [
        {
          kind: 'score_shift',
          time: 18,
          label: '긴장 상승',
          segment_text: '처음에는 원인을 정확히 몰라서 조금 당황했습니다.',
        },
        {
          kind: 'state_transition',
          time: 35,
          label: '자신감 회복',
          segment_text: '이후에는 팀원과 역할을 나눠서 임시 대응과 근본 원인 수정을 병행했습니다.',
        },
      ],
      focus_windows: [
        {
          start: 0,
          end: 20,
          state: 'nervous',
          scores: { focused: 0.48, engagement: 0.5, nervous: 0.76, confidence: 0.38 },
          reasons: ['긴장 신호가 평균보다 높음', '자신감 신호가 낮음'],
        },
        {
          start: 20,
          end: 35,
          state: 'neutral',
          scores: { focused: 0.62, engagement: 0.58, nervous: 0.54, confidence: 0.56 },
          reasons: ['추임새 포함 구간'],
        },
      ],
    },
    session_history: {
      count: 4,
      habit_summary: {
        summary: '최근 세션에서 반복된 추임새와 시선/긴장 관련 행동 신호를 습관 후보로 정리했습니다.',
        patterns: [
          {
            type: 'filler',
            label: '추임새 반복',
            description: "'음', '그' 같은 추임새가 복잡한 상황 설명 직전에 반복적으로 관찰되었습니다.",
          },
          {
            type: 'nonverbal_focus',
            label: '시선/집중 신호 반복',
            description: '문제 원인을 설명하는 구간에서 시선 안정도 저하 신호가 반복되었습니다.',
          },
          {
            type: 'nonverbal_tension',
            label: '긴장 행동 신호 반복',
            description: '답변 초반에 긴장 점수 상승과 자신감 점수 하락이 함께 관찰되는 경향이 있습니다.',
          },
        ],
      },
    },
    delivery_feedback:
      '초반 문제 상황을 설명할 때 긴장 신호가 올라갔지만, 해결 과정을 말하면서 시선과 말의 흐름이 안정되었습니다. 첫 문장에서 결론을 먼저 짚고 들어가면 긴장 구간이 더 짧아질 가능성이 큽니다.',
  },
};
