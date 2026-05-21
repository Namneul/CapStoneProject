import React from 'react';
import { Card } from '../ui/Card';

export const QuestionCard = ({ questionNumber, questionText }) => {
  return (
    <Card className="p-8 flex items-start">
      <div className="bg-blue-50 text-primary rounded-xl w-12 h-12 flex items-center justify-center font-black flex-shrink-0 mr-5 text-xl">
        Q{questionNumber || ''}
      </div>
      <h3 className="text-xl font-bold text-slate-800 leading-relaxed pt-1">
        {questionText}
      </h3>
    </Card>
  );
};
