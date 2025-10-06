"use client";

import { useState } from 'react';

interface ProcessingIndicatorProps {
  status: 'in_progress' | 'complete' | 'failed' | 'idle';
}

export default function ProcessingIndicator({ status }: ProcessingIndicatorProps) {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible || status === 'idle') {
    return null;
  }

  const handleDismiss = () => {
    setIsVisible(false);
  };

  const getStatusContent = () => {
    switch (status) {
      case 'in_progress':
        return {
          bgColor: 'bg-orange-500',
          lightClass: 'animate-pulse',
          text: 'Processing analysis...',
          showDismiss: false,
        };
      case 'complete':
        return {
          bgColor: 'bg-green-500',
          lightClass: '',
          text: 'Processing complete!',
          showDismiss: true,
        };
      case 'failed':
        return {
          bgColor: 'bg-red-500',
          lightClass: '',
          text: 'Processing failed.',
          showDismiss: true,
        };
      default:
        return null;
    }
  };

  const content = getStatusContent();
  if (!content) return null;

  return (
    <div className="fixed bottom-5 right-5 z-50">
      <div className={`flex items-center p-3 rounded-lg shadow-2xl text-white ${content.bgColor}`}>
        <div className={`w-4 h-4 rounded-full mr-3 ${content.lightClass} ${content.bgColor}`}></div>
        <span className="font-medium">{content.text}</span>
        {content.showDismiss && (
          <button onClick={handleDismiss} className="ml-4 text-xl font-bold hover:text-gray-200">
            &times;
          </button>
        )}
      </div>
    </div>
  );
}
