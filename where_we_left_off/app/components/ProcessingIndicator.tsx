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
          containerClass: 'bg-[var(--second-color)] text-[var(--fourth-color)] border border-[color:rgba(56,189,248,0.4)]',
          lightClass: 'animate-pulse bg-[var(--third-color)]',
          text: 'Processing analysis...',
          showDismiss: false,
        };
      case 'complete':
        return {
          containerClass: 'bg-[var(--third-color)] text-[var(--first-color)]',
          lightClass: 'bg-[var(--first-color)]',
          text: 'Processing complete!',
          showDismiss: true,
        };
      case 'failed':
        return {
          containerClass: 'bg-[color:rgba(56,189,248,0.12)] text-[var(--third-color)] border border-[color:rgba(56,189,248,0.4)]',
          lightClass: 'bg-red-500',
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
      <div className={`flex items-center p-3 rounded-lg shadow-2xl gap-3 ${content.containerClass}`}>
        <div className={`w-4 h-4 rounded-full ${content.lightClass}`}></div>
        <span className="font-medium">{content.text}</span>
        {content.showDismiss && (
          <button onClick={handleDismiss} className="ml-2 text-xl font-bold text-inherit hover:text-[color:rgba(248,250,252,0.7)]">
            &times;
          </button>
        )}
      </div>
    </div>
  );
}
