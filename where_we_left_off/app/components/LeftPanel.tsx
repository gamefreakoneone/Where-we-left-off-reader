"use client";

import { useMemo, useState } from 'react';

interface LeftPanelProps {
  graphData: any;
  bookmarkedPage: number;
  onBookmarkPage: (page: number) => void;
}

export default function LeftPanel({ graphData: storyData, bookmarkedPage, onBookmarkPage }: LeftPanelProps) {

  const [isSummaryOpen, setIsSummaryOpen] = useState(false);
  const currentChapter = useMemo(() => {
    if (!storyData || !storyData.chapter) return null;
    return storyData.chapter.find((chap: any) =>
      bookmarkedPage >= chap.pages[0] && bookmarkedPage <= chap.pages[1]
    );
  }, [storyData, bookmarkedPage]);

  return (
    <div className="flex flex-col h-full p-2">
      <h1 className="text-2xl font-bold mb-2">{storyData?.book_title || "Your Story"}</h1>
      <p className="text-sm text-gray-400 mb-6">{storyData?.author || "Unknown Author"}</p>

      <div className="mb-6">
        <div className="flex items-center justify-between gap-4 mb-2">
          <h2 className="text-xl font-semibold">Story So Far</h2>
          <button
            type="button"
            onClick={() => setIsSummaryOpen((prev) => !prev)}
            aria-expanded={isSummaryOpen}
            className="text-sm text-green-400 hover:text-green-300 focus:outline-none focus:ring-2 focus:ring-green-400 rounded"
          >
            {isSummaryOpen ? "Hide summary" : "Show summary"}
          </button>
        </div>
        {isSummaryOpen && (
          <div className="text-gray-300 text-sm overflow-y-auto max-h-64 pr-2">
            {currentChapter ? currentChapter.summary_global : "No summary available for this section."}
          </div>
        )}
      </div>

      <div className="mt-auto">
        <h2 className="text-xl font-semibold mb-2">Bookmark</h2>
        <p className="text-sm text-gray-400 mb-2">Currently on page: {bookmarkedPage}</p>
        <button 
            onClick={() => onBookmarkPage(bookmarkedPage)}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg"
        >
            Set Bookmark to Current Page
        </button>
      </div>
    </div>
  );
}
