"use client";

import { useMemo } from 'react';

interface LeftPanelProps {
  graphData: any;
  bookmarkedPage: number;
  onBookmarkPage: (page: number) => void;
}

export default function LeftPanel({ graphData: storyData, bookmarkedPage, onBookmarkPage }: LeftPanelProps) {

  const currentChapter = useMemo(() => {
    if (!storyData || !storyData.chapter) return null;
    return storyData.chapter.find((chap: any) => 
      bookmarkedPage >= chap.pages[0] && bookmarkedPage <= chap.pages[1]
    );
  }, [storyData, bookmarkedPage]);

  return (
    <div className="flex flex-col h-full p-4 rounded-lg bg-[var(--second-color)] text-[var(--fourth-color)] shadow-inner">
      <h1 className="text-2xl font-bold mb-2 text-[var(--fourth-color)]">{storyData?.book_title || "Your Story"}</h1>
      <p className="text-sm text-[color:rgba(248,250,252,0.7)] mb-6">{storyData?.author || "Unknown Author"}</p>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Story So Far</h2>
        <div className="text-sm text-[color:rgba(248,250,252,0.75)] overflow-y-auto max-h-64 pr-2">
            {currentChapter ? currentChapter.summary_global : "No summary available for this section."}
        </div>
      </div>

      <div className="mt-auto">
        <h2 className="text-xl font-semibold mb-2">Bookmark</h2>
        <p className="text-sm text-[color:rgba(248,250,252,0.7)] mb-2">Currently on page: {bookmarkedPage}</p>
        <button
            onClick={() => onBookmarkPage(bookmarkedPage)}
            className="w-full bg-[var(--third-color)] text-[var(--first-color)] font-bold py-2 px-4 rounded-lg transition-transform transform hover:scale-[1.02]"
        >
            Set Bookmark to Current Page
        </button>
      </div>
    </div>
  );
}
