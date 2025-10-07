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
      {/* <p className="text-sm text-muted mb-6">{storyData?.author || "Unknown Author"}</p> */}

      <div className="mb-6">
        <div className="flex items-center justify-between gap-4 mb-2">
          <h2 className="text-xl font-semibold">Story So Far</h2>
          <button
            type="button"
            onClick={() => setIsSummaryOpen((prev) => !prev)}
            aria-expanded={isSummaryOpen}
            className="text-sm text-fourth-color hover:text-third-color focus:outline-none focus:ring-2 focus:ring-fourth-color rounded"
          >
            {isSummaryOpen ? "Hide summary" : "Show summary"}
          </button>
        </div>
        {isSummaryOpen && (
          <div className="text-muted text-sm overflow-y-auto max-h-64 pr-2">
            {currentChapter ? currentChapter.summary_global : "No summary available for this section."}
          </div>
        )}
      </div>

      <div className="mt-auto">
        <h2 className="text-xl font-semibold mb-2">Bookmark</h2>
        <p className="text-sm text-muted mb-2">Currently on page: {bookmarkedPage}</p>
        <button 
            onClick={() => onBookmarkPage(bookmarkedPage)}
            className="w-full bg-fourth-color hover:bg-third-color text-first-color font-bold py-2 px-4 rounded-lg"
        >
            Set Bookmark to Current Page
        </button>
      </div>
    </div>
  );
}
