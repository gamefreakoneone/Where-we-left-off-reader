"use client";

import RelationshipGraph from "./RelationshipGraph";
import ChatWindow from "./ChatWindow";

interface RightPanelProps {
  graphData: any;
  bookmarkedPage: number;
  bookId: string;
  onNewGraphData: (data: any) => void;
}

export default function RightPanel({ graphData, bookmarkedPage, bookId, onNewGraphData }: RightPanelProps) {
  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex-grow h-1/2 bg-third-color rounded-lg p-2">
        <h2 className="text-xl font-semibold mb-2 text-center">Character Relationships</h2>
        <RelationshipGraph graphData={graphData} bookmarkedPage={bookmarkedPage} />
      </div>
      <div className="flex-grow h-1/2 bg-third-color rounded-lg">
        <ChatWindow 
          bookId={bookId} 
          bookmarkedPage={bookmarkedPage}
          onNewGraphData={onNewGraphData}
        />
      </div>
    </div>
  );
}
