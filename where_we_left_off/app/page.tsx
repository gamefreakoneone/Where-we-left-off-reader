"use client";

import { useState, useEffect } from "react";
import LeftPanel from "./components/LeftPanel";
import PdfViewer from "./components/PdfViewer";
import RightPanel from "./components/RightPanel";
import FileUpload from "./components/FileUpload";

export default function Home() {
  const [bookId, setBookId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [processingStatus, setProcessingStatus] = useState<string>("");
  const [storyData, setStoryData] = useState<any>(null);
  const [bookmarkedPage, setBookmarkedPage] = useState<number>(1);

  // Poll for processing status
  useEffect(() => {
    if (bookId && processingStatus === "in_progress") {
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`http://127.0.0.1:8000/books/status/${bookId}`);
          if (!res.ok) throw new Error("Failed to fetch status");
          const data = await res.json();
          if (data.status === "complete") {
            setProcessingStatus("complete");
            clearInterval(interval);
          }
        } catch (error) {
          console.error("Status check failed:", error);
          setProcessingStatus("failed");
          clearInterval(interval);
        }
      }, 5000); // Poll every 5 seconds

      return () => clearInterval(interval);
    }
  }, [bookId, processingStatus]);

  // Fetch story data once processing is complete
  useEffect(() => {
    if (processingStatus === "complete" && bookId) {
      const fetchStoryData = async () => {
        try {
            // This is a placeholder for an endpoint that should return the full story_global_view.json
            // You may need to add this endpoint to your FastAPI backend
          const res = await fetch(`http://127.0.0.1:8000/books/data/${bookId}`);
          if (!res.ok) throw new Error("Failed to fetch story data");
          const data = await res.json();
          setStoryData(data);
        } catch (error) {
          console.error("Failed to fetch story data:", error);
        }
      };
      fetchStoryData();
    }
  }, [processingStatus, bookId]);

  const handleFileUpload = async (selectedFile: File) => {
    setFile(selectedFile);
    setProcessingStatus("uploading");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("http://127.0.0.1:8000/books/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      setBookId(data.book_id);
      setProcessingStatus("in_progress");
    } catch (error) {
      console.error("Upload failed:", error);
      setProcessingStatus("failed");
    }
  };

  if (!bookId) {
    return <FileUpload onFileUpload={handleFileUpload} status={processingStatus} />;
  }

  if (processingStatus !== "complete" || !storyData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-4">Processing your book...</h2>
          <p className="text-gray-400">This may take a few minutes. Please wait.</p>
          <div className="mt-4 w-16 h-16 border-4 border-dashed rounded-full animate-spin border-blue-500 mx-auto"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-[1fr_2fr_1fr] h-screen gap-4 p-4 bg-gray-900 text-white">
      <div className="bg-gray-800 rounded-lg p-4 overflow-y-auto">
        <LeftPanel 
          storyData={storyData} 
          bookmarkedPage={bookmarkedPage}
          onBookmarkPage={setBookmarkedPage}
        />
      </div>
      <div className="bg-gray-800 rounded-lg flex items-center justify-center">
        {file && <PdfViewer file={file} bookmarkedPage={bookmarkedPage} setBookmarkedPage={setBookmarkedPage} />}
      </div>
      <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
        <RightPanel 
            storyData={storyData} 
            bookmarkedPage={bookmarkedPage} 
            bookId={bookId}
        />
      </div>
    </div>
  );
}