"use client";

import { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

interface PdfViewerProps {
  file: File | string; // Accept either a File object or a URL string
  bookmarkedPage: number;
  setBookmarkedPage: (page: number) => void;
}

export default function PdfViewer({ file, bookmarkedPage, setBookmarkedPage }: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // According to React-PDF documentation, use CDN with proper version
    pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
    setIsReady(true);
  }, []);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
  }

  const goToPrevPage = () => setBookmarkedPage(Math.max(1, bookmarkedPage - 1));
  const goToNextPage = () => setBookmarkedPage(Math.min(numPages!, bookmarkedPage + 1));

  if (!isReady) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-[var(--fourth-color)]">Loading PDF viewer...</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col items-center">
      <div className="flex-grow overflow-y-auto w-full flex justify-center">
        <Document
          file={file}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={<div className="text-[var(--fourth-color)]">Loading PDF...</div>}
          error={<div className="text-red-500">Error loading PDF</div>}
        >
          <Page pageNumber={bookmarkedPage} />
        </Document>
      </div>
      <div className="flex items-center justify-center p-4 bg-[var(--second-color)] rounded-b-lg w-full gap-6 shadow-inner">
        <button onClick={goToPrevPage} disabled={bookmarkedPage <= 1} className="px-4 py-2 bg-[var(--third-color)] text-[var(--first-color)] rounded-md disabled:opacity-60 transition-transform hover:scale-[1.02] disabled:hover:scale-100">
          Previous
        </button>
        <p className="text-[var(--fourth-color)]">Page {bookmarkedPage} of {numPages}</p>
        <button onClick={goToNextPage} disabled={bookmarkedPage >= numPages!} className="px-4 py-2 bg-[var(--third-color)] text-[var(--first-color)] rounded-md disabled:opacity-60 transition-transform hover:scale-[1.02] disabled:hover:scale-100">
          Next
        </button>
      </div>
    </div>
  );
}
