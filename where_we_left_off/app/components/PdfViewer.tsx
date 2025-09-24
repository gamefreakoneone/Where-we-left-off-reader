"use client";

import { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Set up worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

interface PdfViewerProps {
  file: File;
  bookmarkedPage: number;
  setBookmarkedPage: (page: number) => void;
}

export default function PdfViewer({ file, bookmarkedPage, setBookmarkedPage }: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
  }

  const goToPrevPage = () => setBookmarkedPage(Math.max(1, bookmarkedPage - 1));
  const goToNextPage = () => setBookmarkedPage(Math.min(numPages!, bookmarkedPage + 1));

  return (
    <div className="w-full h-full flex flex-col items-center">
      <div className="flex-grow overflow-y-auto w-full flex justify-center">
        <Document file={file} onLoadSuccess={onDocumentLoadSuccess}>
          <Page pageNumber={bookmarkedPage} />
        </Document>
      </div>
      <div className="flex items-center justify-center p-4 bg-gray-700 rounded-b-lg w-full">
        <button onClick={goToPrevPage} disabled={bookmarkedPage <= 1} className="px-4 py-2 bg-blue-600 rounded-md disabled:bg-gray-500 mr-4">
          Previous
        </button>
        <p>Page {bookmarkedPage} of {numPages}</p>
        <button onClick={goToNextPage} disabled={bookmarkedPage >= numPages!} className="px-4 py-2 bg-blue-600 rounded-md disabled:bg-gray-500 ml-4">
          Next
        </button>
      </div>
    </div>
  );
}
