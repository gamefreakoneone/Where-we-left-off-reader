"use client";

import { useState, useEffect, useRef } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

interface PdfViewerProps {
  file: File | string; // Accept either a File object or a URL string
  bookmarkedPage: number;
  setBookmarkedPage: (page: number) => void;
}

export default function PdfViewer({
  file,
  bookmarkedPage,
  setBookmarkedPage,
}: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [isReady, setIsReady] = useState(false);
  // const [scale, setScale] = useState(1.0);
  const [pageWidth, setPageWidth] = useState<number | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // According to React-PDF documentation, use CDN with proper version
    pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
    setIsReady(true);
  }, []);

  // useEffect(() => {
  //   if (containerRef.current) {
  //     // Adjust scale based on container width when it changes
  //     const adjustScale = () => {
  //       const containerWidth = containerRef.current?.clientWidth || 0;
  //       // Limit width to a reasonable size for readability
  //       const maxWidth = Math.min(containerWidth * 0.9, 800);
  //       setScale(maxWidth / 600); // 600 is approximate standard PDF page width
  //     };

  //     adjustScale();
  //     window.addEventListener('resize', adjustScale);
  //     return () => window.removeEventListener('resize', adjustScale);
  //   }
  // }, []);

  useEffect(() => {
    if (containerRef.current) {
      const updateWidth = () => {
        const containerWidth = containerRef.current?.clientWidth || 0;
        setPageWidth(Math.min(containerWidth * 0.9, 1000));
      };

      // updateWidth();
      // window.addEventListener("resize", updateWidth);
      setTimeout(updateWidth, 0);
    window.addEventListener("resize", updateWidth);
      return () => window.removeEventListener("resize", updateWidth);
    }
  }, []);

  // function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
  //   setNumPages(numPages);
  // }

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
  setNumPages(numPages);
  // Recalculate width after PDF loads
  if (containerRef.current) {
    const containerWidth = containerRef.current.clientWidth || 0;
    setPageWidth(Math.min(containerWidth * 0.9, 1000));
  }
}

  const goToPrevPage = () => setBookmarkedPage(Math.max(1, bookmarkedPage - 1));
  const goToNextPage = () =>
    setBookmarkedPage(Math.min(numPages!, bookmarkedPage + 1));

  if (!isReady) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-white">Loading PDF viewer...</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div
        ref={containerRef}
        className="flex-grow overflow-y-auto w-full flex justify-center items-center p-4"
      >
        <Document
          file={file}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={<div className="text-white">Loading PDF...</div>}
          error={<div className="text-red-500">Error loading PDF</div>}
        >
          {/* <Page 
            pageNumber={bookmarkedPage} 
            scale={scale}
            className="max-w-full h-auto"
          /> */}
          <Page pageNumber={bookmarkedPage} width={pageWidth} />
        </Document>
      </div>
      <div className="flex items-center justify-center p-4 bg-second-color rounded-b-lg w-full">
        <button
          onClick={goToPrevPage}
          disabled={bookmarkedPage <= 1}
          className="px-4 py-2 bg-fourth-color rounded-md disabled:bg-third-color mr-4 text-first-color"
        >
          Previous
        </button>
        <p className="text-white">
          Page {bookmarkedPage} of {numPages}
        </p>
        <button
          onClick={goToNextPage}
          disabled={bookmarkedPage >= numPages!}
          className="px-4 py-2 bg-fourth-color rounded-md disabled:bg-third-color ml-4 text-first-color"
        >
          Next
        </button>
      </div>
    </div>
  );
}
