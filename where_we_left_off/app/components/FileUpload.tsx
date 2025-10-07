"use client";

import { useState } from 'react';

interface FileUploadProps {
  onUploadSuccess: (bookId: string, file: File) => void;
  onUploadFailed: () => void;
  setProcessingStatus: (status: string) => void;
  status: string;
}

export default function FileUpload({ onUploadSuccess, onUploadFailed, setProcessingStatus, status }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setProcessingStatus("uploading");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/books/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Upload failed with status: " + res.status);
      }

      const data = await res.json();
      if (data.book_id) {
        onUploadSuccess(data.book_id, file);
      } else {
        throw new Error("book_id not found in response");
      }
    } catch (error) {
      console.error("Upload failed:", error);
      onUploadFailed();
    }
  };

  return (
    <div className="w-full text-[var(--fourth-color)]">
      <div className="mx-auto max-w-xl space-y-6 text-center">
        <h1 className="text-3xl font-bold">Welcome to Project Velcro</h1>
        <p className="text-[color:rgba(248,250,252,0.7)]">Upload your story book in PDF format to begin.</p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="file"
            onChange={handleFileChange}
            accept=".pdf"
            className="w-full text-sm text-[color:rgba(248,250,252,0.75)] file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-[color:rgba(56,189,248,0.15)] file:text-[var(--third-color)] hover:file:bg-[color:rgba(56,189,248,0.25)]"
          />
          <button
            type="submit"
            disabled={!file || status === 'uploading'}
            className="w-full bg-[var(--third-color)] text-[var(--first-color)] font-bold py-2 px-4 rounded-lg transition-transform hover:scale-[1.02] disabled:opacity-60 disabled:hover:scale-100"
          >
            {status === 'uploading' ? 'Uploading...' : 'Start Analyzing'}
          </button>
        </form>
        {status === 'failed' && <p className="text-red-400">Upload failed. Please try again.</p>}
      </div>
    </div>
  );
}