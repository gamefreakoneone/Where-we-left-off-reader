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
      setProcessingStatus('idle');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setProcessingStatus('uploading');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://127.0.0.1:8000/books/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Upload failed with status: ' + res.status);
      }

      const data = await res.json();
      if (data.book_id) {
        onUploadSuccess(data.book_id, file);
      } else {
        throw new Error('book_id not found in response');
      }
    } catch (error) {
      console.error('Upload failed:', error);
      onUploadFailed();
    }
  };

  return (
    <div className="space-y-5">
      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="space-y-2">
          <label htmlFor="book-upload" className="text-sm font-medium text-muted">
            Choose a PDF file
          </label>
          <input
            id="book-upload"
            type="file"
            onChange={handleFileChange}
            accept=".pdf"
            className="block w-full cursor-pointer rounded-2xl border border-third-color bg-first-color px-4 py-3 text-sm text-muted focus:border-fourth-color focus:outline-none focus:ring-2 focus:ring-fourth-color file:mr-4 file:rounded-xl file:border-none file:bg-fourth-color file:px-4 file:py-2 file:font-semibold file:text-first-color file:transition file:duration-200 file:hover:brightness-105"
          />
          {file && (
            <p className="text-xs text-muted">
              Selected file: <span className="text-white">{file.name}</span>
            </p>
          )}
        </div>

        <button
          type="submit"
          disabled={!file || status === 'uploading'}
          className="w-full rounded-2xl bg-fourth-color px-4 py-3 text-sm font-semibold uppercase tracking-wide text-first-color transition-all duration-200 hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {status === 'uploading' ? 'Uploading…' : 'Start analyzing'}
        </button>
      </form>

      {status === 'failed' && <p className="text-sm text-fourth-color">Upload failed. Please try again.</p>}
    </div>
  );
}
