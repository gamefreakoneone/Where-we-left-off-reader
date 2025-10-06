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
    <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
      <div className="bg-gray-800 p-8 rounded-lg shadow-lg text-center w-full max-w-md">
        <h1 className="text-3xl font-bold mb-4">Welcome to Project Velcro</h1>
        <p className="mb-6 text-gray-400">Upload your story book in PDF format to begin.</p>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <input
              type="file"
              onChange={handleFileChange}
              accept=".pdf"
              className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>
          <button
            type="submit"
            disabled={!file || status === 'uploading'}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg disabled:bg-gray-500 transition-colors"
          >
            {status === 'uploading' ? 'Uploading...' : 'Start Analyzing'}
          </button>
        </form>
        {status === 'failed' && <p className="text-red-500 mt-4">Upload failed. Please try again.</p>}
      </div>
    </div>
  );
}