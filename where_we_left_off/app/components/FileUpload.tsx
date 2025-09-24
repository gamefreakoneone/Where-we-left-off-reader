"use client";

import { useState } from 'react';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  status: string;
}

export default function FileUpload({ onFileUpload, status }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (file) {
      onFileUpload(file);
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
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg disabled:bg-gray-500"
          >
            {status === 'uploading' ? 'Uploading...' : 'Start Reading'}
          </button>
        </form>
        {status === 'failed' && <p className="text-red-500 mt-4">Upload failed. Please try again.</p>}
      </div>
    </div>
  );
}
