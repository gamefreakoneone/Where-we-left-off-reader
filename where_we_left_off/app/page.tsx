"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import FileUpload from './components/FileUpload';
import { useRouter } from 'next/navigation';

interface Book {
  book_id: string;
  filename: string;
  status: string;
}

export default function HomePage() {
  const [books, setBooks] = useState<Book[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const router = useRouter();

  useEffect(() => {
    const fetchBooks = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/books');
        if (!res.ok) throw new Error('Failed to fetch books');
        const data = await res.json();
        setBooks(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchBooks();
  }, []);

  const handleUploadSuccess = (bookId: string, file: File) => {
    // Navigate to the new book's reader page
    router.push(`/books/${bookId}`);
  };

  const handleUploadFailed = () => {
    setUploadStatus('failed');
  };

  return (
    <div className="min-h-screen bg-[var(--first-color)] text-[var(--fourth-color)] p-8">
      <div className="max-w-4xl mx-auto space-y-10">
        <h1 className="text-4xl font-bold text-center mb-4 text-[var(--fourth-color)]">Project Velcro Library</h1>

        <div className="bg-[var(--second-color)] p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4">Upload New Book</h2>
            <FileUpload
                onUploadSuccess={handleUploadSuccess}
                onUploadFailed={handleUploadFailed}
                setProcessingStatus={setUploadStatus}
                status={uploadStatus}
            />
        </div>

        <div>
          <h2 className="text-2xl font-semibold mb-4">Your Books</h2>
          {isLoading && <p>Loading books...</p>}
          {error && <p className="text-red-500">Error: {error}</p>}
          {!isLoading && !error && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {books.map((book) => (
                <Link href={`/books/${book.book_id}`} key={book.book_id}>
                  <div className="block bg-[var(--second-color)] p-6 rounded-lg hover:bg-[color:rgba(30,41,59,0.85)] transition-colors cursor-pointer h-full flex flex-col justify-between border border-transparent hover:border-[color:rgba(56,189,248,0.4)]">
                    <h3 className="text-xl font-semibold mb-2 truncate">{book.filename.replace(/\.pdf$/i, '')}</h3>
                    <span
                      className={`text-sm font-medium px-3 py-1 rounded-full self-start ${
                        book.status === 'complete'
                          ? 'bg-[var(--third-color)] text-[var(--first-color)]'
                          : 'bg-[color:rgba(56,189,248,0.15)] text-[var(--third-color)] border border-[color:rgba(56,189,248,0.4)]'
                      }`}
                    >
                      {book.status}
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
