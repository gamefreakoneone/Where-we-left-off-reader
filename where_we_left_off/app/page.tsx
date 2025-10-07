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
    router.push(`/books/${bookId}`);
  };

  const handleUploadFailed = () => {
    setUploadStatus('failed');
  };

  return (
    <div className="min-h-screen bg-first-color text-white">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-10">
        <header className="mb-10 space-y-4 border-b border-third-color pb-6">
          <span className="text-sm font-semibold uppercase tracking-[0.35em] text-fourth-color">
            Project Velcro
          </span>
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <h1 className="text-4xl font-semibold leading-tight md:text-5xl">Where we left off</h1>
            <p className="max-w-xl text-sm text-muted md:text-base">
              Keep your reading progress synced and drop in fresh stories whenever you are ready to explore new worlds.
            </p>
          </div>
        </header>

        <main className="grid flex-1 gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <section className="flex flex-col rounded-3xl bg-second-color p-6 shadow-xl shadow-black/30">
            <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-2xl font-semibold">Your Books</h2>
              {!isLoading && !error && books.length > 0 && (
                <span className="text-xs font-medium uppercase tracking-[0.25em] text-muted">
                  {books.length} {books.length === 1 ? 'book' : 'books'}
                </span>
              )}
            </div>

            {isLoading && <p className="text-muted">Loading books...</p>}
            {error && <p className="text-fourth-color">Error: {error}</p>}

            {!isLoading && !error && books.length === 0 && (
              <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-third-color p-8 text-center text-muted">
                <p>No books yet. Upload a PDF to start your next adventure.</p>
              </div>
            )}

            {!isLoading && !error && books.length > 0 && (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {books.map((book) => {
                  const statusClass =
                    book.status === 'complete'
                      ? 'bg-fourth-color text-first-color'
                      : 'border border-fourth-color text-fourth-color';

                  return (
                    <Link href={`/books/${book.book_id}`} key={book.book_id} className="group block h-full">
                      <div className="flex h-full flex-col justify-between rounded-2xl border border-transparent bg-first-color p-5 transition-all duration-200 hover:-translate-y-1 hover:border-fourth-color hover:bg-third-color hover:shadow-lg hover:shadow-black/40">
                        <div className="space-y-3">
                          <h3 className="text-lg font-semibold leading-tight group-hover:text-fourth-color">
                            {book.filename.replace(/\.pdf$/i, '')}
                          </h3>
                          <p className="text-sm text-muted">Tap to resume reading</p>
                        </div>
                        <span
                          className={`mt-4 inline-flex items-center justify-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${statusClass}`}
                        >
                          {book.status}
                        </span>
                      </div>
                    </Link>
                  );
                })}
              </div>
            )}
          </section>

          <aside className="rounded-3xl bg-second-color p-6 shadow-xl shadow-black/30">
            <div className="mb-6 space-y-2">
              <h2 className="text-2xl font-semibold">Upload a new book</h2>
              <p className="text-sm text-muted">
                Drop in a PDF file to sync it with your library and pick up right where you left off.
              </p>
            </div>
            <FileUpload
              onUploadSuccess={handleUploadSuccess}
              onUploadFailed={handleUploadFailed}
              setProcessingStatus={setUploadStatus}
              status={uploadStatus}
            />
          </aside>
        </main>
      </div>
    </div>
  );
}
