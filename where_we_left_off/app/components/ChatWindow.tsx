"use client";

import { useState } from 'react';

interface ChatWindowProps {
  bookId: string;
  bookmarkedPage: number;
  onNewGraphData: (data: any) => void;
}

interface Message {
  sender: 'user' | 'ai';
  text: string;
}

export default function ChatWindow({ bookId, bookmarkedPage, onNewGraphData }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`http://127.0.0.1:8000/chat/${bookId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input, curr_page: bookmarkedPage }),
      });

      if (!res.ok) throw new Error("Failed to get response from agent.");

      const data = await res.json();
      const aiMessage: Message = { sender: 'ai', text: data.answer };
      setMessages(prev => [...prev, aiMessage]);

      // Check if the response contains graph data and update central state
      try {
        const potentialGraphData = JSON.parse(data.answer);
        // If it's a valid JSON object, assume it's graph data
        if (typeof potentialGraphData === 'object' && potentialGraphData !== null) {
          onNewGraphData(potentialGraphData);
        }
      } catch (jsonError) {
        // It's not JSON, so it's just a regular text message. Do nothing.
      }

    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: Message = { sender: 'ai', text: "Sorry, I couldn't get a response. Please try again." };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full p-4">
      <h2 className="text-xl font-semibold mb-4 text-center">Chat with the Story</h2>
      <div className="flex-grow overflow-y-auto mb-4 p-4 bg-second-color rounded-md">
        <div className="flex flex-col space-y-4">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`max-w-xs rounded-lg border p-3 text-sm leading-relaxed shadow-sm ${
                msg.sender === 'user'
                  ? 'self-end border-third-color bg-third-color/20 text-first-color'
                  : 'self-start border-fourth-color bg-fourth-color/20 text-first-color'
              }`}
            >
              {msg.text}
            </div>
          ))}
          {isLoading && (
            <div className="max-w-xs self-start rounded-lg border border-fourth-color bg-fourth-color/20 p-3 text-sm leading-relaxed text-first-color shadow-sm">
              Thinking...
            </div>
          )}
        </div>
      </div>
      <form onSubmit={handleSendMessage} className="flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the story..."
          className="flex-grow bg-second-color border border-third-color rounded-l-lg p-2 focus:outline-none focus:ring-2 focus:ring-fourth-color text-white"
        />
        <button
          type="submit"
          disabled={isLoading}
          className="bg-fourth-color hover:bg-third-color text-first-color font-bold py-2 px-4 rounded-r-lg disabled:bg-third-color"
        >
          Send
        </button>
      </form>
    </div>
  );
}
