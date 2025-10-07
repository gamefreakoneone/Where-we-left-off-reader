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
    <div className="flex flex-col h-full p-4 text-[var(--fourth-color)]">
        <h2 className="text-xl font-semibold mb-4 text-center">Chat with the Story</h2>
        <div className="flex-grow overflow-y-auto mb-4 p-2 rounded-md bg-[color:rgba(30,41,59,0.7)] backdrop-blur-sm">
            {messages.map((msg, index) => (
            <div key={index} className={`chat ${msg.sender === 'user' ? 'chat-end' : 'chat-start'}`}>
                <div className={`chat-bubble text-sm md:text-base ${msg.sender === 'user' ? 'bg-[var(--third-color)] text-[var(--first-color)]' : 'bg-[color:rgba(56,189,248,0.15)] text-[var(--fourth-color)]'} p-3 rounded-lg max-w-xs`}>
                {msg.text}
                </div>
            </div>
            ))}
            {isLoading && (
                <div className="chat chat-start">
                    <div className="chat-bubble bg-[color:rgba(56,189,248,0.15)] text-[var(--fourth-color)] p-3 rounded-lg max-w-xs">
                        Thinking...
                    </div>
                </div>
            )}
        </div>
        <form onSubmit={handleSendMessage} className="flex">
            <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the story..."
            className="flex-grow bg-[color:rgba(15,23,42,0.6)] border border-[color:rgba(56,189,248,0.4)] rounded-l-lg p-2 focus:outline-none focus:ring-2 focus:ring-[var(--third-color)] text-[var(--fourth-color)] placeholder:text-[color:rgba(248,250,252,0.5)]"
            />
            <button type="submit" disabled={isLoading} className="bg-[var(--third-color)] text-[var(--first-color)] font-bold py-2 px-4 rounded-r-lg disabled:opacity-60 transition-transform hover:scale-[1.02] disabled:hover:scale-100">
            Send
            </button>
        </form>
    </div>
  );
}
