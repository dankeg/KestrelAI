import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Copy, ThumbsUp, ThumbsDown, RotateCcw, ArrowLeft } from "lucide-react";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: number;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue.trim(),
      role: "user",
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "This is a mockup response. In a real implementation, this would be connected to your LLM backend. The interface is designed to match modern LLM chat applications like ChatGPT and Claude, with a clean and minimal design.",
        role: "assistant",
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue]);

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 px-4 py-3">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div className="flex items-center gap-3">
            <a 
              href="/" 
              className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
              title="Back to main app"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </a>
            <h1 className="text-lg font-semibold text-gray-900">KestrelAI</h1>
          </div>
          <button className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
            <RotateCcw className="w-4 h-4 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">How can I help you today?</h2>
              <p className="text-gray-600">Start a conversation with KestrelAI</p>
            </div>
          )}
          
          {messages.map((message) => (
            <div key={message.id} className="mb-8">
              <div className="flex gap-4">
                <div className="flex-shrink-0">
                  {message.role === "user" ? (
                    <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center">
                      <span className="text-white text-sm font-medium">U</span>
                    </div>
                  ) : (
                    <div className="w-8 h-8 bg-gradient-to-r from-amber-500 to-orange-500 rounded-full flex items-center justify-center">
                      <span className="text-white text-sm font-medium">K</span>
                    </div>
                  )}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-semibold text-gray-900">
                      {message.role === "user" ? "You" : "KestrelAI"}
                    </span>
                  </div>
                  
                  <div className="prose prose-sm max-w-none text-gray-900 leading-relaxed">
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  </div>
                  
                  {message.role === "assistant" && (
                    <div className="flex items-center gap-2 mt-3 opacity-0 hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => copyToClipboard(message.content)}
                        className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
                        title="Copy"
                      >
                        <Copy className="w-4 h-4 text-gray-500" />
                      </button>
                      <button
                        className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
                        title="Good response"
                      >
                        <ThumbsUp className="w-4 h-4 text-gray-500" />
                      </button>
                      <button
                        className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
                        title="Poor response"
                      >
                        <ThumbsDown className="w-4 h-4 text-gray-500" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {isLoading && (
            <div className="mb-8">
              <div className="flex gap-4">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-gradient-to-r from-amber-500 to-orange-500 rounded-full flex items-center justify-center">
                    <span className="text-white text-sm font-medium">K</span>
                  </div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-semibold text-gray-900">KestrelAI</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm text-gray-500">Thinking...</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 bg-white">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyPress={handleKeyPress}
              placeholder="Message KestrelAI..."
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent text-sm leading-relaxed"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
            <button
              onClick={handleSend}
              disabled={!inputValue.trim() || isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
          <div className="mt-2 text-xs text-gray-500 text-center">
            KestrelAI can make mistakes. Consider checking important information.
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
