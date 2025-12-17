'use client';

import { useState, useEffect, useRef } from 'react';

export default function Dashboard() {
  const [token, setToken] = useState(null);
  const [messages, setMessages] = useState([]);
  const [activity, setActivity] = useState([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [activityMinimized, setActivityMinimized] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [readmeContent, setReadmeContent] = useState('');
  const messagesEndRef = useRef(null);
  const messageIdRef = useRef(0);
  const activityIdRef = useRef(0);

  // Auto-login on mount
  useEffect(() => {
    const login = async () => {
      try {
        const res = await fetch('/api/login', { method: 'POST' });
        const data = await res.json();
        setToken(data.token);
      } catch (err) {
        console.error('Login failed:', err);
      }
    };
    login();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch README content on mount
  useEffect(() => {
    const fetchReadme = async () => {
      try {
        const response = await fetch('/HOLOGRAM_README.md');
        const text = await response.text();
        setReadmeContent(text);
      } catch (err) {
        console.error('Failed to load README:', err);
        setReadmeContent('# Error\n\nFailed to load README content.');
      }
    };
    fetchReadme();
  }, []);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || !token || isThinking) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { id: ++messageIdRef.current, role: 'user', content: userMessage }]);
    setIsThinking(true);

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) {
        throw new Error('Chat request failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            const eventType = line.slice(7);
            continue;
          }
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              handleSSEEvent(data);
            } catch (err) {
              console.error('Parse error:', err);
            }
          }
        }
      }
    } catch (err) {
      console.error('Chat error:', err);
      setMessages((prev) => [
        ...prev,
        { id: ++messageIdRef.current, role: 'error', content: 'Connection error. Please try again.' },
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleSSEEvent = (event) => {
    // SSE events have event type in a preceding line, but we receive data as JSON
    // The event type is embedded in our data structure
    if (event.status === 'processing') {
      // Thinking event
      return;
    }
    if (event.content !== undefined) {
      // Response event
      setMessages((prev) => [...prev, { id: ++messageIdRef.current, role: 'assistant', content: event.content }]);
      return;
    }
    if (event.type) {
      // Activity event
      setActivity((prev) => [
        { ...event, timestamp: new Date().toISOString(), id: ++activityIdRef.current },
        ...prev.slice(0, 49), // Keep last 50
      ]);
    }
    if (event.message === 'Service error') {
      setMessages((prev) => [
        ...prev,
        { id: ++messageIdRef.current, role: 'error', content: 'Service error. Please try again.' },
      ]);
    }
  };

  const formatActivity = (item) => {
    switch (item.type) {
      case 'intent':
        return `Intent: ${item.intent} (${(item.confidence * 100).toFixed(0)}%)`;
      case 'fact':
        return `Stored: ${item.subject} --${item.predicate}--> ${item.object}`;
      case 'consolidation':
        return `Neural: Consolidated ${item.facts_count} facts`;
      default:
        return JSON.stringify(item);
    }
  };

  const renderMarkdown = (markdown) => {
    const lines = markdown.split('\n');
    const elements = [];
    let i = 0;
    let keyCounter = 0; // Unique key counter

    while (i < lines.length) {
      const line = lines[i];
      const uniqueKey = `md-${keyCounter++}`;

      // Headers
      if (line.startsWith('# ')) {
        elements.push(<h1 key={uniqueKey} className="text-2xl font-bold text-purple-300 mb-4 mt-6">{line.slice(2)}</h1>);
      } else if (line.startsWith('## ')) {
        elements.push(<h2 key={uniqueKey} className="text-xl font-semibold text-purple-300 mb-3 mt-5">{line.slice(3)}</h2>);
      } else if (line.startsWith('### ')) {
        elements.push(<h3 key={uniqueKey} className="text-lg font-semibold text-purple-400 mb-2 mt-4">{line.slice(4)}</h3>);
      }
      // Horizontal rule
      else if (line.trim() === '---') {
        elements.push(<hr key={uniqueKey} className="border-purple-500/30 my-4" />);
      }
      // Bold text (inline)
      else if (line.includes('**')) {
        const formatted = line.split('**').map((part, idx) =>
          idx % 2 === 1 ? <strong key={`${uniqueKey}-${idx}`} className="text-purple-300 font-semibold">{part}</strong> : part
        );
        elements.push(<p key={uniqueKey} className="mb-2 text-gray-300">{formatted}</p>);
      }
      // List items
      else if (line.match(/^[\d]+\.\s/)) {
        elements.push(<li key={uniqueKey} className="ml-6 mb-1 text-gray-300">{line.replace(/^[\d]+\.\s/, '')}</li>);
      } else if (line.startsWith('- ')) {
        elements.push(<li key={uniqueKey} className="ml-6 mb-1 text-gray-300 list-disc">{line.slice(2)}</li>);
      }
      // Code blocks
      else if (line.startsWith('```')) {
        const codeLines = [];
        i++;
        while (i < lines.length && !lines[i].startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        elements.push(
          <pre key={uniqueKey} className="bg-black/40 border border-purple-500/20 rounded p-3 my-2 overflow-x-auto">
            <code className="text-xs text-purple-200">{codeLines.join('\n')}</code>
          </pre>
        );
      }
      // Table detection
      else if (line.startsWith('|')) {
        const tableLines = [];
        while (i < lines.length && lines[i].startsWith('|')) {
          tableLines.push(lines[i]);
          i++;
        }
        const rows = tableLines.map(l => l.split('|').filter(Boolean).map(c => c.trim()));
        elements.push(
          <table key={uniqueKey} className="w-full border-collapse my-3">
            <thead>
              <tr className="border-b border-purple-500/30">
                {rows[0].map((h, idx) => <th key={`${uniqueKey}-h-${idx}`} className="text-left p-2 text-purple-300 font-semibold">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.slice(2).map((row, ridx) => (
                <tr key={`${uniqueKey}-r-${ridx}`} className="border-b border-purple-500/10">
                  {row.map((cell, cidx) => <td key={`${uniqueKey}-r-${ridx}-c-${cidx}`} className="p-2 text-gray-300 text-sm">{cell}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        );
        i--;
      }
      // Regular paragraphs
      else if (line.trim()) {
        elements.push(<p key={uniqueKey} className="mb-2 text-gray-300">{line}</p>);
      }
      // Empty lines
      else {
        elements.push(<div key={uniqueKey} className="h-2" />);
      }

      i++;
    }

    return elements;
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-950 via-gray-950 to-black p-4 flex items-center justify-center">
      <div className="container mx-auto max-w-5xl w-full max-h-[calc(100vh-2rem)]">
        <div className="backdrop-blur-xl bg-purple-950/30 rounded-2xl border border-purple-500/20 shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 bg-black/30 border-b border-purple-500/20">
            <h1 className="text-xl font-semibold text-purple-200 flex items-center gap-3">
              <span className="text-2xl">🧠</span>
              Kent Hologram
            </h1>
            <div className="flex items-center gap-2">
              <span
                className={`status-badge ${isThinking ? 'active' : 'idle'}`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    isThinking ? 'bg-purple-400 animate-pulse' : 'bg-gray-500'
                  }`}
                />
                {isThinking ? 'Thinking' : 'Ready'}
              </span>
              <button
                onClick={() => setShowDebug(!showDebug)}
                className={`status-badge ${showDebug ? 'active' : 'idle'} cursor-pointer transition-all hover:bg-purple-500/20`}
              >
                Debug {showDebug ? '\u25B2' : '\u25BC'}
              </button>
            </div>
          </div>

          {/* Terminal */}
          <div className="terminal flex flex-col h-[calc(100vh-8rem)]">
          {/* Terminal Header */}
          <div className="terminal-header flex-shrink-0 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <span className="terminal-dot red" />
                <span className="terminal-dot yellow" />
                <span className="terminal-dot green" />
              </div>
              <span className="text-xs text-[var(--text-secondary)]">
                hologram@dashboard
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setActiveTab('chat')}
                className={`text-xs px-3 py-1 rounded transition-colors ${
                  activeTab === 'chat'
                    ? 'text-purple-300 bg-purple-500/20 border border-purple-500/50'
                    : 'text-[var(--text-secondary)] hover:text-purple-300'
                }`}
              >
                Chat
              </button>
              <button
                onClick={() => setActiveTab('readme')}
                className={`text-xs px-3 py-1 rounded transition-colors ${
                  activeTab === 'readme'
                    ? 'text-purple-300 bg-purple-500/20 border border-purple-500/50'
                    : 'text-[var(--text-secondary)] hover:text-purple-300'
                }`}
              >
                README
              </button>
            </div>
          </div>

          {/* Activity Panel */}
          {activity.length > 0 && (
            <div className="activity-panel flex-shrink-0">
              <div className="flex items-center justify-between px-3 py-2 text-xs text-[var(--text-secondary)] border-b border-[var(--bg-secondary)]">
                <span>HDC Activity</span>
                <button
                  onClick={() => setActivityMinimized(!activityMinimized)}
                  className="cursor-pointer hover:text-purple-400 transition-colors"
                  title={activityMinimized ? 'Expand' : 'Minimize'}
                >
                  {activityMinimized ? '\u25B2' : '\u25BC'}
                </button>
              </div>
              {!activityMinimized ? (
                activity.slice(0, showDebug ? 20 : 5).map((item) => (
                  <div
                    key={item.id}
                    className={`activity-item ${item.type} fade-in`}
                  >
                    {showDebug ? (
                      <code className="text-xs">{JSON.stringify(item)}</code>
                    ) : (
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2 flex-1">
                          <span className="text-[10px] text-[var(--text-secondary)] flex-shrink-0">
                            {new Date(item.timestamp).toLocaleTimeString()}
                          </span>
                          <span>{formatActivity(item)}</span>
                        </div>
                        {item.confidence !== undefined && (
                          <div className="confidence-bar flex-shrink-0" style={{ width: '60px' }}>
                            <div
                              className="confidence-fill"
                              style={{ width: `${item.confidence * 100}%` }}
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="activity-item fade-in">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 flex-1">
                      <span className="text-[10px] text-[var(--text-secondary)] flex-shrink-0">
                        {new Date(activity[0].timestamp).toLocaleTimeString()}
                      </span>
                      <span>{formatActivity(activity[0])}</span>
                    </div>
                    {activity[0].confidence !== undefined && (
                      <div className="confidence-bar flex-shrink-0" style={{ width: '60px' }}>
                        <div
                          className="confidence-fill"
                          style={{ width: `${activity[0].confidence * 100}%` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 min-h-0">
            {activeTab === 'chat' ? (
              <>
                {messages.length === 0 && (
                  <div className="text-[var(--text-secondary)] text-sm">
                    <p>Welcome to Kent Hologram Dashboard.</p>
                    <p className="mt-2">
                      Type a message to start chatting. Activity events will appear
                      above as the HDC processes your input.
                    </p>
                  </div>
                )}
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`message ${msg.role} fade-in`}
                  >
                    <span className="text-[var(--text-secondary)] mr-2">
                      {msg.role === 'user' ? '>' : '#'}
                    </span>
                    {msg.content}
                  </div>
                ))}
                {isThinking && (
                  <div className="message thinking">
                    <span className="text-[var(--text-secondary)] mr-2">#</span>
                    Processing...
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            ) : (
              <div className="readme-content text-sm">
                {readmeContent ? renderMarkdown(readmeContent) : (
                  <div className="text-[var(--text-secondary)]">
                    <p>Loading README...</p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input */}
          {activeTab === 'chat' && (
            <form onSubmit={sendMessage} className="flex-shrink-0">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={token ? 'Type a message...' : 'Connecting...'}
                disabled={!token || isThinking}
                className="chat-input"
              />
            </form>
          )}
        </div>

          {/* Footer */}
          <div className="px-6 py-3 bg-black/20 border-t border-purple-500/20 text-center text-xs text-purple-300/70">
            Holographic Memory System | HDC + Neural Consolidation
          </div>
        </div>
      </div>
    </main>
  );
}
