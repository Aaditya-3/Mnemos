const { useState, useEffect, useRef } = React;

// API base URL
const API_URL = '/chat';
const CHATS_API = '/chats';

// Main App Component
function App() {
    const [chats, setChats] = useState([]);
    const [currentChatId, setCurrentChatId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const messagesEndRef = useRef(null);

    // Scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Load chats on mount
    useEffect(() => {
        loadChats();
    }, []);

    // Load messages when chat changes
    useEffect(() => {
        if (currentChatId) {
            loadChatMessages(currentChatId);
        } else {
            setMessages([]);
        }
    }, [currentChatId]);

    const loadChats = async () => {
        try {
            const response = await fetch(CHATS_API);
            const data = await response.json();
            setChats(data);
            if (data.length > 0 && !currentChatId) {
                setCurrentChatId(data[0].id);
            }
        } catch (error) {
            console.error('Error loading chats:', error);
        }
    };

    const loadChatMessages = async (chatId) => {
        try {
            const response = await fetch(`${CHATS_API}/${chatId}`);
            const data = await response.json();
            setMessages(data.messages || []);
        } catch (error) {
            console.error('Error loading messages:', error);
        }
    };

    const createNewChat = async () => {
        try {
            const response = await fetch(`${CHATS_API}/new`, { method: 'POST' });
            const data = await response.json();
            setCurrentChatId(data.id);
            setMessages([]);
            await loadChats();
        } catch (error) {
            console.error('Error creating chat:', error);
        }
    };

    const deleteChat = async (chatId) => {
        try {
            await fetch(`${CHATS_API}/${chatId}`, { method: 'DELETE' });
            if (currentChatId === chatId) {
                setCurrentChatId(null);
                setMessages([]);
            }
            await loadChats();
        } catch (error) {
            console.error('Error deleting chat:', error);
        }
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading) return;

        const userMessage = inputMessage.trim();
        setInputMessage('');
        setIsLoading(true);

        // Add user message to UI immediately
        const tempUserMessage = {
            role: 'user',
            content: userMessage,
            timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, tempUserMessage]);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage,
                    chat_id: currentChatId
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to get response');
            }

            // Add assistant response
            const assistantMessage = {
                role: 'assistant',
                content: data.reply,
                timestamp: new Date().toISOString()
            };
            setMessages(prev => [...prev, assistantMessage]);

            // Update current chat ID if it was a new chat
            if (data.chat_id !== currentChatId) {
                setCurrentChatId(data.chat_id);
            }

            // Reload chats to update titles
            await loadChats();
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${error.message}`,
                timestamp: new Date().toISOString()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="flex h-screen bg-slate-900 text-slate-100">
            {/* Sidebar */}
            <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 overflow-hidden bg-slate-800 border-r border-slate-700 flex flex-col`}>
                <div className="p-4 border-b border-slate-700">
                    <button
                        onClick={createNewChat}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        New Chat
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
                    {chats.length === 0 ? (
                        <div className="text-slate-400 text-sm text-center mt-4">
                            No chats yet. Start a new conversation!
                        </div>
                    ) : (
                        chats.map((chat) => (
                            <div
                                key={chat.id}
                                className={`group p-3 mb-2 rounded-lg cursor-pointer transition-colors ${
                                    currentChatId === chat.id
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                                }`}
                                onClick={() => setCurrentChatId(chat.id)}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium truncate">{chat.title}</div>
                                        <div className={`text-xs mt-1 ${currentChatId === chat.id ? 'text-blue-100' : 'text-slate-400'}`}>
                                            {new Date(chat.updated_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            deleteChat(chat.id);
                                        }}
                                        className="ml-2 opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="bg-slate-800 border-b border-slate-700 p-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            className="text-slate-300 hover:text-white transition-colors"
                        >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        <h1 className="text-xl font-bold text-white">AI Chat with Memory</h1>
                    </div>
                    <div className="text-sm text-slate-400">
                        Powered by Google Gemini
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
                    {messages.length === 0 ? (
                        <div className="flex items-center justify-center h-full">
                            <div className="text-center text-slate-400">
                                <svg className="w-16 h-16 mx-auto mb-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                </svg>
                                <p className="text-lg font-medium mb-2">Start a conversation</p>
                                <p className="text-sm">Send a message to begin chatting with AI</p>
                            </div>
                        </div>
                    ) : (
                        <div className="max-w-4xl mx-auto space-y-6">
                            {messages.map((msg, idx) => (
                                <div
                                    key={idx}
                                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div
                                        className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                                            msg.role === 'user'
                                                ? 'bg-blue-600 text-white'
                                                : 'bg-slate-700 text-slate-100'
                                        }`}
                                    >
                                        <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-slate-700 rounded-2xl px-5 py-3">
                                        <div className="flex gap-2">
                                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <div className="bg-slate-800 border-t border-slate-700 p-4">
                    <div className="max-w-4xl mx-auto flex gap-3">
                        <input
                            type="text"
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Type your message..."
                            disabled={isLoading}
                            className="flex-1 bg-slate-700 text-white placeholder-slate-400 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                        />
                        <button
                            onClick={sendMessage}
                            disabled={isLoading || !inputMessage.trim()}
                            className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white font-semibold px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
                        >
                            {isLoading ? (
                                <>
                                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    <span>Sending...</span>
                                </>
                            ) : (
                                <>
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                                    </svg>
                                    <span>Send</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Render the app (compatible with React 17/18 CDN)
const rootElement = document.getElementById('root');
if (ReactDOM.createRoot) {
    // React 18
    const root = ReactDOM.createRoot(rootElement);
    root.render(<App />);
} else {
    // React 17
    ReactDOM.render(<App />, rootElement);
}
