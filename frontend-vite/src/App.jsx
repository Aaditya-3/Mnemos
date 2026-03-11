import { useEffect, useRef, useState } from "react";

const API_STREAM_URL = "/chat/stream";
const THEME_KEY = "mnemos_theme";

function resolveInitialTheme() {
  const storedTheme = localStorage.getItem(THEME_KEY);
  if (storedTheme === "dark" || storedTheme === "light") {
    return storedTheme;
  }
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

function parseSseBlock(block) {
  if (!block) return null;
  const lines = block.split("\n");
  let event = "message";
  const data = [];
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  const raw = data.join("\n");
  let payload = {};
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = { text: raw };
    }
  }
  return { event, payload };
}

export default function App() {
  const CHAR_DELAY_MS = 18;
  const [theme, setTheme] = useState(resolveInitialTheme);
  const [userId, setUserId] = useState(localStorage.getItem("user_id") || "demo");
  const [text, setText] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const endRef = useRef(null);
  const typingQueueRef = useRef("");
  const typingTimerRef = useRef(null);
  const typingDoneRef = useRef(false);
  const typingMessageIdRef = useRef("");

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    return () => {
      if (typingTimerRef.current) {
        clearInterval(typingTimerRef.current);
      }
    };
  }, []);

  const clearTypingTimer = () => {
    if (typingTimerRef.current) {
      clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }
  };

  const pumpTyping = () => {
    if (typingTimerRef.current) return;
    typingTimerRef.current = setInterval(() => {
      const messageId = typingMessageIdRef.current;
      if (!messageId) {
        clearTypingTimer();
        return;
      }
      if (!typingQueueRef.current.length) {
        if (typingDoneRef.current) clearTypingTimer();
        else clearTypingTimer();
        return;
      }
      const nextChar = typingQueueRef.current.slice(0, 1);
      typingQueueRef.current = typingQueueRef.current.slice(1);
      setWaiting(false);
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, content: `${m.content || ""}${nextChar}` } : m))
      );
    }, CHAR_DELAY_MS);
  };

  const startTyping = (messageId) => {
    typingMessageIdRef.current = messageId;
    typingQueueRef.current = "";
    typingDoneRef.current = false;
    clearTypingTimer();
  };

  const queueTyping = (messageId, textChunk) => {
    if (!textChunk) return;
    if (typingMessageIdRef.current !== messageId) typingMessageIdRef.current = messageId;
    typingQueueRef.current += textChunk;
    pumpTyping();
  };

  const finishTyping = () => {
    typingDoneRef.current = true;
    pumpTyping();
  };

  const waitTypingDrain = async () => {
    while (typingQueueRef.current.length > 0 || typingTimerRef.current) {
      await new Promise((resolve) => setTimeout(resolve, CHAR_DELAY_MS));
    }
  };

  const headers = (json = false) => {
    const h = { "X-User-ID": userId };
    if (json) h["Content-Type"] = "application/json";
    return h;
  };

  const send = async () => {
    if (!text.trim() || loading) return;
    const msg = text.trim();
    setText("");
    setLoading(true);
    setWaiting(true);
    const assistantId = `a-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: msg },
      { id: assistantId, role: "assistant", content: "" }
    ]);
    startTyping(assistantId);
    try {
      const r = await fetch(API_STREAM_URL, {
        method: "POST",
        headers: headers(true),
        body: JSON.stringify({ message: msg })
      });
      if (!r.ok || !r.body) {
        const data = await r.json();
        throw new Error(data.detail || data.error || "Request failed");
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = "";
      while (!done) {
        const read = await reader.read();
        done = read.done;
        buffer += decoder.decode(read.value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          const evt = parseSseBlock(block);
          if (!evt) continue;
          if (evt.event === "tool_call") {
            const toolName = evt.payload?.tool_call?.name || evt.payload?.tool || "tool";
            queueTyping(assistantId, `\n[tool:${toolName}] ${JSON.stringify(evt.payload)}`);
          } else if (evt.event === "token") {
            const token = evt.payload?.text || "";
            queueTyping(assistantId, token);
          } else if (evt.event === "done") {
            finishTyping();
          } else if (evt.event === "error") {
            throw new Error(evt.payload?.message || "Stream error");
          }
        }
      }
      finishTyping();
      await waitTypingDrain();
    } catch (err) {
      startTyping(assistantId);
      queueTyping(assistantId, `Error: ${err.message}`);
      finishTyping();
      await waitTypingDrain();
    } finally {
      setLoading(false);
      setWaiting(false);
      clearTypingTimer();
      typingQueueRef.current = "";
      typingDoneRef.current = false;
      typingMessageIdRef.current = "";
    }
  };

  return (
    <div className="min-h-screen bg-[rgb(var(--bg-rgb))] text-[rgb(var(--text-rgb))] p-4 md:p-6">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_8%_10%,rgb(var(--glow-b-rgb)/0.14),transparent_40%),radial-gradient(circle_at_90%_12%,rgb(var(--glow-a-rgb)/0.16),transparent_38%)]" />
      <div className="relative mx-auto max-w-4xl">
        <section className="rounded-3xl border border-[rgb(var(--surface-border-rgb)/0.35)] bg-[rgb(var(--panel-rgb)/0.82)] p-4 md:p-5 backdrop-blur-xl shadow-[0_22px_52px_rgb(var(--shadow-rgb)/0.2)]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[rgb(var(--surface-border-rgb)/0.3)] pb-3 mb-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.16em] text-[rgb(var(--accent-rgb))]">Mnemos</p>
              <h1 className="text-xl font-semibold tracking-tight">Vite Workspace</h1>
              <p className="text-[11px] uppercase tracking-[0.14em] text-[rgb(var(--text-muted-rgb))]">Subtle UI Theme</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
                className="rounded-lg border border-[rgb(var(--surface-border-rgb)/0.45)] bg-[rgb(var(--bg-alt-rgb)/0.85)] px-3 py-1.5 text-sm font-medium transition-colors hover:bg-[rgb(var(--panel-2-rgb))]"
              >
                {theme === "dark" ? "Light Mode" : "Dark Mode"}
              </button>
              <input
                value={userId}
                onChange={(e) => {
                  setUserId(e.target.value);
                  localStorage.setItem("user_id", e.target.value);
                }}
                className="w-[130px] bg-[rgb(var(--bg-alt-rgb)/0.86)] border border-[rgb(var(--surface-border-rgb)/0.42)] rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[rgb(var(--accent-rgb)/0.55)]"
              />
            </div>
          </div>
          <div className="h-[60vh] overflow-y-auto custom-scrollbar space-y-3 pr-1">
            {messages.map((m, i) =>
              m.role !== "user" &&
              !m.content &&
              waiting &&
              m.id === typingMessageIdRef.current ? null : (
                <div
                  key={m.id || i}
                  className={`rounded-2xl border px-4 py-2.5 ${
                    m.role === "user"
                      ? "bg-[linear-gradient(145deg,rgb(var(--accent-rgb)/0.95),rgb(var(--accent-soft-rgb)/0.88))] text-[rgb(var(--on-accent-rgb))] border-[rgb(var(--accent-rgb)/0.62)] shadow-[0_10px_20px_rgb(var(--shadow-rgb)/0.14)]"
                      : "bg-[rgb(var(--panel-2-rgb)/0.92)] border-[rgb(var(--surface-border-rgb)/0.3)] shadow-[0_6px_14px_rgb(var(--shadow-rgb)/0.1)]"
                  }`}
                >
                  <div className="text-[11px] uppercase tracking-[0.08em] text-[rgb(var(--text-muted-rgb))] mb-1">
                    {m.role}
                  </div>
                  <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
                </div>
              )
            )}
            {waiting && (
              <div className="rounded-2xl px-3.5 py-2.5 bg-[rgb(var(--panel-2-rgb)/0.92)] border border-[rgb(var(--surface-border-rgb)/0.3)] inline-flex items-center gap-2 shadow-[0_6px_14px_rgb(var(--shadow-rgb)/0.1)]">
                <span className="typing-dots" aria-label="Assistant is thinking">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </span>
              </div>
            )}
            <div ref={endRef} />
          </div>
          <div className="mt-3 flex gap-2">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="flex-1 min-h-[62px] bg-[rgb(var(--bg-alt-rgb)/0.85)] border border-[rgb(var(--surface-border-rgb)/0.42)] rounded-xl p-3 placeholder-[rgb(var(--text-muted-rgb))] focus:outline-none focus:ring-2 focus:ring-[rgb(var(--accent-rgb)/0.55)]"
              placeholder="Type your message..."
            />
            <button
              onClick={send}
              disabled={loading}
              className="self-end rounded-xl bg-[linear-gradient(135deg,rgb(var(--accent-rgb)),rgb(var(--accent-soft-rgb)))] text-[rgb(var(--on-accent-rgb))] px-4 py-2 font-semibold hover:brightness-105 disabled:opacity-50 transition-all duration-200 shadow-[0_10px_20px_rgb(var(--shadow-rgb)/0.16)]"
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
