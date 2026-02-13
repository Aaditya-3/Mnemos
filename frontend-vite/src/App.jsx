import { useEffect, useRef, useState } from "react";

const API_STREAM_URL = "/chat/stream";
const SEMANTIC_MEMORIES_API = "/memories/semantic";

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
  const [userId, setUserId] = useState(localStorage.getItem("user_id") || "demo");
  const [text, setText] = useState("");
  const [messages, setMessages] = useState([]);
  const [semantic, setSemantic] = useState([]);
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const headers = (json = false) => {
    const h = { "X-User-ID": userId };
    if (json) h["Content-Type"] = "application/json";
    return h;
  };

  const loadSemantic = async () => {
    const r = await fetch(SEMANTIC_MEMORIES_API, { headers: headers(false) });
    const data = await r.json();
    setSemantic(data.memories || []);
  };

  const send = async () => {
    if (!text.trim() || loading) return;
    const msg = text.trim();
    setText("");
    setLoading(true);
    const assistantId = `a-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: msg },
      { id: assistantId, role: "assistant", content: "" }
    ]);
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
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: `${m.content}\n[tool:${toolName}] ${JSON.stringify(evt.payload)}` } : m
              )
            );
          } else if (evt.event === "token") {
            const token = evt.payload?.text || "";
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: `${m.content}${token}` } : m))
            );
          } else if (evt.event === "error") {
            throw new Error(evt.payload?.message || "Stream error");
          }
        }
      }
      await loadSemantic();
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: `Error: ${err.message}` } : m))
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6">
      <div className="mx-auto max-w-6xl grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-4">
        <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-4">
          <div className="flex items-center justify-between border-b border-slate-700 pb-3 mb-3">
            <h1 className="text-xl font-semibold">Mnemos (Vite Build)</h1>
            <input
              value={userId}
              onChange={(e) => {
                setUserId(e.target.value);
                localStorage.setItem("user_id", e.target.value);
              }}
              className="bg-slate-800 border border-slate-600 rounded-lg px-2 py-1 text-sm"
            />
          </div>
          <div className="h-[60vh] overflow-y-auto space-y-3 pr-1">
            {messages.map((m, i) => (
              <div key={m.id || i} className={`rounded-xl px-3 py-2 ${m.role === "user" ? "bg-blue-700/30" : "bg-slate-800"}`}>
                <div className="text-xs text-slate-400 mb-1">{m.role}</div>
                <div className="whitespace-pre-wrap">{m.content}</div>
              </div>
            ))}
            <div ref={endRef} />
          </div>
          <div className="mt-3 flex gap-2">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="flex-1 min-h-[60px] bg-slate-800 border border-slate-600 rounded-xl p-3"
              placeholder="Type your message..."
            />
            <button
              onClick={send}
              disabled={loading}
              className="self-end rounded-xl bg-emerald-400 text-slate-900 px-4 py-2 font-semibold disabled:opacity-50"
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </section>

        <aside className="rounded-2xl border border-slate-700 bg-slate-900/80 p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Semantic Memory</h2>
            <button onClick={loadSemantic} className="text-xs px-2 py-1 rounded bg-slate-700">
              Refresh
            </button>
          </div>
          <div className="space-y-2 max-h-[70vh] overflow-y-auto">
            {semantic.map((m) => (
              <div key={m.memory_id || m.id} className="rounded-lg border border-slate-700 p-2 text-sm">
                <div className="text-xs text-slate-400">
                  {(m.memory_type || m.type || "memory")} - {m.scope} - {Number(m.importance_score || 0).toFixed(2)}
                </div>
                <div className="mt-1">{m.content}</div>
              </div>
            ))}
            {!semantic.length && <div className="text-sm text-slate-400">No semantic memories yet.</div>}
          </div>
        </aside>
      </div>
    </div>
  );
}
