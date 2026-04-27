import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Bot, User, Activity, GitBranch, Zap, Clock, FileText, Cpu, Network, BarChart3, Layers, Trash2, PlayCircle, ArrowLeftRight, MessageSquare } from 'lucide-react';
import GraphVis from './GraphVis';
import type { GraphData } from './GraphVis';
import Compare from './Compare';
import './App.css';

/* ── Types ── */
interface PipelineStep {
  name: string;
  detail: string;
  duration?: number;
  nodes?: { id: string; similarity: number; text: string }[];
  bfs_path?: string[];
  compressed_prompt?: string;
  retrieved_nodes?: { id: string; text: string }[];
  token_metrics?: TokenMetrics;
  causal_count?: number;
  temporal_count?: number;
}

interface TokenMetrics {
  input_tokens: number;
  context_tokens: number;
  total_original_tokens: number;
  compressed_tokens: number;
  compression_ratio: number;
  tokens_saved: number;
  context_usage_percentage?: number;
  max_allowed_tokens?: number;
}

interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  label?: string;
  stats?: { bfsMatches: number; processTime: number };
  pipeline?: PipelineStep[];
  tokenMetrics?: TokenMetrics;
  graphStats?: { total_nodes: number; total_edges: number; question_nodes: number; response_nodes: number };
  context_full?: boolean;
}

type Stage = null | 'sending' | 'creating_node' | 'finding_similar' | 'bfs' |
  'temporal' | 'compressing' | 'llm' | 'add_response' | 'causal' | 'edges' | 'done';

const STAGE_META: Record<string, { label: string; icon: React.ReactNode }> = {
  sending: { label: 'Sending query', icon: <Send size={13} /> },
  creating_node: { label: 'Creating question node + embedding', icon: <GitBranch size={13} /> },
  finding_similar: { label: 'Top-K cosine similarity search', icon: <Activity size={13} /> },
  bfs: { label: 'BFS traversal (similarity threshold)', icon: <Network size={13} /> },
  temporal: { label: 'Timestamp ordering', icon: <Clock size={13} /> },
  compressing: { label: 'Abstractive / extractive compression', icon: <Layers size={13} /> },
  llm: { label: 'Generating response (Ollama LLM)', icon: <Cpu size={13} /> },
  add_response: { label: 'Adding response node to graph', icon: <GitBranch size={13} /> },
  causal: { label: 'Causal + temporal edge detection', icon: <Zap size={13} /> },
  edges: { label: 'Similarity edges for future BFS', icon: <Activity size={13} /> },
  done: { label: 'Complete', icon: <BarChart3 size={13} /> },
};

const STAGES: Stage[] = ['sending', 'creating_node', 'finding_similar', 'bfs', 'temporal', 'compressing', 'llm', 'add_response', 'causal', 'edges', 'done'];

/* ── Error Boundary ── */
class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean }> {
  constructor(props: any) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch(error: any, errorInfo: any) { console.error("Caught by boundary:", error, errorInfo); }
  render() {
    if (this.state.hasError) return <div className="error-screen"><h3>Something went wrong.</h3><button onClick={() => window.location.reload()}>Reload</button></div>;
    return this.props.children;
  }
}

/* ── Component ── */
const App: React.FC = () => {
  const [view, setView] = useState<'chat' | 'compare'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [lastQId, setLastQId] = useState<string | null>(null);
  const [lastRId, setLastRId] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>(null);
  const [selected, setSelected] = useState<Message | null>(null);
  const [testRunning, setTestRunning] = useState(false);
  const [testProgress, setTestProgress] = useState('');
  const [isContextFull, setIsContextFull] = useState(false);
  const [contextSummary, setContextSummary] = useState<any>(null);
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch('http://localhost:8000/graph_state')
      .then(r => r.json()).then(s => setGraphData(s)).catch(() => { });
  }, []);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, isTyping]);

  /* Polling for "node-by-node" graph updates while backend is busy */
  useEffect(() => {
    let interval: any;
    if (isTyping || testRunning) {
      interval = setInterval(async () => {
        try {
          const r = await fetch('http://localhost:8000/graph_state');
          const d = await r.json();
          setGraphData(d);
        } catch { }
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isTyping, testRunning]);

  const tickStages = useCallback(() => {
    const delay = [100, 500, 800, 600, 200, 1000, 6000, 200, 2500, 200, 0];
    let t = 0;
    STAGES.forEach((s, i) => { t += delay[i] || 0; setTimeout(() => setStage(s), t); });
  }, []);

  /* ── Send message ── */
  const send = async () => {
    if (isContextFull) return;
    const txt = input.trim();
    if (!txt) return;
    setInput('');
    setMessages(p => [...p, { id: `u${Date.now()}`, sender: 'user', text: txt }]);
    setIsTyping(true);
    setStage('sending');
    tickStages();

    try {
      const r = await fetch('http://localhost:8000/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: txt })
      });
      const d = await r.json();
      setIsTyping(false);
      setStage('done');

      if (d.context_full) {
        setIsContextFull(true);
        setContextSummary(d.token_metrics);
        setStage(null);
        return;
      }

      if (d.success) {
        const m: Message = {
          id: `b${Date.now()}`, sender: 'bot', text: d.response,
          stats: { bfsMatches: d.relevant_nodes_count, processTime: d.processing_time },
          pipeline: d.pipeline_trace?.steps || [],
          tokenMetrics: d.token_metrics || undefined,
          graphStats: d.graph_stats || undefined,
          context_full: d.context_full || false
        };
        setMessages(p => [...p, m]);
        setSelected(m);
      } else {
        setMessages(p => [...p, { id: `b${Date.now()}`, sender: 'bot', text: `Error: ${d.error}` }]);
      }
      if (d.graph) { setGraphData(d.graph); setLastQId(d.question_id); setLastRId(d.response_id); }
      setTimeout(() => setStage(null), 2500);
    } catch {
      setIsTyping(false);
      setStage(null);
      setMessages(p => [...p, { id: `e${Date.now()}`, sender: 'bot', text: 'Backend unreachable.' }]);
    }
  };

  /* ── Clear context ── */
  const clearContext = async () => {
    try {
      await fetch('http://localhost:8000/clear', { method: 'POST' });
      setMessages([]);
      setGraphData({ nodes: [], edges: [] });
      setSelected(null);
      setLastQId(null);
      setLastRId(null);
      setStage(null);
      setTestProgress('');
      setIsContextFull(false);
      setContextSummary(null);
    } catch { alert('Failed to clear context'); }
  };

  /* ── Test prompts (matches backend TEST_PROMPTS) ── */
  const TEST_PROMPTS = [
    "What is photosynthesis and how do plants use sunlight?",
    "How does chlorophyll absorb light during photosynthesis?",
    "Why do plants need water and carbon dioxide for photosynthesis?",
    "Who was Julius Caesar and what did he accomplish?",
    "What is the role of glucose produced during photosynthesis?",
    "How does the water cycle work? Explain evaporation and condensation.",
    "What causes rainfall and how does precipitation occur?",
    "Deforestation leads to increased carbon dioxide because fewer trees perform photosynthesis.",
    "What was the structure of the Roman government under the Republic?",
    "How does photosynthesis connect to the water cycle through transpiration?",
    "First a seed germinates, then the seedling grows leaves, and finally the mature plant begins photosynthesis.",
    "Explain what photosynthesis is and how plants convert sunlight to energy.",
    "Because burning fossil fuels releases CO2, global warming increases, which disrupts both photosynthesis rates and the water cycle.",
    "Compare the Roman aqueduct system with modern water distribution. How did Roman engineering affect agriculture and food production?",
    "Summarize the connections between plant biology, the water cycle, climate change, and ancient Roman agriculture.",
  ];

  /* ── Run test pipeline (sequential /chat calls, just like manual typing) ── */
  const runTestPipeline = async () => {
    setTestRunning(true);
    const total = TEST_PROMPTS.length;

    for (let idx = 0; idx < total; idx++) {
      const prompt = TEST_PROMPTS[idx];
      setTestProgress(`Test ${idx + 1} / ${total}`);

      // Show user bubble
      setMessages(p => [...p, {
        id: `tu${Date.now()}_${idx}`, sender: 'user', text: prompt,
        label: `Test ${idx + 1}/${total}`
      }]);

      setIsTyping(true);
      setStage('sending');
      tickStages();

      try {
        const r = await fetch('http://localhost:8000/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: prompt })
        });
        const d = await r.json();
        setIsTyping(false);
        setStage('done');

        if (d.context_full) {
          setIsContextFull(true);
          setContextSummary(d.token_metrics);
          setTestProgress('Context Full - Stopped');
          break;
        }

        if (d.success) {
          const m: Message = {
            id: `tb${Date.now()}_${idx}`, sender: 'bot', text: d.response,
            label: `Test ${idx + 1}/${total}`,
            stats: { bfsMatches: d.relevant_nodes_count, processTime: d.processing_time },
            pipeline: d.pipeline_trace?.steps || [],
            tokenMetrics: d.token_metrics || undefined,
            graphStats: d.graph_stats || undefined,
            context_full: d.context_full || false
          };
          setMessages(p => [...p, m]);
          setSelected(m);
        } else {
          setMessages(p => [...p, { id: `te${Date.now()}_${idx}`, sender: 'bot', text: `Error: ${d.error}` }]);
        }

        if (d.graph) { setGraphData(d.graph); setLastQId(d.question_id); setLastRId(d.response_id); }
        if (d.context_full) {
          console.log("Context full reached, stopping pipeline.");
          break;
        }
      } catch {
        setMessages(p => [...p, { id: `te${Date.now()}_${idx}`, sender: 'bot', text: 'Backend error.' }]);
      }

      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    setTestRunning(false);
    setTestProgress('Test complete!');
    setStage(null);
    setTimeout(() => setTestProgress(''), 8000);
  };

  const sIdx = stage ? STAGES.indexOf(stage) : -1;

  if (view === 'compare') {
    return (
      <div className="view-wrap">
        <nav className="view-nav">
          <button className={`nav-tab`} onClick={() => setView('chat')}>
            <MessageSquare size={14} /> Chat + Pipeline
          </button>
          <button className={`nav-tab active`} onClick={() => setView('compare')}>
            <ArrowLeftRight size={14} /> Compare
          </button>
        </nav>
        <Compare />
      </div>
    );
  }

  return (
    <div className="view-wrap">
      <nav className="view-nav">
        <button className={`nav-tab active`} onClick={() => setView('chat')}>
          <MessageSquare size={14} /> Chat + Pipeline
        </button>
        <button className={`nav-tab`} onClick={() => setView('compare')}>
          <ArrowLeftRight size={14} /> Compare
        </button>
      </nav>

      {isContextFull && (
        <div className="summary-overlay">
          <div className="summary-card">
            <div className="summary-head">
              <Zap size={24} color="#fbbf24" strokeWidth={2.5} />
              <h2>PIPELINE CONSOLIDATED</h2>
            </div>
            <div className="summary-desc">The context window has reached its maximum capacity. No further questions will be sent to the model.</div>
            <div className="summary-stats">
              <div className="s-stat"><label>Model</label><span>{contextSummary?.model_name || 'phi3:mini'}</span></div>
              <div className="s-stat"><label>Questions Processed</label><span>{contextSummary?.questions_processed ?? '—'}</span></div>
              <div className="s-stat"><label>Total Input Tokens</label><span>{contextSummary?.cumulative_input_tokens ?? contextSummary?.input_tokens ?? '—'}</span></div>
              <div className="s-stat"><label>Total Original Tokens</label><span>{contextSummary?.cumulative_original_tokens ?? contextSummary?.total_original_tokens ?? '—'}</span></div>
              <div className="s-stat"><label>Total Compressed Tokens</label><span>{contextSummary?.cumulative_compressed_tokens ?? contextSummary?.compressed_tokens ?? '—'}</span></div>
              <div className="s-stat"><label>Tokens Saved</label><span className="hi-green">{contextSummary?.cumulative_tokens_saved ?? contextSummary?.tokens_saved ?? '—'}</span></div>
              <div className="s-stat"><label>Compression Ratio</label><span className="hi-green">{contextSummary?.cumulative_compression_ratio ?? contextSummary?.compression_ratio ?? '—'}%</span></div>
              <div className="s-stat"><label>Context Window</label><span>{contextSummary?.cumulative_compressed_tokens ?? contextSummary?.compressed_tokens} / {contextSummary?.max_allowed_tokens} ({contextSummary?.context_usage_percentage}%)</span></div>
            </div>
            <button className="summary-close" onClick={clearContext}>Clear &amp; Start Fresh</button>
          </div>
        </div>
      )}

      <ErrorBoundary>
        <div className="root-grid">
          {/* ══════════ LEFT: CHAT ══════════ */}
          <section className="col chat-col">
            <header className="col-head">
              <div className="head-row">
                <span className="head-title">Chat</span>
                <div className="head-actions">
                  <button className="act-btn clear-btn" onClick={clearContext} disabled={testRunning}
                    title="Clear all context and graph">
                    <Trash2 size={13} /> Clear
                  </button>
                  <button className="act-btn test-btn" onClick={runTestPipeline} disabled={testRunning || isTyping || isContextFull}
                    title="Run test pipeline">
                    <PlayCircle size={13} /> {testRunning ? testProgress : 'Run Test'}
                  </button>
                </div>
              </div>
            </header>

            <div className="chat-body" ref={chatRef}>
              {messages.length === 0 && (
                <div className="empty">
                  <Network size={36} strokeWidth={1.5} />
                  <p>Send a message or run the test pipeline</p>
                </div>
              )}
              {messages.map(m => (
                <div key={m.id}
                  className={`bubble ${m.sender} ${selected?.id === m.id ? 'sel' : ''}`}
                  onClick={() => m.sender === 'bot' && m.pipeline && setSelected(m)}>
                  <div className="bub-meta">
                    {m.sender === 'user' ? <User size={11} /> : <Bot size={11} />}
                    <span>{m.sender === 'user' ? 'You' : 'System'}</span>
                    {m.label && <span className="bub-label">{m.label}</span>}
                  </div>
                  <p className={`bub-text ${m.context_full ? 'consolidate-text' : ''}`}>{m.text}</p>
                  {m.stats && (
                    <div className="bub-tags">
                      <span className="tag t-blue">BFS {m.stats.bfsMatches}</span>
                      <span className="tag t-green">{m.stats.processTime.toFixed(1)}s</span>
                      {m.tokenMetrics && <span className="tag t-violet">{m.tokenMetrics.compression_ratio}% compressed</span>}
                    </div>
                  )}
                </div>
              ))}
              {isTyping && <div className="dots"><span /><span /><span /></div>}
            </div>

            <div className="chat-input">
              <input value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && send()}
                placeholder={isContextFull ? "Context full. Please clear." : "Ask something..."}
                disabled={isTyping || testRunning || isContextFull} />
              <button onClick={send} disabled={isTyping || !input.trim() || testRunning || isContextFull}><Send size={15} /></button>
            </div>
          </section>

          {/* ══════════ CENTER: PIPELINE + METRICS ══════════ */}
          <section className="col pipe-col">
            <header className="col-head">
              <span className="head-title">Pipeline</span>
              {testProgress && <span className="test-badge">{testProgress}</span>}
            </header>
            <div className="stage-list">
              {STAGES.filter(Boolean).map((s, i) => {
                const done = i < sIdx;
                const act = i === sIdx;
                const meta = STAGE_META[s!];
                return (
                  <div key={s} className={`stg ${done ? 'done' : ''} ${act ? 'act' : ''}`}>
                    <div className="stg-icon">{done ? '\u2713' : act ? meta.icon : ''}</div>
                    <span>{meta.label}</span>
                  </div>
                );
              })}
              {!stage && <div className="idle-hint">Idle — waiting for input</div>}
            </div>

            {/* ── Detail Panel (selected message) ── */}
            {selected && selected.pipeline && (
              <div className="detail-scroll">
                {selected.tokenMetrics && (
                  <div className="card">
                    <h4><BarChart3 size={13} /> Token Metrics</h4>
                    <div className="bars">
                      <div className="bar-row">
                        <span className="bar-label">Original</span>
                        <div className="bar-track"><div className="bar-fill fill-amber" style={{ width: '100%' }} /></div>
                        <span className="bar-val">{selected.tokenMetrics.total_original_tokens}</span>
                      </div>
                      <div className="bar-row">
                        <span className="bar-label">Compressed</span>
                        <div className="bar-track">
                          <div className="bar-fill fill-green" style={{ width: `${Math.max(5, Math.min(100, 100 - (selected.tokenMetrics.compression_ratio || 0)))}%` }} />
                        </div>
                        <span className="bar-val">{selected.tokenMetrics.compressed_tokens}</span>
                      </div>
                      {selected.tokenMetrics.context_usage_percentage !== undefined && (
                        <div className="bar-row">
                          <span className="bar-label">Context Limit</span>
                          <div className="bar-track">
                            <div className="bar-fill" style={{ background: '#3b82f6', width: `${Math.max(2, Math.min(100, selected.tokenMetrics.context_usage_percentage || 0))}%` }} />
                          </div>
                          <span className="bar-val">{selected.tokenMetrics.compressed_tokens}/{selected.tokenMetrics.max_allowed_tokens} ({selected.tokenMetrics.context_usage_percentage}%)</span>
                        </div>
                      )}
                    </div>
                    <div className="metric-grid">
                      <div className="mg"><span className="mg-n green">{selected.tokenMetrics.compression_ratio}%</span><span className="mg-l">reduction</span></div>
                      <div className="mg"><span className="mg-n blue">{selected.tokenMetrics.tokens_saved}</span><span className="mg-l">saved</span></div>
                      <div className="mg"><span className="mg-n violet">{selected.tokenMetrics.input_tokens}</span><span className="mg-l">input</span></div>
                      <div className="mg"><span className="mg-n cyan">{selected.tokenMetrics.context_tokens}</span><span className="mg-l">context</span></div>
                    </div>
                  </div>
                )}

                <div className="card">
                  <h4><FileText size={13} /> Step Trace</h4>
                  <div className="trace">
                    {selected.pipeline.map((s, i) => (
                      <div key={i} className="tr-step">
                        <div className="tr-head">
                          <span className="tr-num">{i + 1}</span>
                          <span className="tr-name">{s.name}</span>
                          {s.duration != null && <span className="tr-time">{s.duration}s</span>}
                        </div>
                        <p className="tr-det">{s.detail}</p>
                        {s.nodes && s.nodes.length > 0 && (
                          <div className="tr-chips">
                            {s.nodes.map((n, j) => (
                              <span key={j} className="chip">
                                <b>{n.id}</b> <em>{n.similarity?.toFixed(3) || 'sim'}</em>
                              </span>
                            ))}
                          </div>
                        )}
                        {s.retrieved_nodes && s.retrieved_nodes.length > 0 && (
                          <div className="tr-nodes">
                            <span className="tr-ctx-title">Nodes sent to LLM:</span>
                            {s.retrieved_nodes.map((rn, k) => (
                              <div key={k} className="tr-rn">
                                <span className="rn-id">{rn.id}</span>
                                <span className="rn-txt">{rn.text}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {s.compressed_prompt && (
                          <div className="tr-prompt-box">
                            <span className="tr-ctx-title">Final compressed prompt:</span>
                            <pre className="tr-pre">{s.compressed_prompt}</pre>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* BFS explanation */}
                <div className="card bfs-card">
                  <h4><Network size={13} /> How BFS Works</h4>
                  <ol className="bfs-ol">
                    <li>The input is embedded with a sentence transformer model.</li>
                    <li>Top-K nodes are found via <b>global cosine similarity</b> against all existing nodes.</li>
                    <li>BFS explores graph edges (similarity, causal, temporal, Q→R), pruning branches where <code>cosine &lt; threshold</code>.</li>
                    <li>Results are merged, deduplicated, and sorted by timestamp.</li>
                    <li>The compressed context is fed to the LLM for response generation.</li>
                  </ol>
                </div>

                {selected.graphStats && (
                  <div className="gstat-bar">
                    <span><b>{selected.graphStats.total_nodes}</b> nodes</span>
                    <span><b>{selected.graphStats.total_edges}</b> edges</span>
                    <span><b>{selected.graphStats.question_nodes}</b> Q</span>
                    <span><b>{selected.graphStats.response_nodes}</b> R</span>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* ══════════ RIGHT: GRAPH ══════════ */}
          <section className="col graph-col">
            <header className="col-head">
              <span className="head-title">Knowledge Graph</span>
            </header>
            <GraphVis graphData={graphData} questionId={lastQId} responseId={lastRId} />
          </section>
        </div>
      </ErrorBoundary>
    </div>
  );
};

export default App;
