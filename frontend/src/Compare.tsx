import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Trash2, Zap, ArrowLeftRight } from 'lucide-react';
import './Compare.css';

/* ── Types ── */
interface CompareMsg {
    id: string;
    prompt: string;
    model: string;
    compressed: {
        response: string;
        time: number;
        input_tokens: number;
        context_tokens: number;
        total_original_tokens: number;
        compressed_tokens: number;
        compression_ratio: number;
        tokens_saved: number;
        bfs_nodes: number;
        pipeline_steps: number;
        context_full?: boolean;
        token_metrics?: any;
    };
    raw: {
        response: string;
        time: number;
        input_tokens: number;
        context_tokens: number;
        total_tokens_sent: number;
        response_tokens: number;
        history_length: number;
        context_full: boolean;
        token_metrics?: any;
    };
    context_full?: boolean;
}

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

const Compare: React.FC = () => {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<CompareMsg[]>([]);
    const [loading, setLoading] = useState(false);
    const [testRunning, setTestRunning] = useState(false);
    const [testProgress, setTestProgress] = useState('');
    const [isContextFull, setIsContextFull] = useState(false);
    const [contextSummary, setContextSummary] = useState<any>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages, loading]);

    const send = async () => {
        const txt = input.trim();
        if (!txt || loading) return;
        setInput('');
        setLoading(true);

        try {
            const r = await fetch('http://localhost:8000/compare', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: txt })
            });
            const d = await r.json();
            setMessages(prev => [...prev, {
                id: `c${Date.now()}`,
                prompt: d.prompt,
                model: d.model,
                compressed: d.compressed,
                raw: d.raw,
            }]);
        } catch {
            /* ignore */
        } finally {
            setLoading(false);
        }
    };

    const clear = async () => {
        try {
            await fetch('http://localhost:8000/clear_compare', { method: 'POST' });
            setMessages([]);
            setTestProgress('');
            setIsContextFull(false);
            setContextSummary(null);
        } catch { /* ignore */ }
    };

    const runTestPipeline = async () => {
        setTestRunning(true);
        let idx = 0;
        let bothFull = false;

        while (!bothFull) {
            const prompt = TEST_PROMPTS[idx % TEST_PROMPTS.length];
            setTestProgress(`Test Cycle #${idx + 1}...`);
            setLoading(true);

            // Node-by-node simulation: periodically fetch graph state while loading
            const poll = setInterval(async () => {
                try {
                    const gr = await fetch('http://localhost:8000/graph_state');
                    // This is handled globally in App.tsx if we had shared state, 
                    // but for Compare we just want the console/result to show progress.
                } catch { }
            }, 2000);

            try {
                const r = await fetch('http://localhost:8000/compare', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: prompt })
                });
                const d = await r.json();

                setMessages(prev => [...prev, {
                    id: `c${Date.now()}_${idx}`,
                    prompt: d.prompt,
                    model: d.model,
                    compressed: d.compressed,
                    raw: d.raw,
                }]);

                if (d.compressed.context_full || d.raw.context_full) {
                    bothFull = true;
                    setIsContextFull(true);
                    setContextSummary({
                        compressed: d.compressed.token_metrics,
                        raw: d.raw.token_metrics
                    });
                    setTestProgress("CONSOLIDATION: Pipeline Stopped");
                }

                // Safety break
                if (idx > 40) bothFull = true;

            } catch {
                bothFull = true;
            } finally {
                clearInterval(poll);
                setLoading(false);
            }

            if (bothFull) break;
            idx++;
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        setTestRunning(false);
        setTimeout(() => setTestProgress(''), 10000);
    };

    // Running totals
    const totalCompressedTokens = messages.reduce((s, m) => s + m.compressed.compressed_tokens, 0);
    const totalRawTokens = messages.reduce((s, m) => s + m.raw.total_tokens_sent, 0);
    const totalCompressedTime = messages.reduce((s, m) => s + m.compressed.time, 0);
    const totalRawTime = messages.reduce((s, m) => s + m.raw.time, 0);

    return (
        <div className="compare-root">
            {isContextFull && (
                <div className="summary-overlay">
                    <div className="summary-card" style={{ maxWidth: '800px' }}>
                        <div className="summary-head">
                            <Zap size={24} color="#fbbf24" strokeWidth={2.5} />
                            <h2>COMPARE CONSOLIDATED</h2>
                        </div>
                        <div className="summary-desc">One or both context windows have reached their limits. Pipeline halted for the comparison.</div>
                        <div className="summary-grid-compare">
                            <div className="s-bit-box">
                                <h3>COMPRESSED (Ours)</h3>
                                <div className="s-stat"><label>Efficiency</label><span className="hi-green">{contextSummary?.compressed?.compression_ratio}%</span></div>
                                <div className="s-stat"><label>Window</label><span>{contextSummary?.compressed?.compressed_tokens} / {contextSummary?.compressed?.max_allowed_tokens}</span></div>
                            </div>
                            <div className="s-bit-box">
                                <h3>RAW (Baseline)</h3>
                                <div className="s-stat"><label>Efficiency</label><span>0%</span></div>
                                <div className="s-stat"><label>Window</label><span>{contextSummary?.raw?.compressed_tokens} / {contextSummary?.raw?.max_allowed_tokens}</span></div>
                            </div>
                        </div>
                        <button className="summary-close" onClick={clear}>Clear & Reset Comparison</button>
                    </div>
                </div>
            )}

            <div className="compare-grid">
                {/* ── Header ── */}
                <header className="cmp-header">
                    <div className="cmp-title-row">
                        <ArrowLeftRight size={16} />
                        <span className="cmp-title">Side-by-Side Comparison {testProgress && <span style={{ fontSize: '0.8rem', marginLeft: '10px', color: '#fbbf24' }}>{testProgress}</span>}</span>
                        <span className="cmp-model">{messages[0]?.model || 'phi3:mini'}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button className="cmp-clear" onClick={clear} disabled={loading || testRunning} style={{ background: 'rgba(239, 68, 68, 0.2)', padding: '4px 8px', borderRadius: '4px' }}>
                            <Trash2 size={12} /> Clear
                        </button>
                        <button className="cmp-clear" onClick={runTestPipeline} disabled={loading || testRunning} style={{ background: 'rgba(34, 197, 94, 0.2)', padding: '4px 8px', borderRadius: '4px', color: '#4ade80' }}>
                            <Zap size={12} /> Run Test
                        </button>
                    </div>
                </header>

                {/* ── Column Headers ── */}
                <div className="cmp-col-heads">
                    <div className="cmp-col-label left">
                        <Zap size={13} /> With Compression
                    </div>
                    <div className="cmp-col-label right">
                        <span className="no-icon">⊘</span> Without Compression (Raw)
                    </div>
                </div>

                {/* ── Messages ── */}
                <div className="cmp-body" ref={scrollRef}>
                    {messages.length === 0 && !loading && (
                        <div className="cmp-empty">
                            <ArrowLeftRight size={32} strokeWidth={1.5} />
                            <p>Type a prompt to see how compression affects the pipeline</p>
                        </div>
                    )}

                    {messages.map(m => (
                        <div key={m.id} className="cmp-round">
                            {/* Prompt */}
                            <div className="cmp-prompt">
                                <User size={11} /> <span>{m.prompt}</span>
                            </div>

                            {/* Side-by-side responses */}
                            <div className="cmp-sides">
                                {/* LEFT: Compressed */}
                                <div className="cmp-side compressed">
                                    <div className="side-response">
                                        <Bot size={11} />
                                        <p>{m.compressed.response}</p>
                                    </div>
                                    <div className="side-metrics">
                                        <div className="sm-row">
                                            <span className="sm-label">Time</span>
                                            <span className="sm-val">{m.compressed.time}s</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">Tokens sent</span>
                                            <span className="sm-val">{m.compressed.compressed_tokens}</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">Original tokens</span>
                                            <span className="sm-val">{m.compressed.total_original_tokens}</span>
                                        </div>
                                        <div className="sm-row highlight">
                                            <span className="sm-label">Saved</span>
                                            <span className="sm-val green">{m.compressed.tokens_saved} ({m.compressed.compression_ratio}%)</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">BFS nodes</span>
                                            <span className="sm-val">{m.compressed.bfs_nodes}</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">Context tokens</span>
                                            <span className="sm-val">{m.compressed.context_tokens}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* RIGHT: Raw */}
                                <div className="cmp-side raw">
                                    <div className="side-response">
                                        <Bot size={11} />
                                        <p>{m.raw.response}</p>
                                    </div>
                                    <div className="side-metrics">
                                        <div className="sm-row">
                                            <span className="sm-label">Time</span>
                                            <span className="sm-val">{m.raw.time}s</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">Tokens sent</span>
                                            <span className="sm-val">{m.raw.total_tokens_sent}</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">Response tokens</span>
                                            <span className="sm-val">{m.raw.response_tokens}</span>
                                        </div>
                                        <div className="sm-row highlight">
                                            <span className="sm-label">Saved</span>
                                            <span className="sm-val red">0 (0%)</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">History msgs</span>
                                            <span className="sm-val">{m.raw.history_length}</span>
                                        </div>
                                        <div className="sm-row">
                                            <span className="sm-label">Context tokens</span>
                                            <span className="sm-val">{m.raw.context_tokens}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}

                    {loading && (
                        <div className="cmp-loading">
                            <div className="cmp-loading-sides">
                                <div className="cmp-loading-side">Processing with compression...</div>
                                <div className="cmp-loading-side">Processing without compression...</div>
                            </div>
                        </div>
                    )}
                </div>

                {/* ── Totals ── */}
                {messages.length > 0 && (
                    <div className="cmp-totals">
                        <div className="cmp-total-item">
                            <span>Compressed total</span>
                            <b>{totalCompressedTokens} tokens</b>
                            <em>{totalCompressedTime.toFixed(1)}s</em>
                        </div>
                        <div className="cmp-total-item">
                            <span>Raw total</span>
                            <b>{totalRawTokens} tokens</b>
                            <em>{totalRawTime.toFixed(1)}s</em>
                        </div>
                        <div className="cmp-total-item savings">
                            <span>Savings</span>
                            <b>{totalRawTokens - totalCompressedTokens} tokens</b>
                            <em>{totalRawTokens > 0 ? ((1 - totalCompressedTokens / totalRawTokens) * 100).toFixed(1) : 0}%</em>
                        </div>
                    </div>
                )}

                {/* ── Consolidation Summaries ── */}
                {messages.length > 0 && (
                    <div className="cmp-consolidation">
                        <div className="cmp-summary-box compressed">
                            <h4><Zap size={14} /> COMPRESSED PIPELINE CONSOLIDATE</h4>
                            <div className="summary-grid">
                                <div className="s-bit"><b>Model:</b> {messages[0]?.model}</div>
                                <div className="s-bit"><b>Total Prompts:</b> {messages.length}</div>
                                <div className="s-bit"><b>Total Sent:</b> {totalCompressedTokens} tokens</div>
                                <div className="s-bit"><b>Saved:</b> {messages.reduce((s, m) => s + m.compressed.tokens_saved, 0)} tokens</div>
                                <div className="s-bit"><b>Status:</b> {messages[messages.length - 1].compressed.context_full ? 'HALTED (WINDOW FULL)' : 'ACTIVE'}</div>
                                <div className="s-bit"><b>Window:</b> {messages[messages.length - 1].compressed.context_tokens} / {messages[messages.length - 1].compressed.total_original_tokens} original</div>
                            </div>
                        </div>
                        <div className="cmp-summary-box raw">
                            <h4><Bot size={14} /> RAW PIPELINE CONSOLIDATE</h4>
                            <div className="summary-grid">
                                <div className="s-bit"><b>Model:</b> {messages[0]?.model}</div>
                                <div className="s-bit"><b>Total Prompts:</b> {messages.length}</div>
                                <div className="s-bit"><b>Total Sent:</b> {totalRawTokens} tokens</div>
                                <div className="s-bit"><b>Saved:</b> 0 tokens (0%)</div>
                                <div className="s-bit"><b>Status:</b> {messages[messages.length - 1].raw.context_full ? 'HALTED (WINDOW FULL)' : 'ACTIVE'}</div>
                                <div className="s-bit"><b>Window:</b> {messages[messages.length - 1].raw.context_tokens} / {messages[messages.length - 1].raw.context_tokens}</div>
                            </div>
                        </div>
                    </div>
                )}

                {/* ── Input ── */}
                <div className="cmp-input">
                    <input value={input} onChange={e => setInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && send()}
                        placeholder="Type a prompt to compare..." disabled={loading || testRunning} />
                </div>
            </div>
        </div>
    );
};

export default Compare;
