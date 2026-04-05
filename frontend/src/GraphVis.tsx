import React, { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';
import './GraphVis.css';

export interface GraphData {
    nodes: any[];
    edges: any[];
}

interface GraphVisProps {
    graphData: GraphData | null;
    questionId: string | null;
    responseId: string | null;
}

const GraphVis: React.FC<GraphVisProps> = ({ graphData, questionId, responseId }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const networkRef = useRef<any>(null);
    const nodesRef = useRef<any>(null);
    const edgesRef = useRef<any>(null);

    const [stats, setStats] = useState({ nodes: 0, edges: 0 });

    useEffect(() => {
        if (!containerRef.current) return;

        nodesRef.current = new DataSet([]);
        edgesRef.current = new DataSet([]);

        const options = {
            nodes: {
                shape: 'dot',
                size: 18,
                font: {
                    color: '#e2e8f0',
                    face: 'Inter',
                    size: 11,
                    background: 'rgba(10, 14, 26, 0.8)'
                },
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.4)',
                    size: 8,
                    x: 2,
                    y: 2
                }
            },
            edges: {
                width: 1.5,
                font: {
                    color: '#f8fafc',
                    size: 10,
                    face: 'Inter',
                    background: 'rgba(15, 23, 42, 0.95)',
                    strokeWidth: 1,
                    strokeColor: 'rgba(0,0,0,0.8)'
                },
                smooth: { enabled: true, type: 'continuous', roundness: 0.5 },
                arrows: { to: { enabled: true, scaleFactor: 0.4 } }
            },
            physics: {
                barnesHut: {
                    gravitationalConstant: -2500,
                    centralGravity: 0.3,
                    springLength: 160,
                    springConstant: 0.04
                },
                stabilization: { iterations: 120 }
            },
            groups: {
                question: {
                    color: {
                        background: '#1e3a8a',
                        border: '#3b82f6',
                        highlight: { background: '#2563eb', border: '#60a5fa' }
                    }
                },
                response: {
                    color: {
                        background: '#064e3b',
                        border: '#10b981',
                        highlight: { background: '#059669', border: '#34d399' }
                    }
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            }
        };

        const data = { nodes: nodesRef.current, edges: edgesRef.current };
        networkRef.current = new Network(containerRef.current, data, options);

        return () => {
            if (networkRef.current) {
                networkRef.current.destroy();
                networkRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (!graphData || !nodesRef.current || !edgesRef.current || !networkRef.current) return;

        const defaultNodes = nodesRef.current;
        const defaultEdges = edgesRef.current;

        const existingNodes = defaultNodes.getIds();
        const incomingNodeIds = graphData.nodes.map(n => n.id);

        // Remove old nodes not in incoming data (handles clears and resets)
        const nodesToRemove = existingNodes.filter((id: string | number) => !incomingNodeIds.includes(id as string));
        if (nodesToRemove.length > 0) defaultNodes.remove(nodesToRemove);

        // Add new nodes
        const newNodes = graphData.nodes
            .filter(n => !existingNodes.includes(n.id))
            .map(n => ({
                id: n.id,
                label: n.label,
                title: n.full_text,
                group: n.group,
            }));

        if (newNodes.length > 0) defaultNodes.add(newNodes);

        const newEdges: any[] = [];
        graphData.edges.forEach(e => {
            const edgeId = `${e.from}-${e.to}-${e.label}`;
            if (!defaultEdges.get(edgeId)) {
                let color = '#475569';
                let dashes: boolean | number[] = false;
                let displayLabel = '';

                if (e.label === 'causal') {
                    color = '#ef4444';
                    displayLabel = 'causal';
                } else if (e.label === 'temporal') {
                    color = '#a78bfa';
                    dashes = true;
                    displayLabel = 'temporal';
                } else if (e.label === 'similarity') {
                    color = '#22d3ee';
                    dashes = [5, 5];
                    displayLabel = e.confidence != null ? e.confidence.toFixed(2) : 'sim';
                } else if (e.label === 'question_response') {
                    color = '#3b82f6';
                }

                newEdges.push({
                    id: edgeId,
                    from: e.from,
                    to: e.to,
                    label: displayLabel,
                    color: { color, highlight: color },
                    dashes
                });
            }
        });

        // Remove old edges not in incoming data
        const existingEdges = defaultEdges.getIds();
        const incomingEdgeIds = new Set(graphData.edges.map(e => `${e.from}-${e.to}-${e.label}`));
        const edgesToRemove = existingEdges.filter((id: string | number) => !incomingEdgeIds.has(id as string));
        if (edgesToRemove.length > 0) defaultEdges.remove(edgesToRemove);

        if (newEdges.length > 0) defaultEdges.add(newEdges);

        setStats({
            nodes: defaultNodes.length,
            edges: defaultEdges.length
        });

        const highlightIds = [questionId, responseId]
            .filter(Boolean)
            .filter(id => defaultNodes.get(id as any)) as string[];

        if (highlightIds.length > 0) {
            try {
                networkRef.current.fit({ nodes: highlightIds, animation: true });
            } catch (e) {
                console.warn("Graph fit failed:", e);
            }

            highlightIds.forEach(id => {
                const original = defaultNodes.get(id);
                if (original) {
                    try {
                        defaultNodes.update({ id, borderWidth: 5, color: { border: '#fbbf24' } });
                        setTimeout(() => {
                            if (defaultNodes.get(id)) {
                                defaultNodes.update({ id, borderWidth: 2, color: { border: undefined } });
                            }
                        }, 4000);
                    } catch { }
                }
            });
        }
    }, [graphData, questionId, responseId]);

    return (
        <div id="graph-vis-root">
            <div id="network-container" ref={containerRef}></div>

            <div className="graph-legend">
                <div className="leg-item"><span className="leg-dot" style={{ background: '#3b82f6' }}></span>Question</div>
                <div className="leg-item"><span className="leg-dot" style={{ background: '#10b981' }}></span>Response</div>
                <div className="leg-item"><span className="leg-line" style={{ background: '#ef4444' }}></span>Causal</div>
                <div className="leg-item"><span className="leg-line" style={{ background: '#a78bfa' }}></span>Temporal</div>
                <div className="leg-item"><span className="leg-line dashed" style={{ background: '#22d3ee' }}></span>Similarity</div>
            </div>

            <div className="graph-counter">{stats.nodes} nodes · {stats.edges} edges</div>
        </div>
    );
};

export default GraphVis;
