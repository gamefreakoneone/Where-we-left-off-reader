"use client";

import { useMemo, useState, useCallback, useEffect } from 'react';
import ReactFlow, { Background, Controls, MiniMap, Node, Edge, useNodesState, useEdgesState } from 'reactflow';
import 'reactflow/dist/style.css';

interface RelationshipGraphProps {
  graphData: any;
  bookmarkedPage: number;
}

export default function RelationshipGraph({ graphData: storyData, bookmarkedPage }: RelationshipGraphProps) {

  const initialGraphData = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const edgeMap = new Map<string, boolean>();

    if (!storyData || !storyData.chapter) return { nodes, edges };

    const currentChapter = storyData.chapter.find((chap: any) => 
      bookmarkedPage >= chap.pages[0] && bookmarkedPage <= chap.pages[1]
    );

    if (!currentChapter || !currentChapter.characters) return { nodes, edges };

    const characterNames = new Set(currentChapter.characters.map((c: any) => c.name));

    // Create nodes
    currentChapter.characters.forEach((character: any, index: number) => {
      nodes.push({
        id: character.name,
        position: { x: (index % 4) * 250, y: Math.floor(index / 4) * 180 },
        data: { label: character.name },
        draggable: true,
        style: { 
          background: '#2d3748', 
          color: 'white', 
          border: '2px solid #4a5568',
          padding: '12px 20px',
          borderRadius: '8px',
          fontSize: '14px',
          fontWeight: '600',
          cursor: 'grab'
        }
      });
    });

    // Create edges - directional with justification tooltips
    let edgeCounter = 0;
    const bidirectionalPairs = new Map<string, number>();
    
    // First pass: identify bidirectional relationships
    currentChapter.characters.forEach((character: any) => {
      if (character.relationships) {
        character.relationships.forEach((rel: any) => {
          if (characterNames.has(rel.with_name) && rel.type !== 'unknown') {
            const sortedPair = [character.name, rel.with_name].sort().join('|||');
            bidirectionalPairs.set(sortedPair, (bidirectionalPairs.get(sortedPair) || 0) + 1);
          }
        });
      }
    });
    
    // Second pass: create edges with appropriate curvature
    currentChapter.characters.forEach((character: any) => {
      if (character.relationships) {
        character.relationships.forEach((rel: any, relIndex: number) => {
          if (characterNames.has(rel.with_name) && rel.type !== 'unknown') {
            const edgeKey = `${character.name}->${rel.with_name}`;
            const sortedPair = [character.name, rel.with_name].sort().join('|||');
            const isBidirectional = (bidirectionalPairs.get(sortedPair) || 0) > 1;
            
            edgeMap.set(edgeKey, true);
            
            // Alternate label positions and add curvature offset for bidirectional edges
            const labelPosition = edgeCounter % 2 === 0 ? 0.3 : 0.7;
            const isFirstDirection = character.name < rel.with_name;
            const curveOffset = isBidirectional ? (isFirstDirection ? 50 : -50) : 0;
            edgeCounter++;
            
            edges.push({
              id: `${character.name}-${rel.with_name}-${relIndex}`,
              source: character.name,
              target: rel.with_name,
              label: rel.type,
              type: 'default',
              labelPosition: labelPosition,
              pathOptions: { offset: curveOffset, curvature: 0.5 },
              markerEnd: {
                type: 'arrowclosed',
                color: '#60a5fa',
                width: 25,
                height: 25
              },
              style: { 
                stroke: '#60a5fa',
                strokeWidth: 3
              },
              labelStyle: { 
                fill: '#fbbf24', 
                fontWeight: 700,
                fontSize: 14
              },
              labelBgStyle: {
                fill: '#1e293b',
                fillOpacity: 0.95
              },
              labelBgPadding: [8, 6] as [number, number],
              data: { 
                justification: rel.justification || 'No justification provided'
              }
            });
          }
        });
      }
    });

    return { nodes, edges };
  }, [storyData, bookmarkedPage]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialGraphData.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialGraphData.edges);
  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null);

  // Update nodes and edges when chapter changes
  useEffect(() => {
    setNodes(initialGraphData.nodes);
    setEdges(initialGraphData.edges);
  }, [initialGraphData, setNodes, setEdges]);

  const onEdgeMouseEnter = useCallback((event: any, edge: Edge) => {
    const justification = edge.data?.justification;
    if (justification) {
      setTooltip({
        text: justification,
        x: event.clientX,
        y: event.clientY
      });
    }
  }, []);

  const onEdgeMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  if (nodes.length === 0) {
    return <div className="flex items-center justify-center h-full text-muted">No character data for this section.</div>
  }

  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onEdgeMouseEnter={onEdgeMouseEnter}
        onEdgeMouseLeave={onEdgeMouseLeave}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Background color="#475569" gap={16} />
        <Controls />
        <MiniMap 
          nodeColor="#2d3748"
          maskColor="rgba(0, 0, 0, 0.6)"
          style={{
            height: 80,
            width: 120,
            backgroundColor: '#1e293b'
          }}
        />
      </ReactFlow>
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x + 10,
            top: tooltip.y + 10,
            backgroundColor: '#1e293b',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            border: '1px solid #475569',
            maxWidth: '300px',
            zIndex: 1000,
            pointerEvents: 'none',
            fontSize: '13px',
            lineHeight: '1.4'
          }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}