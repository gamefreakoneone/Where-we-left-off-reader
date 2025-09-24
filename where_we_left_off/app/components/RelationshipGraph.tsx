"use client";

import { useMemo } from 'react';
import ReactFlow, { Background, Controls, MiniMap, Node, Edge } from 'reactflow';
import 'reactflow/dist/style.css';

interface RelationshipGraphProps {
  storyData: any;
  bookmarkedPage: number;
}

export default function RelationshipGraph({ storyData, bookmarkedPage }: RelationshipGraphProps) {

  const graphData = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];

    if (!storyData || !storyData.chapter) return { nodes, edges };

    const currentChapter = storyData.chapter.find((chap: any) => 
      bookmarkedPage >= chap.pages[0] && bookmarkedPage <= chap.pages[1]
    );

    if (!currentChapter || !currentChapter.characters) return { nodes, edges };

    const characterNames = new Set(currentChapter.characters.map((c: any) => c.name));

    currentChapter.characters.forEach((character: any, index: number) => {
      nodes.push({
        id: character.name,
        position: { x: (index % 4) * 200, y: Math.floor(index / 4) * 150 },
        data: { label: character.name },
        style: { background: '#2d3748', color: 'white', border: '1px solid #4a5568' }
      });

      if (character.relationships) {
        character.relationships.forEach((rel: any) => {
          // Only create edges for characters present in the current chapter
          if (characterNames.has(rel.with_name)) {
            edges.push({
              id: `${character.name}-${rel.with_name}`,
              source: character.name,
              target: rel.with_name,
              label: rel.type,
              animated: true,
              style: { stroke: '#a0aec0' },
              labelStyle: { fill: 'white', fontWeight: 600 }
            });
          }
        });
      }
    });

    return { nodes, edges };
  }, [storyData, bookmarkedPage]);

  if (graphData.nodes.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-400">No character data for this section.</div>
  }

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <ReactFlow
        nodes={graphData.nodes}
        edges={graphData.edges}
        fitView
      >
        <Background color="#4a5568" />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}
