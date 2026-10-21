"""
Knowledge Graph Module for PaperPilot.
Constructs rich semantic graphs of papers, claims, contradictions, gaps,
and hypotheses, rendering interactive Vis.js HTML visualizations.
"""

import json
import html
from typing import Dict, List, Any, Optional
import networkx as nx


def build_knowledge_graph_data(
    papers: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Builds NetworkX graph and returns nodes and edges formatted for Vis.js.
    """
    G = nx.DiGraph()
    nodes = []
    edges = []

    # 1. Paper Nodes
    for idx, p in enumerate(papers):
        paper_id = p.get("paper_id", f"P{idx+1}")
        title = p.get("title", f"Paper {idx+1}")
        short_title = title[:35] + ("..." if len(title) > 35 else "")
        nodes.append({
            "id": paper_id,
            "label": f"📄 {short_title}",
            "group": "paper",
            "title": f"<b>{title}</b><br>Authors: {p.get('authors', 'N/A')}<br>Year: {p.get('year', 'N/A')}",
            "shape": "box",
            "color": {"background": "#1E40AF", "border": "#1E3A8A"},
            "font": {"color": "#FFFFFF", "face": "Helvetica", "size": 14, "bold": True},
            "margin": 10,
            "shadow": True
        })
        G.add_node(paper_id, type="paper", label=title)

    # 2. Claim Nodes & Paper->Claim Edges
    for c in claims:
        cid = c["id"]
        topic = c.get("topic", "Claim")
        statement = c.get("statement", "")
        short_stmt = statement[:45] + "..." if len(statement) > 45 else statement
        conf = int(c.get("confidence", 0.9) * 100)
        
        nodes.append({
            "id": cid,
            "label": f"💡 {topic}\n{short_stmt}",
            "group": "claim",
            "title": f"<b>Claim [{cid}]</b><br><b>Topic:</b> {topic}<br><b>Statement:</b> {statement}<br><b>Evidence Quote:</b> <i>\"{c.get('quote', '')}\"</i><br><b>Confidence:</b> {conf}%",
            "shape": "ellipse",
            "color": {"background": "#0D9488", "border": "#0F766E"},
            "font": {"color": "#FFFFFF", "size": 11},
            "shadow": True
        })
        G.add_node(cid, type="claim", topic=topic, statement=statement)

        # Edge from paper to claim
        pid = c.get("paper_id")
        if pid:
            edges.append({
                "from": pid,
                "to": cid,
                "label": "asserts",
                "arrows": "to",
                "color": {"color": "#94A3B8", "highlight": "#64748B"},
                "font": {"size": 9, "color": "#64748B"},
                "dashes": False
            })
            G.add_edge(pid, cid, relation="asserts")

    # 3. Contradiction Edges / Tensions
    for idx, cont in enumerate(contradictions):
        cont_id = cont.get("id", f"CONT_{idx+1}")
        topic = cont.get("topic", "Contradiction")
        # Link the two representative claims by id (new engine). Older data falls back to statement matching.
        p_a_matched = cont.get("claim_a", {}).get("claim_id")
        p_b_matched = cont.get("claim_b", {}).get("claim_id")
        if not (p_a_matched and p_b_matched):
            for c in claims:
                sa = cont.get("claim_a", {}).get("statement", "")
                sb = cont.get("claim_b", {}).get("statement", "")
                if c.get("statement", "") and (c["statement"] in sa or sa in c["statement"]):
                    p_a_matched = c["id"]
                if c.get("statement", "") and (c["statement"] in sb or sb in c["statement"]):
                    p_b_matched = c["id"]

        source_node = p_a_matched
        target_node = p_b_matched

        if source_node and target_node and source_node != target_node:
            edges.append({
                "from": source_node,
                "to": target_node,
                "label": f"⚡ CONTRADICTS: {topic[:28]}...",
                "arrows": "to;from",
                "color": {"color": "#EF4444", "highlight": "#DC2626"},
                "font": {"color": "#DC2626", "size": 11, "bold": True, "strokeWidth": 2, "strokeColor": "#FFFFFF"},
                "width": 3,
                "dashes": [6, 4],
                "title": f"<b>Contradiction:</b> {html.escape(str(topic))}<br><b>Cause:</b> {html.escape(str(cont.get('root_cause', '')))}"
            })

    # 4. Research Gap Nodes
    for g in gaps:
        gid = g["id"]
        title = g.get("title", "Research Gap")
        short_title = title[:40] + "..." if len(title) > 40 else title
        nodes.append({
            "id": gid,
            "label": f"⚠️ GAP: {short_title}",
            "group": "gap",
            "title": f"<b>Research Gap [{gid}]</b><br><b>Category:</b> {html.escape(str(g.get('category')))}<br><b>Description:</b> {html.escape(str(g.get('description')))}<br><b>Impact:</b> {html.escape(str(g.get('impact_score')))}",
            "shape": "diamond",
            "color": {"background": "#D97706", "border": "#B45309"},
            "font": {"color": "#FFFFFF", "size": 11, "bold": True},
            "size": 24,
            "shadow": True
        })
        G.add_node(gid, type="gap", title=title)

        # Link from source papers or relevant claims
        for p in papers:
            linked = (p.get("paper_id") in g.get("paper_ids", [])) if g.get("paper_ids") else \
                any(p.get("title", "") in sp or sp in p.get("title", "") for sp in g.get("source_papers", []))
            if linked:
                edges.append({
                    "from": p.get("paper_id", "P1"),
                    "to": gid,
                    "label": "exposes",
                    "arrows": "to",
                    "color": {"color": "#F59E0B"},
                    "font": {"size": 9, "color": "#B45309"},
                    "dashes": True
                })

    # 5. Hypothesis Nodes
    for h in hypotheses:
        hid = h["id"]
        title = h.get("title", "Hypothesis")
        short_title = title[:40] + "..." if len(title) > 40 else title
        nodes.append({
            "id": hid,
            "label": f"🔮 HYPOTHESIS: {short_title}",
            "group": "hypothesis",
            "title": f"<b>Hypothesis [{hid}]: {html.escape(str(title))}</b><br><b>Formulation:</b> {html.escape(str(h.get('formulation')))}<br><b>Rationale:</b> {html.escape(str(h.get('rationale')))}",
            "shape": "box",
            "color": {"background": "#7C3AED", "border": "#6D28D9"},
            "font": {"color": "#FFFFFF", "size": 12, "bold": True},
            "margin": 8,
            "shadow": True
        })
        G.add_node(hid, type="hypothesis", title=title)

        # Edge to Gap it addresses
        if h.get("addresses_gap") and h["addresses_gap"] in {g["id"] for g in gaps}:
            edges.append({
                "from": hid,
                "to": h["addresses_gap"],
                "label": "resolves gap",
                "arrows": "to",
                "color": {"color": "#8B5CF6", "highlight": "#6D28D9"},
                "font": {"size": 10, "color": "#7C3AED", "bold": True},
                "width": 2
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "paper_count": len(papers),
            "claim_count": len(claims),
            "gap_count": len(gaps),
            "hypothesis_count": len(hypotheses)
        }
    }


def generate_interactive_graph_html(graph_data: Dict[str, Any], height_px: int = 680) -> str:
    """
    Produces complete self-contained HTML with Vis.js network, search bar,
    group filters, physics controls, and node detail drawer.
    """
    nodes_json = json.dumps(graph_data["nodes"])
    edges_json = json.dumps(graph_data["edges"])
    summary = graph_data["summary"]

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>PaperPilot Knowledge Graph</title>
      <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js" onerror="document.body.insertAdjacentHTML('afterbegin', '&lt;p style=color:#B91C1C;font-family:sans-serif&gt;The graph library could not be loaded. The first view needs an internet connection.&lt;/p&gt;')"></script>
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          margin: 0;
          padding: 0;
          background-color: #0F172A;
          color: #F8FAFC;
          overflow: hidden;
        }}
        #toolbar {{
          background: #1E293B;
          border-bottom: 1px solid #334155;
          padding: 10px 16px;
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 12px;
          font-size: 13px;
        }}
        .badge {{
          display: inline-flex;
          align-items: center;
          gap: 5px;
          padding: 4px 10px;
          border-radius: 9999px;
          font-weight: 600;
          font-size: 11px;
        }}
        .badge-paper {{ background: rgba(30, 64, 175, 0.4); color: #93C5FD; border: 1px solid #1E40AF; }}
        .badge-claim {{ background: rgba(13, 148, 136, 0.4); color: #5EEAD4; border: 1px solid #0D9488; }}
        .badge-contra {{ background: rgba(225, 29, 72, 0.4); color: #FDA4AF; border: 1px solid #E11D48; }}
        .badge-gap {{ background: rgba(217, 119, 6, 0.4); color: #FCD34D; border: 1px solid #D97706; }}
        .badge-hyp {{ background: rgba(124, 58, 237, 0.4); color: #C4B5FD; border: 1px solid #7C3AED; }}

        .search-box {{
          background: #0F172A;
          border: 1px solid #475569;
          color: #F8FAFC;
          padding: 6px 10px;
          border-radius: 6px;
          font-size: 12px;
          outline: none;
        }}
        .search-box:focus {{ border-color: #38BDF8; }}
        .btn {{
          background: #334155;
          color: #F8FAFC;
          border: none;
          padding: 6px 12px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 12px;
          font-weight: 500;
          transition: background 0.2s;
        }}
        .btn:hover {{ background: #475569; }}
        .btn-active {{ background: #0D9488; color: #FFFFFF; }}

        #network-container {{
          position: relative;
          width: 100%;
          height: {height_px - 55}px;
          background: radial-gradient(circle at center, #1E293B 0%, #0F172A 100%);
        }}
        #network {{
          width: 100%;
          height: 100%;
        }}
        #detail-card {{
          position: absolute;
          bottom: 16px;
          left: 16px;
          max-width: 440px;
          background: rgba(15, 23, 42, 0.94);
          backdrop-filter: blur(8px);
          border: 1px solid #38BDF8;
          border-radius: 8px;
          padding: 14px 18px;
          box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
          display: none;
          z-index: 100;
          font-size: 12px;
          line-height: 1.5;
        }}
        #detail-card h4 {{ margin: 0 0 6px 0; color: #38BDF8; font-size: 14px; }}
        #detail-card p {{ margin: 0 0 6px 0; color: #CBD5E1; }}
        #detail-card .close-btn {{
          position: absolute;
          top: 8px;
          right: 10px;
          cursor: pointer;
          color: #94A3B8;
          font-weight: bold;
        }}
      </style>
    </head>
    <body>
      <div id="toolbar">
        <span style="font-weight: 700; color: #38BDF8; letter-spacing: 0.5px;">KNOWLEDGE GRAPH</span>
        <span class="badge badge-paper">📄 {summary['paper_count']} Papers</span>
        <span class="badge badge-claim">💡 {summary['claim_count']} Claims</span>
        <span class="badge badge-contra">⚡ Contradictions</span>
        <span class="badge badge-gap">⚠️ {summary['gap_count']} Gaps</span>
        <span class="badge badge-hyp">🔮 {summary['hypothesis_count']} Hypotheses</span>
        
        <div style="margin-left: auto; display: flex; gap: 8px; align-items: center;">
          <input type="text" id="searchInput" class="search-box" placeholder="Search claims or papers..." oninput="searchGraph()">
          <button class="btn" onclick="fitView()">⛶ Fit View</button>
          <button class="btn" id="physicsToggle" onclick="togglePhysics()">⏸ Freeze Physics</button>
        </div>
      </div>

      <div id="network-container">
        <div id="network"></div>
        <div id="detail-card">
          <span class="close-btn" onclick="closeDetails()">✕</span>
          <h4 id="detail-title">Node Title</h4>
          <div id="detail-body">Node details will appear here.</div>
        </div>
      </div>

      <script type="text/javascript">
        var rawNodes = {nodes_json};
        var rawEdges = {edges_json};

        var nodes = new vis.DataSet(rawNodes);
        var edges = new vis.DataSet(rawEdges);

        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var physicsEnabled = true;

        var options = {{
          nodes: {{
            shape: 'dot',
            size: 18,
            font: {{
              size: 12,
              color: '#F8FAFC',
              face: 'Segoe UI, Helvetica, Arial'
            }},
            borderWidth: 2,
            shadow: true
          }},
          edges: {{
            width: 1.5,
            smooth: {{
              type: 'cubicBezier',
              forceDirection: 'none',
              roundness: 0.35
            }},
            shadow: true
          }},
          physics: {{
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {{
              gravitationalConstant: -70,
              centralGravity: 0.015,
              springLength: 130,
              springConstant: 0.08,
              damping: 0.85
            }},
            stabilization: {{ iterations: 120 }}
          }},
          interaction: {{
            hover: true,
            tooltipDelay: 150,
            zoomView: true,
            dragView: true
          }}
        }};

        var network = new vis.Network(container, data, options);

        network.on("click", function (params) {{
          if (params.nodes.length > 0) {{
            var nodeId = params.nodes[0];
            var clickedNode = nodes.get(nodeId);
            if (clickedNode) {{
              showDetails(clickedNode);
            }}
          }} else {{
            closeDetails();
          }}
        }});

        function showDetails(node) {{
          var card = document.getElementById("detail-card");
          var titleEl = document.getElementById("detail-title");
          var bodyEl = document.getElementById("detail-body");
          
          titleEl.innerHTML = (node.label || node.id).replace(/\\n/g, ' ');
          bodyEl.innerHTML = node.title || "<p>No further metadata available.</p>";
          card.style.display = "block";
        }}

        function closeDetails() {{
          document.getElementById("detail-card").style.display = "none";
        }}

        function fitView() {{
          network.fit({{ animation: {{ duration: 700, easingFunction: 'easeInOutQuad' }} }});
        }}

        function togglePhysics() {{
          physicsEnabled = !physicsEnabled;
          network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
          var btn = document.getElementById("physicsToggle");
          btn.innerHTML = physicsEnabled ? "⏸ Freeze Physics" : "▶ Enable Physics";
        }}

        function searchGraph() {{
          var q = document.getElementById("searchInput").value.toLowerCase().trim();
          if (!q) {{
            nodes.forEach(function(n) {{
              nodes.update({{ id: n.id, hidden: false, opacity: 1 }});
            }});
            return;
          }}

          nodes.forEach(function(n) {{
            var label = (n.label || '').toLowerCase();
            var title = (n.title || '').toLowerCase();
            var match = label.includes(q) || title.includes(q);
            nodes.update({{ id: n.id, hidden: !match }});
          }});
        }}
      </script>
    </body>
    </html>
    """
    return html_content
