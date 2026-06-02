"""
HTML dashboard generator for layer sweep results.

Reads artifacts produced by src/layer_sweep.run_layer_sweep() and produces
a single self-contained HTML file with:
- Layer accuracy curve and overlap curve (top)
- Per-layer tab selector
- Metrics cards + histogram + PCA scatter for selected layer
- Summary table across all layers

Usage:
    from src.dashboard import generate_dashboard
    generate_dashboard("../data/sweep_base", out_filename="report.html")

The dashboard reads sweep_metadata.json and each layer{i}/plot_data.json,
so it can be regenerated without rerunning the model sweep.
"""

import os
import json
from pathlib import Path

def _load_sweep_artifacts(sweep_dir: str) -> dict:
    """Load sweep metadata and per-layer plot data into a single dict."""
    sweep_path = Path(sweep_dir)
    with open(sweep_path / "sweep_metadata.json") as f:
        metadata = json.load(f)

    layer_data = {}
    for layer in metadata["layers"]:
        plot_path = sweep_path / f"layer{layer}" / "plot_data.json"
        with open(plot_path) as f:
            plot_data = json.load(f)
        layer_data[str(layer)] = {
            **metadata["metrics"][str(layer)],
            **plot_data,
        }

    return {
        "model_id": metadata["model_id"],
        "layers":   metadata["layers"],
        "data":     layer_data,
    }


def _render_html(payload: dict) -> str:
    """Build the dashboard HTML from loaded sweep artifacts."""
    data_json   = json.dumps(payload["data"])
    layers_json = json.dumps(payload["layers"])
    model_id    = payload["model_id"]

    # The HTML/CSS/JS is intentionally inlined here so the output is a single
    # portable file. Plotly is loaded from CDN. The JS handles tab switching
    # and per-layer figure updates by calling Plotly.react with precomputed data.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Unlearning Probes — Layer Sweep</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #0f1117;
         color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 1.3rem; font-weight: 600; color: #94a3b8;
        letter-spacing: 0.05em; text-transform: uppercase;
        margin-bottom: 6px; padding-bottom: 12px; border-bottom: 1px solid #1e293b; }}
  .subtitle {{ color: #64748b; font-size: 0.8rem; margin-bottom: 24px; }}
  .section-title {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase;
                    letter-spacing: 0.1em; margin-bottom: 10px; }}
  #curve-card, #overlap-card {{ background: #1e293b; border-radius: 8px;
                 padding: 16px; margin-bottom: 28px; }}
  .layer-selector {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; }}
  .layer-btn {{ padding: 6px 14px; border-radius: 4px; border: 1px solid #334155;
                background: #1e293b; color: #94a3b8;
                font-family: inherit; font-size: 0.8rem;
                cursor: pointer; transition: all 0.15s; }}
  .layer-btn:hover {{ border-color: #3b82f6; color: #e2e8f0; }}
  .layer-btn.active {{ background: #3b82f6; border-color: #3b82f6;
                       color: white; font-weight: 600; }}
  .plots-row {{ display: grid; grid-template-columns: 1fr 1fr;
                gap: 16px; margin-bottom: 20px; }}
  .plot-card {{ background: #1e293b; border-radius: 8px; padding: 16px; }}
  .metrics-row {{ display: grid; grid-template-columns: repeat(4, 1fr);
                  gap: 12px; margin-bottom: 28px; }}
  .metric-card {{ background: #1e293b; border-radius: 8px; padding: 16px;
                  text-align: center; }}
  .metric-label {{ font-size: 0.7rem; color: #64748b; text-transform: uppercase;
                   letter-spacing: 0.08em; margin-bottom: 6px; }}
  .metric-value {{ font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }}
  .metric-value.good {{ color: #22c55e; }}
  .metric-value.mid  {{ color: #f59e0b; }}
  .metric-value.bad  {{ color: #ef4444; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem;
           background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th {{ background: #0f172a; color: #64748b; text-transform: uppercase;
        font-size: 0.7rem; letter-spacing: 0.08em;
        padding: 10px 14px; text-align: right; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 9px 14px; border-top: 1px solid #0f172a;
        text-align: right; color: #cbd5e1; }}
  td:first-child {{ text-align: left; color: #94a3b8; }}
  tr {{ cursor: pointer; }}
  tr.best-row td {{ color: #22c55e; font-weight: 600; }}
  tr:hover td {{ background: #243347; }}
</style>
</head>
<body>

<h1>Unlearning Probes — Layer Sweep</h1>
<div class="subtitle">Model: {model_id}</div>

<div id="curve-card">
  <div class="section-title">Probe Accuracy Across Layers</div>
  <div id="curve-plot"></div>
</div>

<div id="overlap-card">
  <div class="section-title">Projection Overlap Across Layers</div>
  <div id="overlap-plot"></div>
</div>

<div class="section-title">Layer Inspector</div>
<div class="layer-selector" id="layer-buttons"></div>

<div class="metrics-row" id="metrics-row"></div>

<div class="plots-row">
  <div class="plot-card">
    <div class="section-title">Projection Histogram</div>
    <div id="hist-plot"></div>
  </div>
  <div class="plot-card">
    <div class="section-title">PCA Scatter</div>
    <div id="pca-plot"></div>
  </div>
</div>

<div class="section-title" style="margin-bottom: 10px">All Layers Summary</div>
<table>
  <thead>
    <tr><th>Layer</th><th>Train Acc</th><th>Test Acc</th>
        <th>Sep. Gap</th><th>Overlap</th></tr>
  </thead>
  <tbody id="table-body"></tbody>
</table>

<script>
const DATA   = {data_json};
const LAYERS = {layers_json};
const bestLayer = LAYERS.reduce((a, b) =>
  DATA[a].test_accuracy >= DATA[b].test_accuracy ? a : b);
const minOverlapLayer = LAYERS.reduce((a, b) =>
  DATA[a].overlap_ratio <= DATA[b].overlap_ratio ? a : b);

// Shared layout fragments
const baseLayout = {{
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#94a3b8', size: 11 }},
  legend: {{ bgcolor: 'transparent' }},
  margin: {{ t: 10, b: 40, l: 50, r: 20 }},
}};

// Accuracy curve
Plotly.newPlot('curve-plot', [
  {{ x: LAYERS, y: LAYERS.map(l => DATA[l].train_accuracy),
     mode: 'lines+markers', name: 'Train',
     line: {{ color: '#f97316', dash: 'dash' }}, marker: {{ size: 7 }} }},
  {{ x: LAYERS, y: LAYERS.map(l => DATA[l].test_accuracy),
     mode: 'lines+markers', name: 'Test',
     line: {{ color: '#3b82f6' }}, marker: {{ size: 7 }} }},
  {{ x: [bestLayer], y: [DATA[bestLayer].test_accuracy],
     mode: 'markers', name: `Best (layer ${{bestLayer}})`,
     marker: {{ color: '#22c55e', size: 14, symbol: 'star' }} }},
], {{
  ...baseLayout,
  shapes: [{{ type: 'line', x0: LAYERS[0], x1: LAYERS[LAYERS.length-1],
              y0: 0.5, y1: 0.5,
              line: {{ color: '#475569', dash: 'dot', width: 1 }} }}],
  xaxis: {{ title: 'Layer', tickmode: 'linear', dtick: 1,
            color: '#64748b', gridcolor: '#1e293b' }},
  yaxis: {{ title: 'Accuracy', range: [0.4, 1.05],
            color: '#64748b', gridcolor: '#1e293b' }},
  height: 280,
  legend: {{ x: 0.01, y: 0.05, bgcolor: 'transparent' }},
}}, {{responsive: true, displayModeBar: false}});

// Overlap curve (lower is better separation)
Plotly.newPlot('overlap-plot', [
  {{ x: LAYERS, y: LAYERS.map(l => DATA[l].overlap_ratio),
     mode: 'lines+markers', name: 'Overlap ratio',
     line: {{ color: '#a855f7' }}, marker: {{ size: 7 }} }},
  {{ x: [minOverlapLayer], y: [DATA[minOverlapLayer].overlap_ratio],
     mode: 'markers', name: `Min overlap (layer ${{minOverlapLayer}})`,
     marker: {{ color: '#22c55e', size: 14, symbol: 'star' }} }},
], {{
  ...baseLayout,
  xaxis: {{ title: 'Layer', tickmode: 'linear', dtick: 1,
            color: '#64748b', gridcolor: '#1e293b' }},
  yaxis: {{ title: 'Overlap ratio', range: [0, 1.05],
            color: '#64748b', gridcolor: '#1e293b' }},
  height: 240,
  legend: {{ x: 0.01, y: 0.95, bgcolor: 'transparent' }},
}}, {{responsive: true, displayModeBar: false}});

// Buttons
const btnContainer = document.getElementById('layer-buttons');
LAYERS.forEach(l => {{
  const btn = document.createElement('button');
  btn.className = 'layer-btn' + (l == bestLayer ? ' active' : '');
  btn.textContent = `Layer ${{l}}`;
  btn.id = `btn-${{l}}`;
  btn.onclick = () => selectLayer(l);
  btnContainer.appendChild(btn);
}});

// Summary table
const tbody = document.getElementById('table-body');
LAYERS.forEach(l => {{
  const d = DATA[l];
  const row = document.createElement('tr');
  if (l == bestLayer) row.className = 'best-row';
  row.innerHTML = `
    <td>Layer ${{l}}${{l == bestLayer ? ' ★' : ''}}</td>
    <td>${{d.train_accuracy.toFixed(3)}}</td>
    <td>${{d.test_accuracy.toFixed(3)}}</td>
    <td>${{d.separation_gap.toFixed(3)}}</td>
    <td>${{d.overlap_ratio.toFixed(3)}}</td>`;
  row.onclick = () => selectLayer(l);
  tbody.appendChild(row);
}});

function gradeAcc(v) {{
  if (v >= 0.9)  return 'good';
  if (v >= 0.75) return 'mid';
  return 'bad';
}}

function selectLayer(layer) {{
  // Highlight button
  LAYERS.forEach(l => {{
    document.getElementById(`btn-${{l}}`).className =
      'layer-btn' + (l == layer ? ' active' : '');
  }});

  const d = DATA[layer];

  // Metrics
  document.getElementById('metrics-row').innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Train Accuracy</div>
      <div class="metric-value ${{gradeAcc(d.train_accuracy)}}">${{d.train_accuracy.toFixed(3)}}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Test Accuracy</div>
      <div class="metric-value ${{gradeAcc(d.test_accuracy)}}">${{d.test_accuracy.toFixed(3)}}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Separation Gap</div>
      <div class="metric-value">${{d.separation_gap.toFixed(3)}}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Overlap Ratio</div>
      <div class="metric-value ${{d.overlap_ratio < 0.1 ? 'good' : d.overlap_ratio < 0.3 ? 'mid' : 'bad'}}">${{d.overlap_ratio.toFixed(3)}}</div>
    </div>`;

  // Histogram
  const h = d.histogram;
  Plotly.react('hist-plot', [
    {{ x: h.bin_centers, y: h.forget_counts, type: 'bar',
       name: 'Forget', marker: {{ color: 'rgba(239,68,68,0.65)' }} }},
    {{ x: h.bin_centers, y: h.retain_counts, type: 'bar',
       name: 'Retain', marker: {{ color: 'rgba(59,130,246,0.65)' }} }},
  ], {{
    ...baseLayout,
    barmode: 'overlay',
    xaxis: {{ title: 'Projection value', color: '#64748b', gridcolor: '#0f172a' }},
    yaxis: {{ title: 'Count', color: '#64748b', gridcolor: '#0f172a' }},
    height: 300,
  }}, {{responsive: true, displayModeBar: false}});

  // PCA scatter
  const p = d.pca;
  Plotly.react('pca-plot', [
    {{ x: p.forget_x, y: p.forget_y, mode: 'markers',
       name: 'Forget', marker: {{ color: 'rgba(239,68,68,0.55)', size: 5 }} }},
    {{ x: p.retain_x, y: p.retain_y, mode: 'markers',
       name: 'Retain', marker: {{ color: 'rgba(59,130,246,0.55)', size: 5 }} }},
  ], {{
    ...baseLayout,
    xaxis: {{ title: `PC1 (${{(p.var_pc1*100).toFixed(1)}}% var)`,
              color: '#64748b', gridcolor: '#0f172a' }},
    yaxis: {{ title: `PC2 (${{(p.var_pc2*100).toFixed(1)}}% var)`,
              color: '#64748b', gridcolor: '#0f172a' }},
    height: 300,
  }}, {{responsive: true, displayModeBar: false}});
}}

selectLayer(bestLayer);
</script>
</body>
</html>"""


def generate_dashboard(
    sweep_dir: str,
    out_filename: str = "dashboard.html",
) -> str:
    """
    Generate a self-contained HTML dashboard from saved sweep artifacts.

    Args:
        sweep_dir:     directory containing sweep_metadata.json and layer{i}/ subdirs
        out_filename:  name of the output HTML file (placed inside sweep_dir)

    Returns:
        path to the generated HTML file
    """
    payload = _load_sweep_artifacts(sweep_dir)
    html = _render_html(payload)

    out_path = os.path.join(sweep_dir, out_filename)
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Dashboard written to {out_path}")
    return out_path