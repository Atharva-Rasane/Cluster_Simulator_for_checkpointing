#!/usr/bin/env python3

"""Build a structured simulation timeline and an interactive Plotly dashboard."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGE_SPECS: dict[str, dict[str, Any]] = {
    "COLLECTIVE": {
        "resource": "Network",
        "label": "Ring setup",
        "short": "Ring",
        "color": "#64748b",
        "starts": {"INIT"},
        "ends": {"READY"},
    },
    "RECOVERY": {
        "resource": "Network",
        "label": "Checkpoint lookup / recovery",
        "short": "Recover",
        "color": "#06b6d4",
        "starts": {"START"},
        "ends": {"END", "MISS"},
    },
    "TRAIN/FORWARD": {
        "resource": "GPU",
        "label": "Forward pass",
        "short": "F",
        "color": "#3b82f6",
        "starts": {"START"},
        "ends": {"END"},
    },
    "TRAIN/BACKWARD": {
        "resource": "GPU",
        "label": "Backward pass",
        "short": "B",
        "color": "#8b5cf6",
        "starts": {"START"},
        "ends": {"END"},
    },
    "GRADIENT_SYNC": {
        "resource": "Network",
        "label": "Gradient synchronization",
        "short": "G",
        "color": "#10b981",
        "starts": {"START"},
        "ends": {"END"},
    },
    "UPDATE": {
        "resource": "GPU",
        "label": "Parameter update",
        "short": "U",
        "color": "#f59e0b",
        "starts": {"START"},
        "ends": {"END"},
    },
    "CHECKPOINT/GPU2RAM": {
        "resource": "DRAM",
        "label": "GPU → DRAM checkpoint copy",
        "short": "C",
        "color": "#14b8a6",
        "starts": {"START"},
        "ends": {"END"},
    },
    "CHECKPOINT/NETWORK": {
        "resource": "Network",
        "label": "Checkpoint transfer",
        "short": "Tx",
        "color": "#ec4899",
        "starts": {"START"},
        "ends": {"SENT"},
    },
    "CHECKPOINT/OBJECT": {
        "resource": "Object Store",
        "label": "Object-store persistence",
        "short": "Store",
        "color": "#d97706",
        "starts": {"START"},
        "ends": {"COMMIT"},
    },
    "CHECKPOINT/DRAM": {
        "resource": "DRAM",
        "label": "Local DRAM checkpoint",
        "short": "C",
        "color": "#14b8a6",
        "starts": {"START"},
        "ends": {"END"},
    },
    "CHECKPOINT/SSD": {
        "resource": "SSD",
        "label": "Local SSD checkpoint",
        "short": "SSD",
        "color": "#a855f7",
        "starts": {"START"},
        "ends": {"END"},
    },
}

RESOURCE_ORDER = ["Status", "GPU", "Network", "DRAM", "SSD", "Object Store"]


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def _span_key(marker: dict[str, Any]) -> tuple[Any, ...]:
    return (
        marker.get("variant"),
        int(marker.get("attempt", 0)),
        marker.get("host"),
        int(marker.get("rank", 0)),
        marker.get("iteration"),
        marker.get("component"),
        marker.get("thread"),
    )


def _make_span(
    start: dict[str, Any],
    end: dict[str, Any],
    spec: dict[str, Any],
    *,
    status: str = "completed",
) -> dict[str, Any]:
    start_s = float(start["monotonic_s"])
    end_s = max(start_s, float(end["monotonic_s"]))
    iteration = start.get("iteration")
    suffix = "" if iteration is None else str(iteration)
    details = str(start.get("message", ""))
    end_details = str(end.get("message", ""))
    if end_details and end_details != details:
        details = f"{details}; {end_details}" if details else end_details
    return {
        "variant": str(start["variant"]),
        "attempt": int(start.get("attempt", 0)),
        "host": str(start.get("host", "unknown")),
        "rank": int(start.get("rank", 0)),
        "iteration": iteration,
        "component": str(start["component"]),
        "resource": str(spec["resource"]),
        "label": str(spec["label"]),
        "short": f"{spec['short']}{suffix}",
        "color": str(spec["color"]),
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "status": status,
        "details": details,
    }


def _aggregate_gradient_buckets(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    for event in events:
        if event["component"] != "GRADIENT_SYNC":
            retained.append(event)
            continue
        grouped[
            (
                event["variant"],
                event["attempt"],
                event["host"],
                event["rank"],
                event["iteration"],
                event["status"],
            )
        ].append(event)

    for bucket_events in grouped.values():
        bucket_events.sort(key=lambda event: event["start_s"])
        combined = dict(bucket_events[0])
        combined["start_s"] = min(event["start_s"] for event in bucket_events)
        combined["end_s"] = max(event["end_s"] for event in bucket_events)
        combined["duration_s"] = combined["end_s"] - combined["start_s"]
        combined["details"] = (
            f"{len(bucket_events)} gradient bucket(s); "
            f"communication span={combined['duration_s']:.3f}s"
        )
        retained.append(combined)
    return retained


def _load_variant(
    variant_directory: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    for marker_file in sorted(
        variant_directory.glob("attempt-*/rank-*-timeline.jsonl")
    ):
        markers.extend(_read_json_lines(marker_file))
    markers.sort(key=lambda marker: float(marker.get("monotonic_s", 0.0)))

    supervisor_events = _read_json_lines(
        variant_directory / "supervisor-events.jsonl"
    )
    attempt_ends = {
        int(event["attempt"]): float(event["restart_start_monotonic_s"])
        for event in supervisor_events
        if "attempt" in event and "restart_start_monotonic_s" in event
    }

    open_spans: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    attempt_last_marker: dict[int, float] = defaultdict(float)

    for marker in markers:
        attempt = int(marker.get("attempt", 0))
        marker_s = float(marker.get("monotonic_s", 0.0))
        attempt_last_marker[attempt] = max(attempt_last_marker[attempt], marker_s)
        component = str(marker.get("component", ""))
        marker_event = str(marker.get("event", ""))

        if component == "FAILURE":
            points.append(
                {
                    "variant": str(marker.get("variant", summary["variant"])),
                    "attempt": attempt,
                    "host": str(marker.get("host", "unknown")),
                    "rank": int(marker.get("rank", 0)),
                    "iteration": marker.get("iteration"),
                    "time_s": marker_s,
                    "label": f"{marker_event.title()} failure",
                    "details": str(marker.get("message", "")),
                    "color": "#dc2626",
                    "symbol": "x",
                }
            )
            continue

        spec = STAGE_SPECS.get(component)
        if spec is None:
            continue
        key = _span_key(marker)
        if marker_event in spec["starts"]:
            open_spans[key].append(marker)
        elif marker_event in spec["ends"] and open_spans.get(key):
            start = open_spans[key].pop()
            events.append(_make_span(start, marker, spec))

    for starts in open_spans.values():
        for start in starts:
            attempt = int(start.get("attempt", 0))
            end_s = attempt_ends.get(attempt, attempt_last_marker.get(attempt, 0.0))
            if end_s <= float(start["monotonic_s"]):
                continue
            end = dict(start)
            end["monotonic_s"] = end_s
            end["message"] = "interrupted by failed attempt"
            events.append(
                _make_span(
                    start,
                    end,
                    STAGE_SPECS[str(start["component"])],
                    status="interrupted",
                )
            )

    for restart in supervisor_events:
        if "restart_start_monotonic_s" not in restart:
            continue
        host = str(restart.get("failed_host", "unknown"))
        rank = int(restart.get("failed_rank", 0))
        start_s = float(restart["restart_start_monotonic_s"])
        end_s = float(restart["restart_end_monotonic_s"])
        events.append(
            {
                "variant": str(summary["variant"]),
                "attempt": int(restart.get("attempt", 0)),
                "host": host,
                "rank": rank,
                "iteration": restart.get("failure_iteration"),
                "component": "SUPERVISOR/RESTART",
                "resource": "Status",
                "label": f"{str(restart.get('failure_type', 'process')).title()} restart",
                "short": "Restart",
                "color": "#94a3b8",
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "status": "completed",
                "details": (
                    f"Job-wide restart after {restart.get('failure_type', 'process')} "
                    f"failure on {host}"
                ),
            }
        )

    events = _aggregate_gradient_buckets(events)
    all_times = [event["start_s"] for event in events]
    all_times.extend(point["time_s"] for point in points)
    origin_s = min(all_times, default=0.0)

    for event in events:
        event["start_s"] -= origin_s
        event["end_s"] -= origin_s
        event["lane"] = f"{event['host']} · {event['resource']}"
    for point in points:
        point["time_s"] -= origin_s
        point["lane"] = f"{point['host']} · Status"

    ranks = sorted(
        {(event["rank"], event["host"]) for event in events}
        | {(point["rank"], point["host"]) for point in points}
    )
    used_resources = {
        (event["host"], event["resource"]) for event in events
    } | {(point["host"], "Status") for point in points}
    lanes = [
        f"{host} · {resource}"
        for _, host in ranks
        for resource in RESOURCE_ORDER
        if (host, resource) in used_resources
    ]

    attempt_count = int(summary.get("attempts", 1))
    return {
        "variant": str(summary["variant"]),
        "events": sorted(events, key=lambda event: (event["start_s"], event["rank"])),
        "points": sorted(points, key=lambda point: point["time_s"]),
        "lanes": lanes,
        "attempts": attempt_count,
        "final_attempt": attempt_count - 1,
        "runtime_s": float(summary.get("wall_runtime_s", 0.0)),
        "failures": int(summary.get("failures_total", 0)),
        "target_reached": bool(summary.get("target_reached", False)),
    }


def _dashboard_html(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mininet distributed-training timeline</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; color: #172033; background: #f5f7fb; }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .subtitle, .guide {{ color: #5b6475; font-size: 14px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: end; margin: 20px 0 12px; }}
    label {{ display: grid; gap: 5px; color: #5b6475; font-size: 12px; font-weight: 650; }}
    select {{ min-width: 190px; padding: 8px 10px; border: 1px solid #cbd2df; border-radius: 7px; background: white; }}
    .cards {{ display: flex; gap: 10px; flex-wrap: wrap; margin-left: auto; }}
    .card {{ min-width: 105px; padding: 8px 12px; border: 1px solid #e0e5ee; border-radius: 8px; background: white; }}
    .card span {{ display: block; color: #6b7280; font-size: 11px; }}
    .card strong {{ font-size: 16px; }}
    #timeline {{ background: white; border: 1px solid #e0e5ee; border-radius: 10px; min-height: 560px; }}
    .guide {{ margin-top: 12px; line-height: 1.5; }}
    code {{ background: #e9edf5; border-radius: 4px; padding: 2px 5px; }}
  </style>
</head>
<body>
<main>
  <h1>Distributed-training execution timeline</h1>
  <div class="subtitle">Drag to zoom, double-click to reset, and hover over a block for exact timing and stage details.</div>
  <div class="toolbar">
    <label>EXPERIMENT
      <select id="variant"></select>
    </label>
    <label>ATTEMPT
      <select id="attempt"></select>
    </label>
    <div class="cards">
      <div class="card"><span>Runtime</span><strong id="runtime">–</strong></div>
      <div class="card"><span>Attempts</span><strong id="attempts">–</strong></div>
      <div class="card"><span>Failures</span><strong id="failures">–</strong></div>
      <div class="card"><span>Target</span><strong id="target">–</strong></div>
    </div>
  </div>
  <div id="timeline"></div>
  <div class="guide">
    Normal iteration flow is <code>Forward → Backward + Gradient sync → Update</code>.
    The Network lane makes communication overlap visible. Checkpointing continues as
    <code>GPU → DRAM → Network → Object Store</code>. SSD appears only when a local-SSD
    worker path is used. Interrupted blocks and restart time remain visible for failed attempts.
  </div>
</main>
<script>
const dashboard = {encoded};
const variantSelect = document.getElementById("variant");
const attemptSelect = document.getElementById("attempt");
const categories = [...new Set(Object.values(dashboard.variants).flatMap(v => v.events.map(e => e.label)))];

for (const name of Object.keys(dashboard.variants)) {{
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  variantSelect.appendChild(option);
}}

function populateAttempts() {{
  const view = dashboard.variants[variantSelect.value];
  attemptSelect.replaceChildren();
  const choices = [["all", "All attempts"], ["final", `Final attempt (${{view.final_attempt}})`]];
  for (let i = 0; i < view.attempts; i++) choices.push([String(i), `Attempt ${{i}}`]);
  for (const [value, label] of choices) {{
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    attemptSelect.appendChild(option);
  }}
}}

function selected(event, choice, finalAttempt) {{
  if (choice === "all") return true;
  if (choice === "final") return event.attempt === finalAttempt;
  return event.attempt === Number(choice);
}}

function draw() {{
  const view = dashboard.variants[variantSelect.value];
  const choice = attemptSelect.value || "all";
  const events = view.events.filter(e => selected(e, choice, view.final_attempt));
  const points = view.points.filter(e => selected(e, choice, view.final_attempt));
  const activeLanes = new Set([...events.map(e => e.lane), ...points.map(e => e.lane)]);
  const lanes = view.lanes.filter(lane => activeLanes.has(lane));
  const traces = [];

  for (const label of categories) {{
    const rows = events.filter(e => e.label === label);
    if (!rows.length) continue;
    traces.push({{
      type: "bar",
      orientation: "h",
      name: label,
      x: rows.map(e => Math.max(e.duration_s, 0.0005)),
      base: rows.map(e => e.start_s),
      y: rows.map(e => e.lane),
      marker: {{
        color: rows.map(e => e.status === "interrupted" ? "#fecaca" : e.color),
        line: {{ color: rows.map(e => e.status === "interrupted" ? "#dc2626" : "#ffffff"), width: 1 }}
      }},
      text: rows.map(e => e.short),
      textposition: "inside",
      insidetextanchor: "middle",
      customdata: rows.map(e => [e.host, e.resource, e.iteration ?? "–", e.attempt, e.duration_s, e.status, e.details]),
      hovertemplate:
        "<b>%{{fullData.name}}</b><br>" +
        "Node: %{{customdata[0]}} · %{{customdata[1]}}<br>" +
        "Iteration: %{{customdata[2]}} · Attempt: %{{customdata[3]}}<br>" +
        "Start: %{{base:.3f}}s · Duration: %{{customdata[4]:.3f}}s<br>" +
        "Status: %{{customdata[5]}}<br>%{{customdata[6]}}<extra></extra>"
    }});
  }}

  if (points.length) {{
    traces.push({{
      type: "scatter",
      mode: "markers",
      name: "Failure",
      x: points.map(p => p.time_s),
      y: points.map(p => p.lane),
      marker: {{ color: points.map(p => p.color), symbol: points.map(p => p.symbol), size: 12, line: {{ width: 2 }} }},
      customdata: points.map(p => [p.label, p.host, p.iteration ?? "–", p.attempt, p.details]),
      hovertemplate: "<b>%{{customdata[0]}}</b><br>Node: %{{customdata[1]}}<br>Iteration: %{{customdata[2]}} · Attempt: %{{customdata[3]}}<br>Time: %{{x:.3f}}s<br>%{{customdata[4]}}<extra></extra>"
    }});
  }}

  const height = Math.max(560, 150 + lanes.length * 42);
  Plotly.react("timeline", traces, {{
    barmode: "overlay",
    bargap: 0.28,
    height,
    margin: {{ l: 150, r: 30, t: 55, b: 70 }},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfcfe",
    title: {{ text: `${{view.variant}} · ${{choice === "all" ? "complete execution" : attemptSelect.selectedOptions[0].textContent}}`, x: 0.01, font: {{ size: 16 }} }},
    xaxis: {{ title: "Elapsed time (seconds)", rangemode: "tozero", gridcolor: "#e8ecf3", zeroline: false }},
    yaxis: {{ categoryorder: "array", categoryarray: lanes, autorange: "reversed", gridcolor: "#eef1f6", tickfont: {{ size: 12 }} }},
    legend: {{ orientation: "h", yanchor: "bottom", y: 1.02, x: 0, font: {{ size: 11 }} }},
    hoverlabel: {{ align: "left" }},
    dragmode: "zoom"
  }}, {{
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  }});

  document.getElementById("runtime").textContent = `${{view.runtime_s.toFixed(2)}} s`;
  document.getElementById("attempts").textContent = view.attempts;
  document.getElementById("failures").textContent = view.failures;
  document.getElementById("target").textContent = view.target_reached ? "Reached" : "Missed";
}}

variantSelect.addEventListener("change", () => {{ populateAttempts(); draw(); }});
attemptSelect.addEventListener("change", draw);
populateAttempts();
draw();
</script>
</body>
</html>
"""


def write_timeline_dashboard(
    results_root: Path,
    summaries: list[dict[str, Any]],
) -> Path:
    """Create timeline-data.json and an interactive Plotly HTML dashboard."""
    variants = {
        str(summary["variant"]): _load_variant(
            results_root / str(summary["variant"]), summary
        )
        for summary in summaries
    }
    data = {
        "schema_version": 1,
        "description": "Structured per-node distributed-training timeline",
        "variants": variants,
    }
    data_path = results_root / "timeline-data.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    html_path = results_root / "timeline.html"
    html_path.write_text(_dashboard_html(data), encoding="utf-8")
    return html_path
