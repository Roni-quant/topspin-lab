"""Build standalone HTML report — London 2026 tournament predictions.

Single self-contained HTML file with headline stats, calibration chart,
and every match annotated with prediction + outcome.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "london_2026_predictions.csv"
LONDON_RAW = ROOT / "london_2026_matches.parquet"
OUT = ROOT / "london_2026_report.html"

MODEL_NAME = "Enhanced RF (9 features)"
PROB_COL = "p_rf_enhanced"


def load() -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS)
    raw = pd.read_parquet(LONDON_RAW)[
        ["player_a_id", "player_b_id", "player_a_name", "player_b_name", "result", "games"]
    ]
    df = preds.merge(raw.drop_duplicates(["player_a_id", "player_b_id"]),
                     on=["player_a_id", "player_b_id"], how="left")

    df["pick"] = np.where(df[PROB_COL] >= 0.5, "A", "B")
    df["correct"] = ((df["pick"] == "A") & (df["target"] == 1)) | (
        (df["pick"] == "B") & (df["target"] == 0)
    )
    df["confidence"] = np.where(df[PROB_COL] >= 0.5, df[PROB_COL], 1 - df[PROB_COL])
    df["pick_name"] = np.where(df["pick"] == "A", df["player_a_name"], df["player_b_name"])
    df["actual_name"] = np.where(df["target"] == 1, df["player_a_name"], df["player_b_name"])
    return df


def headline_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    correct = int(df["correct"].sum())
    return {
        "n": n,
        "correct": correct,
        "wrong": n - correct,
        "accuracy": correct / n,
        "mt_acc": df[df["category"] == "MT"]["correct"].mean(),
        "wt_acc": df[df["category"] == "WT"]["correct"].mean(),
        "mt_n": int((df["category"] == "MT").sum()),
        "wt_n": int((df["category"] == "WT").sum()),
        "high_conf_acc": df[df["confidence"] >= 0.75]["correct"].mean(),
        "high_conf_n": int((df["confidence"] >= 0.75).sum()),
        "low_conf_acc": df[df["confidence"] < 0.6]["correct"].mean(),
        "low_conf_n": int((df["confidence"] < 0.6).sum()),
    }


def calibration_buckets(df: pd.DataFrame, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    p = df[PROB_COL].values
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            out.append((edges[b], edges[b+1], 0, 0.0, 0.0))
            continue
        out.append((
            edges[b], edges[b+1], int(mask.sum()),
            float(p[mask].mean()),
            float(df["target"].values[mask].mean()),
        ))
    return out


def render_match_row(r) -> str:
    pa = _html.escape(str(r.player_a_name or f"#{r.player_a_id}"))
    pb = _html.escape(str(r.player_b_name or f"#{r.player_b_id}"))
    pick = _html.escape(str(r.pick_name or ""))
    actual = _html.escape(str(r.actual_name or ""))
    score = _html.escape(str(r.result or ""))
    conf = r.confidence * 100
    prob_a = getattr(r, PROB_COL) * 100
    elo_a = r.elo_a_before
    elo_b = r.elo_b_before
    elo_diff = r.elo_difference

    mark = "✓" if r.correct else "✗"
    cls = "ok" if r.correct else "bad"

    return (
        f'<tr class="{cls}">'
        f'<td class="mark">{mark}</td>'
        f'<td class="cat">{r.category}</td>'
        f'<td class="pa">{pa}</td>'
        f'<td class="vs">vs</td>'
        f'<td class="pb">{pb}</td>'
        f'<td class="score">{score}</td>'
        f'<td class="winner">{actual}</td>'
        f'<td class="pick">{pick}</td>'
        f'<td class="conf"><div class="bar"><div class="fill" style="width:{conf:.1f}%"></div><span>{conf:.0f}%</span></div></td>'
        f'<td class="prob">{prob_a:.0f}%</td>'
        f'<td class="elo">{elo_a:.0f}</td>'
        f'<td class="elo">{elo_b:.0f}</td>'
        f'<td class="elo diff">{elo_diff:+.0f}</td>'
        f"</tr>"
    )


def main() -> None:
    df = load()
    df = df.sort_values(["category", "correct", PROB_COL], ascending=[True, False, False]).reset_index(drop=True)

    s = headline_stats(df)
    cal = calibration_buckets(df)

    # Calibration SVG
    svg_w, svg_h = 520, 280
    pad = 40
    plot_w, plot_h = svg_w - 2 * pad, svg_h - 2 * pad
    bars = []
    line_pts = []
    for i, (lo, hi, n, mean_p, actual) in enumerate(cal):
        if n == 0:
            continue
        x = pad + (lo + hi) / 2 * plot_w
        y_pred = svg_h - pad - mean_p * plot_h
        y_actual = svg_h - pad - actual * plot_h
        bars.append(
            f'<circle cx="{x:.1f}" cy="{y_actual:.1f}" r="{max(3, n**0.5):.1f}" '
            f'fill="#22c55e" opacity="0.75"><title>bin [{lo:.2f},{hi:.2f}) '
            f'n={n} pred={mean_p:.2f} actual={actual:.2f}</title></circle>'
        )
        line_pts.append(f"{x:.1f},{y_pred:.1f}")
    diag = f'<line x1="{pad}" y1="{svg_h - pad}" x2="{svg_w - pad}" y2="{pad}" stroke="#94a3b8" stroke-dasharray="4 4"/>'
    pred_line = f'<polyline points="{" ".join(line_pts)}" stroke="#3b82f6" stroke-width="2" fill="none"/>'

    svg = f"""
    <svg viewBox="0 0 {svg_w} {svg_h}" width="100%" preserveAspectRatio="xMidYMid meet">
      <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" fill="#0f172a" stroke="#1e293b"/>
      {diag}
      {pred_line}
      {''.join(bars)}
      <text x="{svg_w/2}" y="{svg_h - 8}" text-anchor="middle" fill="#cbd5e1" font-size="12">Predicted probability →</text>
      <text x="14" y="{svg_h/2}" text-anchor="middle" fill="#cbd5e1" font-size="12" transform="rotate(-90 14 {svg_h/2})">Actual win rate →</text>
      <text x="{pad}" y="{pad - 10}" fill="#e2e8f0" font-size="13" font-weight="600">Calibration — green = actual, blue line = predicted</text>
    </svg>
    """

    # Confidence histogram
    conf_bins = np.linspace(0.5, 1.0, 11)
    conf_hist_correct = np.zeros(10, dtype=int)
    conf_hist_wrong = np.zeros(10, dtype=int)
    for c, ok in zip(df["confidence"], df["correct"]):
        b = min(int((c - 0.5) / 0.05), 9)
        if ok:
            conf_hist_correct[b] += 1
        else:
            conf_hist_wrong[b] += 1

    hist_w, hist_h = 520, 220
    bar_w = (hist_w - 2 * pad) / 10
    hist_max = max((conf_hist_correct + conf_hist_wrong).max(), 1)
    hist_bars = []
    for i in range(10):
        x = pad + i * bar_w
        total = conf_hist_correct[i] + conf_hist_wrong[i]
        h_total = (total / hist_max) * (hist_h - 2 * pad)
        h_ok = (conf_hist_correct[i] / hist_max) * (hist_h - 2 * pad)
        y_total_top = hist_h - pad - h_total
        y_ok_top = hist_h - pad - h_ok
        # wrong (red) bottom-ish, then correct (green) below it visually stacked
        hist_bars.append(
            f'<rect x="{x+1}" y="{y_total_top}" width="{bar_w-2}" '
            f'height="{h_total - h_ok}" fill="#ef4444" opacity="0.85"/>'
        )
        hist_bars.append(
            f'<rect x="{x+1}" y="{y_ok_top}" width="{bar_w-2}" '
            f'height="{h_ok}" fill="#22c55e" opacity="0.9"><title>conf {0.5+i*0.05:.2f}-{0.5+(i+1)*0.05:.2f}: '
            f'{conf_hist_correct[i]} right, {conf_hist_wrong[i]} wrong</title></rect>'
        )
        hist_bars.append(
            f'<text x="{x + bar_w/2}" y="{hist_h - pad + 14}" text-anchor="middle" fill="#94a3b8" font-size="10">'
            f'{int((0.5+i*0.05)*100)}</text>'
        )

    hist_svg = f"""
    <svg viewBox="0 0 {hist_w} {hist_h}" width="100%" preserveAspectRatio="xMidYMid meet">
      <text x="{pad}" y="{pad - 10}" fill="#e2e8f0" font-size="13" font-weight="600">Predictions by confidence — green = right, red = wrong</text>
      {''.join(hist_bars)}
      <text x="{hist_w/2}" y="{hist_h - 8}" text-anchor="middle" fill="#cbd5e1" font-size="12">Confidence (%) →</text>
    </svg>
    """

    rows_html = "\n".join(render_match_row(r) for r in df.itertuples(index=False))

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>London 2026 — Model Validation Report</title>
<style>
  :root {{
    --bg: #0b1220;
    --panel: #111c34;
    --panel-2: #18243f;
    --text: #e6edf7;
    --muted: #94a3b8;
    --ok: #22c55e;
    --bad: #ef4444;
    --accent: #3b82f6;
    --border: #1f2c4a;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }}
  .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 24px 64px; }}
  header {{
    background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 50%, #7c3aed 100%);
    border-radius: 16px;
    padding: 32px 28px;
    margin-bottom: 24px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
  }}
  h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.01em; }}
  .sub {{ color: rgba(255,255,255,0.85); font-size: 15px; margin: 0; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 22px; }}
  .stat {{ background: rgba(15,23,42,0.55); border-radius: 12px; padding: 14px 16px; }}
  .stat .v {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em; }}
  .stat .l {{ font-size: 12px; color: rgba(226,232,240,0.75); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }}
  .stat.green .v {{ color: #4ade80; }}
  .stat.blue  .v {{ color: #93c5fd; }}
  .stat.pink  .v {{ color: #f472b6; }}
  .stat.amber .v {{ color: #fbbf24; }}

  .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{ .panels {{ grid-template-columns: 1fr; }} }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }}

  .controls {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }}
  .btn {{ background: var(--panel-2); border: 1px solid var(--border); color: var(--text); padding: 8px 14px; border-radius: 999px; cursor: pointer; font-size: 13px; }}
  .btn.active {{ background: var(--accent); border-color: var(--accent); }}
  .legend {{ margin-left: auto; color: var(--muted); font-size: 12px; }}
  .legend .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin: 0 4px 0 12px; vertical-align: middle; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead th {{ position: sticky; top: 0; background: var(--panel-2); color: var(--muted); text-align: left; padding: 10px 10px; border-bottom: 1px solid var(--border); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
  tbody td {{ padding: 9px 10px; border-bottom: 1px solid rgba(255,255,255,0.04); }}
  tbody tr.ok  .mark {{ color: var(--ok);  font-size: 16px; font-weight: 700; }}
  tbody tr.bad .mark {{ color: var(--bad); font-size: 16px; font-weight: 700; }}
  tbody tr.ok  {{ background: linear-gradient(90deg, rgba(34,197,94,0.06), transparent 30%); }}
  tbody tr.bad {{ background: linear-gradient(90deg, rgba(239,68,68,0.07), transparent 30%); }}
  td.cat {{ color: var(--muted); font-weight: 600; }}
  td.pa, td.pb {{ font-weight: 500; }}
  td.pick {{ color: #93c5fd; }}
  td.winner {{ color: #fde68a; }}
  td.vs {{ color: var(--muted); }}
  td.score, td.prob, td.elo {{ font-variant-numeric: tabular-nums; }}
  td.elo.diff {{ color: var(--accent); font-weight: 600; }}
  .bar {{ position: relative; width: 90px; background: #1e293b; border-radius: 4px; height: 16px; overflow: hidden; }}
  .bar .fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }}
  .bar span {{ position: absolute; left: 0; right: 0; top: 0; bottom: 0; text-align: center; font-size: 11px; line-height: 16px; color: #fff; }}
  .table-wrap {{ max-height: 720px; overflow: auto; border-radius: 12px; background: var(--panel); border: 1px solid var(--border); }}

  footer {{ color: var(--muted); font-size: 12px; margin-top: 16px; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>ITTF World Team Championships Finals — London 2026</h1>
    <p class="sub">Model validation on truly unseen tournament. Predictions made with Elo + recent-form features, trained on data strictly before 2026-05-10. Compared to actual outcomes.</p>
    <div class="stats">
      <div class="stat green"><div class="l">Accuracy</div><div class="v">{s['accuracy']*100:.1f}%</div></div>
      <div class="stat blue"><div class="l">Matches scored</div><div class="v">{s['n']}</div></div>
      <div class="stat amber"><div class="l">Correct calls</div><div class="v">{s['correct']}</div></div>
      <div class="stat pink"><div class="l">High-confidence acc (≥75%)</div><div class="v">{s['high_conf_acc']*100:.1f}%</div></div>
      <div class="stat blue"><div class="l">Men's Team (MT)</div><div class="v">{s['mt_acc']*100:.1f}%</div></div>
      <div class="stat pink"><div class="l">Women's Team (WT)</div><div class="v">{s['wt_acc']*100:.1f}%</div></div>
    </div>
  </header>

  <div class="panels">
    <div class="panel">{svg}</div>
    <div class="panel">{hist_svg}</div>
  </div>

  <div class="controls">
    <button class="btn active" data-filter="all">All ({s['n']})</button>
    <button class="btn" data-filter="ok">Correct ({s['correct']})</button>
    <button class="btn" data-filter="bad">Wrong ({s['wrong']})</button>
    <button class="btn" data-filter="mt">Men ({s['mt_n']})</button>
    <button class="btn" data-filter="wt">Women ({s['wt_n']})</button>
    <span class="legend">
      <span class="sw" style="background:#22c55e"></span>correct
      <span class="sw" style="background:#ef4444"></span>wrong
    </span>
  </div>

  <div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th></th><th>Cat</th><th>Player A</th><th></th><th>Player B</th>
        <th>Score</th><th>Actual winner</th><th>Model pick</th>
        <th>Confidence</th><th>P(A)</th><th>Elo A</th><th>Elo B</th><th>ΔElo</th>
      </tr>
    </thead>
    <tbody id="rows">
      {rows_html}
    </tbody>
  </table>
  </div>

  <footer>
    Generated from <code>experiments/validate_london_2026.py</code>.
    Priors frozen at 2026-03-16. Cold-start matches (35) excluded. {s['n']} singles matches.
    Model: {MODEL_NAME}.
  </footer>
</div>
<script>
  const buttons = document.querySelectorAll('.btn');
  const rows = document.querySelectorAll('#rows tr');
  buttons.forEach(b => b.addEventListener('click', () => {{
    buttons.forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const f = b.dataset.filter;
    rows.forEach(r => {{
      let show = true;
      if (f === 'ok')  show = r.classList.contains('ok');
      if (f === 'bad') show = r.classList.contains('bad');
      if (f === 'mt')  show = r.querySelector('.cat').textContent.trim() === 'MT';
      if (f === 'wt')  show = r.querySelector('.cat').textContent.trim() === 'WT';
      r.style.display = show ? '' : 'none';
    }});
  }}));
</script>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
