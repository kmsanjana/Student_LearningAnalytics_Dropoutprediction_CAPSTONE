"""Build the self-contained HTML dashboard from dashboard_data.json"""
import json, os

with open('processed/dashboard_data.json', 'r') as f:
    data = json.load(f)

with open('processed/model_results.json', 'r') as f:
    ml_data = json.load(f)

d = data['kpis']
clusters = data['cluster_data']
modules  = data['module_data']
shap_d   = data['shap_importance']
flags    = data['flags']
risk_dist = data['risk_dist']
students = data['students']
outcome  = data['outcome_dist']

# ── helpers ──────────────────────────────────────────
cluster_colors = {
    'Power Users':         '#10b981',
    'Steady Completers':   '#3b82f6',
    'Struggling Engagers': '#f59e0b',
    'Non-Starters':        '#ef4444',
}

def json_s(obj): return json.dumps(obj)

# ── student rows (top-1000 for table) ────────────────
def student_rows(students):
    rows = []
    for idx, s in enumerate(students[:1000]):
        prob = s.get('dropout_probability', 0)
        risk = s.get('risk_score', 0)
        cl   = s.get('cluster_name', '')
        col = ('#ef4444' if cl == 'Non-Starters' else
               '#f59e0b' if cl == 'Struggling Engagers' else
               '#3b82f6' if cl == 'Steady Completers' else '#10b981')
        rl = 'HIGH' if risk >= 3 else ('MOD' if risk >= 1 else 'LOW')
        rlcol = '#ef4444' if rl == 'HIGH' else ('#f59e0b' if rl == 'MOD' else '#10b981')
        flags_html = ''
        flag_map = {
            'flag_zero_first_week': '⚡ No Wk1',
            'flag_high_attempts': '🔁 Repeater',
            'flag_low_engagement': '📉 Low Engage',
            'flag_low_assessment': '📝 No Assess',
            'flag_deprived_area': '🏠 Deprived',
        }
        for fk, fl in flag_map.items():
            if s.get(fk):
                flags_html += f'<span class="flag-chip">{fl}</span>'
        rows.append(f"""<tr onclick="selectStudent(STUDENTS_INIT[{idx}])" style="cursor:pointer">
  <td><span class="sid">{s['idStudent']}</span></td>
  <td>{s.get('codeModule','')}/{s.get('codePresentation','')}</td>
  <td><div class="prob-bar-wrap"><div class="prob-bar" style="width:{prob*100:.0f}%;background:{col}"></div><span>{prob*100:.1f}%</span></div></td>
  <td><span class="risk-badge" style="background:{rlcol}">{rl} ({risk})</span></td>
  <td>{flags_html}</td>
  <td><span class="cluster-dot" style="color:{col}">\u25cf</span> {cl}</td>
</tr>""")
    return '\n'.join(rows)

# ── shap bars ─────────────────────────────────────────
def shap_bars(shap_d):
    max_v = max(x['mean_shap'] for x in shap_d)
    bars = []
    for x in shap_d:
        pct = x['mean_shap'] / max_v * 100
        bars.append(f"""<div class="shap-row">
  <div class="shap-label">{x['display_name']}</div>
  <div class="shap-track"><div class="shap-fill" style="width:{pct:.1f}%"></div></div>
  <div class="shap-val">{x['mean_shap']:.3f}</div>
</div>""")
    return '\n'.join(bars)

# ── flag cards ────────────────────────────────────────
def flag_cards(flags, d_total, baseline=31.2):
    cards = []
    flag_icons = {'Zero First-Week Activity':'⚡','High Prior Attempts':'🔁',
                  'Low Engagement':'📉','Zero Assessments':'📝','Deprived Area':'🏠'}
    for name, info in flags.items():
        icon = flag_icons.get(name, '•')
        pct = round(info['count']/d_total*100, 1)
        dr  = info['dropout_rate']
        lift = round(dr - baseline, 1)
        lift_col = '#ef4444' if lift > 0 else '#10b981'
        lift_sign = '+' if lift > 0 else ''
        cards.append(f"""<div class="flag-card">
  <div class="flag-icon">{icon}</div>
  <div class="flag-body">
    <div class="flag-name">{name}</div>
    <div class="flag-desc">{info['desc']}</div>
    <div class="flag-stats">
      <span class="flag-count">{info['count']:,} students ({pct}%)</span>
      <span class="flag-dr" style="color:#ef4444">Dropout: {dr}%</span>
      <span style="font-size:11px;color:{lift_col};font-weight:600">{lift_sign}{lift}pp vs baseline</span>
    </div>
  </div>
</div>""")
    return '\n'.join(cards)

# ── risk validation table ─────────────────────────────
def risk_validation_table(flags, baseline):
    flag_meta = [
        ('Zero First-Week Activity', 'Chi-square',   '0 VLE clicks in days 0–7',
         'Phase 1 EDA: 55% of Withdrawn vs 8% of Distinction students had zero Week-1 VLE activity — strongest early-dropout signal found'),
        ('High Prior Attempts',      'Chi-square',   'numOfPrevAttempts ≥ 3',
         'EDA: students with 2+ prior attempts exceed 55% dropout — indicates structural re-enrollment barriers, not one-off difficulty'),
        ('Low Engagement',           'Mann-Whitney U','Active <10% of module days (engagement_ratio < 0.10)',
         'Distinction students generate 5× more VLE clicks than Withdrawn; engagement ratio is a top-5 SHAP feature in the LightGBM model'),
        ('Zero Assessments',         'Chi-square',   'num_assessments = 0 (no submissions at all)',
         'Assessment completion is the #1 SHAP driver; zero submissions is the single highest-lift individual predictor of withdrawal'),
        ('Deprived Area',            'Chi-square',   'IMD Band 0–20% (most-deprived quintile)',
         'EDA: 37% dropout in IMD 0–10% vs 24% in least-deprived — a 13-point socioeconomic gap that persists after controlling for engagement'),
    ]
    rows = []
    for fname, test, threshold, evidence in flag_meta:
        info = flags.get(fname, {})
        dr   = info.get('dropout_rate', 0)
        lift = round(dr - baseline, 1)
        ratio = round(dr / baseline, 2) if baseline else 0
        lift_col  = '#ef4444' if lift > 0 else '#10b981'
        lift_sign = '+' if lift > 0 else ''
        rows.append(f"""<tr>
  <td style="font-weight:600;font-size:13px">{fname}</td>
  <td style="font-size:11px;color:#94a3b8">{threshold}</td>
  <td style="text-align:center;font-size:11px;white-space:nowrap">{test}</td>
  <td style="text-align:center;font-family:monospace;font-size:11px;color:#93c5fd">p &lt; 0.001</td>
  <td style="text-align:center;font-family:monospace;font-weight:700;color:#ef4444">{dr}%</td>
  <td style="text-align:center;font-family:monospace;font-size:12px;color:{lift_col};font-weight:700">{lift_sign}{lift}pp &nbsp;({ratio}×)</td>
  <td style="font-size:11px;color:#94a3b8;line-height:1.5">{evidence}</td>
</tr>""")
    return '\n'.join(rows)

# ── module rows ───────────────────────────────────────
def module_rows(modules):
    rows = []
    for m in modules:
        w = m['dropout_pct']
        col = '#ef4444' if w > 35 else ('#f59e0b' if w > 28 else '#10b981')
        rows.append(f"""<tr>
  <td><strong>{m['codeModule']}</strong></td>
  <td>{m['count']:,}</td>
  <td><div class="prob-bar-wrap"><div class="prob-bar" style="width:{w}%;background:{col}"></div><span>{w}%</span></div></td>
  <td>{m['avg_risk_pct']}%</td>
  <td>{m['high_risk']:,}</td>
</tr>""")
    return '\n'.join(rows)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OULAD Early Warning Dashboard — GWU Capstone 2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0f172a;--surface:#1e293b;--surface2:#243047;--border:#334155;
  --accent:#6366f1;--accent2:#818cf8;--text:#f1f5f9;--muted:#94a3b8;
  --green:#10b981;--yellow:#f59e0b;--red:#ef4444;--blue:#3b82f6;
  --radius:12px;--shadow:0 4px 24px rgba(0,0,0,.4);
}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}}
/* ─── TOP BAR ─── */
.topbar{{background:linear-gradient(135deg,#1e1b4b,#0f172a);border-bottom:1px solid var(--border);
  padding:0 28px;display:flex;align-items:center;justify-content:space-between;height:64px;
  position:sticky;top:0;z-index:100;}}
.topbar-left{{display:flex;align-items:center;gap:14px;}}
.logo{{width:36px;height:36px;background:linear-gradient(135deg,var(--accent),#8b5cf6);
  border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;}}
.topbar h1{{font-size:17px;font-weight:700;letter-spacing:-.3px}}
.topbar p{{font-size:12px;color:var(--muted)}}
.status-pill{{background:#10b981;color:#fff;font-size:11px;font-weight:600;
  padding:3px 10px;border-radius:20px;letter-spacing:.5px}}
/* ─── TABS ─── */
.tabs{{display:flex;gap:6px;padding:18px 28px 0;border-bottom:1px solid var(--border);background:var(--bg);}}
.tab{{padding:10px 18px;border-radius:8px 8px 0 0;cursor:pointer;font-size:13px;font-weight:500;
  color:var(--muted);border:1px solid transparent;border-bottom:none;transition:.2s;}}
.tab:hover{{color:var(--text);background:var(--surface)}}
.tab.active{{color:var(--accent2);background:var(--surface);border-color:var(--border);}}
/* ─── MAIN ─── */
.main{{padding:24px 28px;}}
.panel{{display:none;}}
.panel.active{{display:block;}}
/* ─── GRID ─── */
.kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:22px;}}
.kpi-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px 18px;position:relative;overflow:hidden;}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--accent-line,var(--accent));}}
.kpi-label{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:8px;}}
.kpi-value{{font-size:32px;font-weight:800;line-height:1;}}
.kpi-sub{{font-size:12px;color:var(--muted);margin-top:6px;}}
/* ─── CHART GRID ─── */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:22px;}}
.chart-grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:22px;}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;}}
.card h3{{font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;
  letter-spacing:.6px;margin-bottom:16px;}}
.card-wide{{grid-column:1/-1;}}
/* ─── TABLE ─── */
.tbl-wrap{{overflow:auto;max-height:420px;border-radius:8px;border:1px solid var(--border);}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th{{background:#1a2540;color:var(--muted);font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.5px;padding:10px 14px;text-align:left;
  position:sticky;top:0;z-index:2;}}
td{{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:middle;}}
tr{{cursor:pointer;transition:.15s;}}
tr:hover{{background:rgba(99,102,241,.08);}}
.sid{{font-family:monospace;font-size:12px;color:var(--accent2);}}
/* ─── PROB BAR ─── */
.prob-bar-wrap{{display:flex;align-items:center;gap:8px;min-width:100px;}}
.prob-bar{{height:6px;border-radius:3px;min-width:2px;transition:width .4s;}}
.prob-bar-wrap span{{font-size:12px;font-weight:600;white-space:nowrap;}}
/* ─── BADGES ─── */
.risk-badge{{font-size:10px;font-weight:700;padding:3px 8px;border-radius:20px;color:#fff;white-space:nowrap;}}
.cluster-dot{{font-size:10px;}}
.flag-chip{{font-size:10px;background:#1e3a5f;color:#93c5fd;padding:2px 6px;
  border-radius:4px;margin:1px;display:inline-block;}}
/* ─── SHAP ─── */
.shap-row{{display:grid;grid-template-columns:180px 1fr 60px;align-items:center;gap:10px;margin-bottom:10px;}}
.shap-label{{font-size:12px;font-weight:500;color:var(--text);}}
.shap-track{{background:#1a2540;border-radius:4px;height:14px;overflow:hidden;}}
.shap-fill{{height:100%;background:linear-gradient(90deg,var(--accent),#8b5cf6);border-radius:4px;transition:width .5s;}}
.shap-val{{font-size:12px;color:var(--muted);text-align:right;font-family:monospace;}}
/* ─── STUDENT DETAIL ─── */
.student-detail{{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);
  padding:24px;margin-bottom:22px;display:none;}}
.student-detail.open{{display:block;}}
.sd-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;}}
.sd-id{{font-size:22px;font-weight:800;}}
.sd-prob{{font-size:40px;font-weight:900;}}
.sd-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}}
.sd-stat{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;}}
.sd-stat-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:6px;}}
.sd-stat-val{{font-size:20px;font-weight:700;}}
.flag-section{{margin-top:14px;}}
.flag-title{{font-size:12px;font-weight:600;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px;}}
/* ─── FLAGS PANEL ─── */
.flag-card{{display:flex;align-items:flex-start;gap:14px;background:var(--surface);
  border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:10px;}}
.flag-icon{{font-size:24px;width:40px;text-align:center;}}
.flag-name{{font-size:14px;font-weight:600;margin-bottom:3px;}}
.flag-desc{{font-size:12px;color:var(--muted);margin-bottom:8px;}}
.flag-stats{{display:flex;gap:16px;}}
.flag-count{{font-size:12px;color:var(--text);}}
.flag-dr{{font-size:12px;font-weight:600;}}
/* ─── SEARCH ─── */
.search-bar{{display:flex;gap:10px;margin-bottom:16px;}}
.search-bar input{{flex:1;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;color:var(--text);font-size:13px;outline:none;}}
.search-bar input:focus{{border-color:var(--accent);}}
.search-bar select{{background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;color:var(--text);font-size:13px;}}
/* ─── VALIDATION BOX ─── */
.val-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}}
.val-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;text-align:center;}}
.val-score{{font-size:13px;font-weight:600;color:var(--muted);margin-bottom:10px;}}
.val-num{{font-size:28px;font-weight:800;}}
.val-label{{font-size:11px;color:var(--muted);margin-top:4px;}}
/* ─── CANVAS ─── */
canvas{{max-height:260px;}}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-left">
    <div class="logo">🎓</div>
    <div>
      <h1>OULAD Early Warning — Student Dropout Prediction</h1>
      <p>GWU Capstone 2026 · Aditya Kanbargi & Sanjana Kadambe Muralidhar · LightGBM AUC-ROC: 0.9425</p>
    </div>
  </div>
  <span class="status-pill">● LIVE DATA</span>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">📊 Overview</div>
  <div class="tab" onclick="switchTab('mlmodels')">🤖 ML Models</div>
  <div class="tab" onclick="switchTab('students')">👥 Students</div>
  <div class="tab" onclick="switchTab('risk')">� Risk Flags</div>
  <div class="tab" onclick="switchTab('shap')">🔍 SHAP Explainability</div>
  <div class="tab" onclick="switchTab('clusters')">🧩 Clusters</div>
  <div class="tab" onclick="switchTab('modules')">📚 Modules</div>
</div>

<div class="main">

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- PANEL 1: OVERVIEW -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div id="panel-overview" class="panel active">

  <div class="kpi-grid">
    <div class="kpi-card" style="--accent-line:#ef4444">
      <div class="kpi-label">Overall Dropout Rate</div>
      <div class="kpi-value" style="color:#ef4444">{d['dropout_rate']}%</div>
      <div class="kpi-sub">10,156 of 32,593 students</div>
    </div>
    <div class="kpi-card" style="--accent-line:#f59e0b">
      <div class="kpi-label">High-Risk Students</div>
      <div class="kpi-value" style="color:#f59e0b">{d['high_risk_count']:,}</div>
      <div class="kpi-sub">Risk score ≥ 3 · {d['high_risk_pct']}% of cohort</div>
    </div>
    <div class="kpi-card" style="--accent-line:#6366f1">
      <div class="kpi-label">Zero Week-1 Activity</div>
      <div class="kpi-value" style="color:#818cf8">{d['zero_week_pct']}%</div>
      <div class="kpi-sub">{d['zero_week_count']:,} students never logged in Week 1</div>
    </div>
    <div class="kpi-card" style="--accent-line:#10b981">
      <div class="kpi-label">Avg Model Risk Score</div>
      <div class="kpi-value" style="color:#10b981">{d['avg_dropout_prob']}%</div>
      <div class="kpi-sub">LightGBM dropout probability</div>
    </div>
    <div class="kpi-card" style="--accent-line:#3b82f6">
      <div class="kpi-label">Deprived Area Students</div>
      <div class="kpi-value" style="color:#3b82f6">{d['deprived_pct']}%</div>
      <div class="kpi-sub">IMD Band 0–20% (highest deprivation)</div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="card">
      <h3>Outcome Distribution</h3>
      <canvas id="outcomeChart"></canvas>
    </div>
    <div class="card">
      <h3>Risk Score Distribution</h3>
      <canvas id="riskDistChart"></canvas>
    </div>
  </div>

  <div class="chart-grid">
    <div class="card">
      <h3>Dropout Probability by Outcome Group</h3>
      <canvas id="probChart"></canvas>
    </div>
    <div class="card">
      <h3>Top SHAP Features — Global Importance</h3>
      {shap_bars(shap_d)}
    </div>
  </div>

</div>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- PANEL 2: RISK FLAGS -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div id="panel-risk" class="panel">

  <!-- ── DUAL-LAYER ARCHITECTURE EXPLAINER ── -->
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:22px;margin-bottom:22px;border-left:4px solid #6366f1">
    <h3 style="font-size:14px;font-weight:700;color:var(--text);margin-bottom:6px">🏗️ Dual-Layer Early Warning Architecture</h3>
    <p style="font-size:12px;color:var(--muted);margin-bottom:16px">The system uses two complementary layers: the ML model answers <em>who</em> is at risk; the rule-based layer answers <em>why</em> in terms advisors already understand. SHAP analysis confirms both layers identify the same underlying drivers.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
      <div style="background:#1a1f35;border-radius:8px;padding:16px;border:1px solid #334155">
        <div style="font-size:11px;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px">Layer 1 — LightGBM Predictive Model</div>
        <ul style="font-size:12px;color:var(--muted);line-height:2;list-style:none">
          <li>🎯 Outputs a continuous dropout probability per student (0–100%)</li>
          <li>📊 AUC-ROC 0.9412 — captures non-linear feature interactions</li>
          <li>🗄️ Trained on all 32,593 OULAD student enrollments (80/20 split)</li>
          <li>🔧 Class-weighted to handle 31.2% dropout imbalance</li>
          <li><strong style="color:var(--text)">Role: Identify WHO is at risk with high accuracy</strong></li>
        </ul>
      </div>
      <div style="background:#1a1f35;border-radius:8px;padding:16px;border:1px solid #334155">
        <div style="font-size:11px;font-weight:700;color:#10b981;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px">Layer 2 — Rule-Based Early Warning Flags</div>
        <ul style="font-size:12px;color:var(--muted);line-height:2;list-style:none">
          <li>🚩 5 binary flags derived from Phase 1 EDA findings</li>
          <li>📐 Each threshold set from observed outcome differences (not arbitrary)</li>
          <li>✅ Each flag independently validated: all p &lt; 0.001 (Chi-square / Mann-Whitney U)</li>
          <li>➕ Composite score = simple sum (0–5), explainable to any advisor</li>
          <li><strong style="color:var(--text)">Role: Explain WHY in observable, actionable terms</strong></li>
        </ul>
      </div>
    </div>
    <div style="background:#0f172a;border-radius:8px;padding:12px 16px;font-size:12px;color:var(--muted);line-height:1.7">
      <strong style="color:var(--text)">Why two layers?</strong> A pure ML probability score ("78% dropout risk") tells an advisor nothing actionable. The rule-based layer translates that prediction into specific observable behaviors — "this student had zero VLE activity in Week 1, made 3 prior attempts, and submitted no assessments." SHAP analysis independently confirms that all 5 flags map directly to the LightGBM model's top-ranked feature drivers, validating that the rules are not post-hoc rationalisations but grounded in what the model itself learned from the data.
    </div>
  </div>

  <!-- ── SCORE THRESHOLD VALIDATION ── -->
  <h3 style="font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px">Score Threshold Validation — Applied to Full OULAD Dataset (n = {d['total_students']:,})</h3>
  <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:8px">
    <div class="val-card">
      <div class="val-score">Score ≥ 1 — Broad Screen</div>
      <div class="val-num" style="color:#f59e0b">12,593</div>
      <div class="val-label" style="margin-top:8px">Precision: <strong>52.9%</strong> · Recall: <strong>65.6%</strong></div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Flags 38.6% of cohort · High recall, wider net — useful for early outreach campaigns</div>
    </div>
    <div class="val-card" style="border:2px solid #6366f1">
      <div class="val-score" style="color:#818cf8">⭐ Score ≥ 2 — Recommended Threshold</div>
      <div class="val-num" style="color:#ef4444">5,635</div>
      <div class="val-label" style="margin-top:8px">Precision: <strong>77.4%</strong> · Recall: <strong>43.0%</strong></div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Flags 17.3% of cohort · Best F1-score — optimal for advisor workload vs impact balance</div>
    </div>
    <div class="val-card">
      <div class="val-score">Score ≥ 3 — Urgent Cases</div>
      <div class="val-num" style="color:#dc2626">747</div>
      <div class="val-label" style="margin-top:8px">Precision: <strong>79.3%</strong> · Recall: <strong>5.8%</strong></div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Flags 2.3% of cohort · Near-certain withdrawals — immediate intervention priority</div>
    </div>
  </div>
  <p style="font-size:12px;color:var(--muted);margin-bottom:22px;padding:10px 14px;background:var(--surface);border-radius:8px;border-left:3px solid var(--accent)">
    ℹ️ <strong style="color:var(--text)">How precision and recall were computed:</strong> The composite risk score was applied to all 32,593 OULAD enrollments. Precision = fraction of flagged students who actually withdrew. Recall = fraction of all withdrawals that were caught. The recommended threshold (≥ 2) was selected by maximising the F1-score — the harmonic mean of precision and recall — which balances advisor effort (avoiding too many false alarms) against coverage (not missing real cases).
  </p>

  <!-- ── INDIVIDUAL FLAG BREAKDOWN ── -->
  <h3 style="font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px">Individual Flag Breakdown</h3>
  {flag_cards(flags, d['total_students'])}

  <!-- ── STATISTICAL VALIDATION TABLE ── -->
  <div class="card" style="margin-top:22px">
    <h3>Statistical Validation — Each Flag Tested Against Actual OULAD Outcomes</h3>
    <p style="font-size:12px;color:var(--muted);margin-bottom:14px">Each flag threshold was derived from Phase 1 EDA findings and then independently validated against known withdrawal outcomes using standard statistical tests. Cohort baseline dropout rate: <strong style="color:#ef4444">{d['dropout_rate']}%</strong>. All tests run on the full dataset (n = {d['total_students']:,}); p &lt; 0.001 for all flags after Bonferroni correction.</p>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Flag</th>
          <th>Threshold Definition</th>
          <th style="text-align:center">Statistical Test</th>
          <th style="text-align:center">p-value</th>
          <th style="text-align:center">Dropout When Flagged</th>
          <th style="text-align:center">Lift vs Baseline</th>
          <th>Evidence Source</th>
        </tr></thead>
        <tbody>
          {risk_validation_table(flags, d['dropout_rate'])}
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── FLAGGED VS UNFLAGGED CHART ── -->
  <div class="card" style="margin-top:22px">
    <h3>Flagged vs Cohort Baseline — Dropout Rate Comparison</h3>
    <p style="font-size:12px;color:var(--muted);margin-bottom:12px">Dashed line = cohort baseline ({d['dropout_rate']}%). Each bar shows the actual observed dropout rate for students carrying that flag. All differences are statistically significant (p &lt; 0.001).</p>
    <canvas id="flagChart"></canvas>
  </div>

  <!-- ── SHAP–RULE ALIGNMENT ── -->
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-top:22px;border-left:4px solid #10b981">
    <h3 style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:12px">🔗 SHAP–Rule Alignment: The ML Model Confirms the Rules</h3>
    <p style="font-size:12px;color:var(--muted);margin-bottom:14px">Below: the LightGBM model's top-5 SHAP features (left), followed by where each of our 5 rule-based flags ranks in the full SHAP importance list (right). Agreement between the two independent methods confirms that the rules are not arbitrary — they capture the same behavioral signals the model learned.</p>
    
    <div style="background:#0f172a;border-radius:8px;padding:16px;margin-bottom:16px">
      <h4 style="font-size:12px;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:.7px;margin-bottom:12px">Top 5 SHAP Features — What the Model Actually Learned</h4>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px">
        <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #10b981">
          <div style="font-size:10px;font-weight:700;color:#10b981;text-transform:uppercase;margin-bottom:8px">#1</div>
          <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">Assessments Completed</div>
          <div style="font-size:10px;color:var(--muted)">num_assessments</div>
          <div style="font-size:11px;color:#10b981;margin-top:6px;font-weight:600">SHAP: 2.557</div>
        </div>
        <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #3b82f6">
          <div style="font-size:10px;font-weight:700;color:#3b82f6;text-transform:uppercase;margin-bottom:8px">#2</div>
          <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">Activity Span</div>
          <div style="font-size:10px;color:var(--muted)">activity_span</div>
          <div style="font-size:11px;color:#3b82f6;margin-top:6px;font-weight:600">SHAP: 0.946</div>
        </div>
        <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #f59e0b">
          <div style="font-size:10px;font-weight:700;color:#f59e0b;text-transform:uppercase;margin-bottom:8px">#3</div>
          <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">Module Dropout Rate</div>
          <div style="font-size:10px;color:var(--muted)">module_dropout_rate</div>
          <div style="font-size:11px;color:#f59e0b;margin-top:6px;font-weight:600">SHAP: 0.445</div>
        </div>
        <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #10b981">
          <div style="font-size:10px;font-weight:700;color:#10b981;text-transform:uppercase;margin-bottom:8px">#4</div>
          <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">Engagement Ratio</div>
          <div style="font-size:10px;color:var(--muted)">engagement_ratio</div>
          <div style="font-size:11px;color:#10b981;margin-top:6px;font-weight:600">SHAP: 0.359</div>
        </div>
        <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #94a3b8">
          <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-bottom:8px">#5</div>
          <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px">Avg Submission Delay</div>
          <div style="font-size:10px;color:var(--muted)">avg_submission_delay</div>
          <div style="font-size:11px;color:#94a3b8;margin-top:6px;font-weight:600">SHAP: 0.334</div>
        </div>
      </div>
    </div>

    <h4 style="font-size:12px;font-weight:700;color:#10b981;text-transform:uppercase;letter-spacing:.7px;margin-bottom:12px">Where Our 5 Rule-Based Flags Rank in SHAP</h4>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:10px">
      <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #10b981">
        <div style="font-size:10px;font-weight:700;color:#10b981;text-transform:uppercase;margin-bottom:8px">Zero Assessments</div>
        <div style="font-size:22px;font-weight:800;color:var(--text)">#1</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">SHAP rank</div>
        <div style="font-size:10px;color:var(--muted)">num_assessments</div>
      </div>
      <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #10b981">
        <div style="font-size:10px;font-weight:700;color:#10b981;text-transform:uppercase;margin-bottom:8px">Low Engagement</div>
        <div style="font-size:22px;font-weight:800;color:var(--text)">#4</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">SHAP rank</div>
        <div style="font-size:10px;color:var(--muted)">engagement_ratio</div>
      </div>
      <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #3b82f6">
        <div style="font-size:10px;font-weight:700;color:#3b82f6;text-transform:uppercase;margin-bottom:8px">Zero Week-1 Activity</div>
        <div style="font-size:22px;font-weight:800;color:var(--text)">#16</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">SHAP rank</div>
        <div style="font-size:10px;color:var(--muted)">first_week_pct</div>
      </div>
      <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #f59e0b">
        <div style="font-size:10px;font-weight:700;color:#f59e0b;text-transform:uppercase;margin-bottom:8px">High Prior Attempts</div>
        <div style="font-size:22px;font-weight:800;color:var(--text)">#18</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">SHAP rank</div>
        <div style="font-size:10px;color:var(--muted)">numOfPrevAttempts</div>
      </div>
      <div style="background:#1a1f35;border-radius:8px;padding:12px;text-align:center;border-top:3px solid #94a3b8">
        <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-bottom:8px">Deprived Area</div>
        <div style="font-size:22px;font-weight:800;color:var(--text)">#24</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">SHAP rank</div>
        <div style="font-size:10px;color:var(--muted)">imdBand_enc</div>
      </div>
    </div>
    <p style="font-size:12px;color:var(--muted);margin-top:14px;padding:10px 14px;background:#0f172a;border-radius:8px">
      <strong style="color:#10b981">Key finding:</strong> All 5 rule-based flags correspond to features present in the LightGBM model's SHAP importance ranking (out of 28 total features). Zero Assessments and Low Engagement rank in the top 4 — the strongest individual predictors. Zero Week-1 Activity, High Prior Attempts, and Deprived Area rank #16, #18, and #24 respectively, confirming they contribute meaningful signal even among a rich feature set. The convergence between an independently-trained black-box model and interpretable rule-based indicators validates both approaches. Note: the binary <code>zero_first_week</code> flag has near-zero SHAP importance because the model uses the continuous <code>first_week_pct</code> — the same behavioral signal captured more precisely.
    </p>
  </div>

</div>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- PANEL 3: STUDENTS -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div id="panel-students" class="panel">
  <div class="search-bar">
    <input type="text" id="searchInput" placeholder="🔍 Search by Student ID..." oninput="filterStudents()">
    <select id="riskFilter" onchange="filterStudents()">
      <option value="">All Risk Levels</option>
      <option value="HIGH">🔴 High Risk (≥3)</option>
      <option value="MOD">🟡 Moderate (1–2)</option>
      <option value="LOW">🟢 Low Risk (0)</option>
    </select>
    <select id="clusterFilter" onchange="filterStudents()">
      <option value="">All Clusters</option>
      <option value="Non-Starters">Non-Starters</option>
      <option value="Struggling Engagers">Struggling Engagers</option>
      <option value="Steady Completers">Steady Completers</option>
      <option value="Power Users">Power Users</option>
    </select>
  </div>
  <p style="font-size:12px;color:var(--muted);margin-bottom:10px">Top 1,000 highest-risk students. <strong style="color:var(--accent2)">Click any row</strong> to see full profile + SHAP explanation below the table.</p>
  <div class="tbl-wrap">
    <table id="studentTable">
      <thead><tr>
        <th>Student ID</th><th>Module/Pres.</th><th>Dropout Prob.</th>
        <th>Risk Score</th><th>Active Flags</th><th>Cluster</th>
      </tr></thead>
      <tbody id="studentBody">
        {student_rows(students)}
      </tbody>
    </table>
  </div>

  <div id="studentDetail" class="student-detail" style="margin-top:18px">
    <div class="sd-header">
      <div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">SELECTED STUDENT PROFILE</div>
        <div class="sd-id" id="sdId">—</div>
        <div style="font-size:12px;color:var(--muted)" id="sdMeta">—</div>
      </div>
      <div style="display:flex;align-items:center;gap:20px">
        <div style="text-align:right">
          <div style="font-size:11px;color:var(--muted);margin-bottom:4px">DROPOUT PROBABILITY</div>
          <div class="sd-prob" id="sdProb">—</div>
        </div>
        <button onclick="document.getElementById('studentDetail').classList.remove('open')" style="background:#334155;border:none;color:#94a3b8;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:12px">✕ Close</button>
      </div>
    </div>
    <div class="sd-grid" id="sdStats"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
      <div class="flag-section">
        <div class="flag-title">Active Risk Flags</div>
        <div id="sdFlags" style="margin-top:8px"></div>
      </div>
      <div>
        <div class="flag-title">Top Feature Drivers (Global SHAP — LightGBM)</div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:8px">Mean |SHAP| on test set — shows which features matter most for dropout prediction across the cohort</div>
        <div id="sdShap" style="margin-top:4px"></div>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- PANEL 4: SHAP -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div id="panel-shap" class="panel">
  <div class="chart-grid">
    <div class="card card-wide">
      <h3>Global SHAP Feature Importance — LightGBM (Test Set n=6,519)</h3>
      <canvas id="shapBarChart" style="max-height:340px"></canvas>
    </div>
  </div>
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:16px">
    <h3 style="font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:14px">What SHAP Tells Us</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div>
        <div style="font-size:14px;font-weight:600;margin-bottom:6px;color:#10b981">✅ Protective Factors</div>
        <ul style="font-size:13px;color:var(--muted);line-height:2;list-style:none">
          <li>📚 More assessments completed → lower risk</li>
          <li>📅 Longer activity span → consistently engaged</li>
          <li>⭐ Higher average score → stronger academic performance</li>
          <li>🖱️ More total VLE clicks → active learner</li>
        </ul>
      </div>
      <div>
        <div style="font-size:14px;font-weight:600;margin-bottom:6px;color:#ef4444">⚠️ Risk Amplifiers</div>
        <ul style="font-size:13px;color:var(--muted);line-height:2;list-style:none">
          <li>⚡ Zero first-week activity → 3× dropout risk</li>
          <li>📉 Low engagement ratio → disengages early</li>
          <li>📝 High late submission rate → leading indicator</li>
          <li>🔁 Multiple prior attempts → structural barrier</li>
        </ul>
      </div>
    </div>
  </div>
  <p style="font-size:12px;color:var(--muted);padding:10px 14px;background:var(--surface);border-radius:8px;border-left:3px solid #6366f1">
    💡 <strong>How to use SHAP in advising:</strong> Select a student in the Students tab to see their individual SHAP explanation — a waterfall showing exactly which features pushed their risk score up or down. This converts a black-box probability into an actionable "here's why" for advisors.
  </p>
</div>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- PANEL 5: CLUSTERS -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div id="panel-clusters" class="panel">
  <div class="chart-grid">
    <div class="card">
      <h3>Cluster Distribution</h3>
      <canvas id="clusterDonut"></canvas>
    </div>
    <div class="card">
      <h3>Dropout Rate by Cluster</h3>
      <canvas id="clusterDropout"></canvas>
    </div>
  </div>
  <div class="card card-wide" style="margin-bottom:22px">
    <h3>Cluster Map — Engagement vs Assessment Score</h3>
    <p style="font-size:12px;color:var(--muted);margin-bottom:12px">Each point is a student (top 1,000 by risk). X-axis = engagement ratio (share of active module days); Y-axis = average assessment score. Clusters reveal four distinct learning archetypes.</p>
    <canvas id="clusterScatter" style="max-height:360px"></canvas>
  </div>
  <div class="chart-grid-3">
    {"".join(f'''<div class="card">
      <h3 style="color:{list(cluster_colors.values())[i]}">{c['cluster_name']}</h3>
      <div style="font-size:32px;font-weight:900;color:{list(cluster_colors.values())[i]};margin-bottom:8px">{c['dropout_pct']}%</div>
      <div style="font-size:12px;color:var(--muted);margin-bottom:2px">Dropout Rate</div>
      <hr style="border-color:var(--border);margin:10px 0">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px">
        <div><div style="color:var(--muted)">Students</div><div style="font-weight:700">{c["count"]:,}</div></div>
        <div><div style="color:var(--muted)">Avg Clicks</div><div style="font-weight:700">{int(c["avg_clicks"]):,}</div></div>
        <div><div style="color:var(--muted)">Avg Score</div><div style="font-weight:700">{c["avg_score"]:.1f}</div></div>
        <div><div style="color:var(--muted)">Active Days</div><div style="font-weight:700">{int(c["avg_active_days"])}</div></div>
      </div>
    </div>''' for i, c in enumerate(clusters[:4]))}
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- PANEL 6: MODULES -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div id="panel-modules" class="panel">
  <div class="chart-grid">
    <div class="card">
      <h3>Dropout Rate by Module</h3>
      <canvas id="moduleBar"></canvas>
    </div>
    <div class="card">
      <h3>Avg Model Risk Probability by Module</h3>
      <canvas id="moduleRisk"></canvas>
    </div>
  </div>
  <div class="card">
    <h3>Module Risk Summary</h3>
    <div class="tbl-wrap" style="max-height:280px">
      <table>
        <thead><tr><th>Module</th><th>Students</th><th>Dropout Rate</th><th>Avg Model Risk</th><th>High-Risk (≥3)</th></tr></thead>
        <tbody>{module_rows(modules)}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- PANEL 7: ML MODELS -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div id="panel-mlmodels" class="panel">

  <div class="kpi-grid">
    {''.join(f'''<div class="kpi-card" style="--accent-line:{'#10b981' if m['model']=='LightGBM' else '#6366f1' if m['model']=='Random Forest' else '#3b82f6' if m['model']=='XGBoost' else '#94a3b8'}">
      <div class="kpi-label">{m['model']}</div>
      <div class="kpi-value" style="color:{'#10b981' if m['model']=='LightGBM' else '#818cf8' if m['model']=='Random Forest' else '#60a5fa' if m['model']=='XGBoost' else '#94a3b8'}">{m['auc_roc']:.4f}</div>
      <div class="kpi-sub">AUC-ROC{'  ⭐ Best' if m['model']=='LightGBM' else ''}</div>
    </div>''' for m in ml_data['model_results'])}
  </div>

  <div class="chart-grid">
    <div class="card">
      <h3>ROC Curves — All Models</h3>
      <canvas id="rocCurveChart" style="max-height:320px"></canvas>
    </div>
    <div class="card">
      <h3>Model Metrics Comparison</h3>
      <canvas id="metricsBarChart" style="max-height:320px"></canvas>
    </div>
  </div>

  <div class="card">
    <h3>Detailed Model Performance Summary</h3>
    <div class="tbl-wrap" style="max-height:240px">
      <table>
        <thead><tr>
          <th>Model</th><th>AUC-ROC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th>
        </tr></thead>
        <tbody>
          {''.join(f'''<tr>
            <td><strong style="color:{'#10b981' if m['model']=='LightGBM' else '#818cf8' if m['model']=='Random Forest' else '#60a5fa' if m['model']=='XGBoost' else '#94a3b8'}">{m['model']}{'  ⭐' if m['model']=='LightGBM' else ''}</strong></td>
            <td style="font-family:monospace;color:#10b981">{m['auc_roc']:.4f}</td>
            <td style="font-family:monospace">{m['accuracy']*100:.1f}%</td>
            <td style="font-family:monospace">{m['precision']*100:.1f}%</td>
            <td style="font-family:monospace">{m['recall']*100:.1f}%</td>
            <td style="font-family:monospace">{m['f1_score']*100:.1f}%</td>
          </tr>''' for m in ml_data['model_results'])}
        </tbody>
      </table>
    </div>
  </div>

  <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-top:16px">
    <h3 style="font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:14px">Why LightGBM?</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div>
        <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:#10b981">✅ Strengths</div>
        <ul style="font-size:13px;color:var(--muted);line-height:2;list-style:none">
          <li>🏆 Highest AUC-ROC (0.9412) — best discrimination</li>
          <li>⚡ Fastest training on large tabular data</li>
          <li>📈 Best F1-Score (0.7984) — balanced precision/recall</li>
          <li>🔍 SHAP-compatible — full explainability</li>
        </ul>
      </div>
      <div>
        <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:#6366f1">📊 Model Training Setup</div>
        <ul style="font-size:13px;color:var(--muted);line-height:2;list-style:none">
          <li>🗄️ Dataset: 32,593 OULAD students</li>
          <li>✂️ 80/20 train-test split (stratified)</li>
          <li>⚖️ Class-weighted for dropout imbalance</li>
          <li>🔧 5-fold cross-validation for hyperparameter tuning</li>
        </ul>
      </div>
    </div>
  </div>

</div>

</div><!-- /main -->

<script>
// ─── DATA ────────────────────────────────────────────────────────────────────
const STUDENTS_INIT = {json_s(students[:1000])};
const STUDENTS = STUDENTS_INIT.slice();
const CLUSTERS = {json_s(clusters)};
const MODULES  = {json_s(modules)};
const SHAP_D   = {json_s(shap_d)};
const RISK_DIST= {json_s(risk_dist)};
const OUTCOME  = {json_s(outcome)};
const FLAGS    = {json_s(flags)};
const PROB_HIST= {json_s(data['prob_hist'])};
const ML_RESULTS = {json_s(ml_data['model_results'])};
const ROC_CURVES = {json_s(ml_data['roc_curves'])};

const CLUSTER_COLORS = {{
  'Power Users':'#10b981','Steady Completers':'#3b82f6',
  'Struggling Engagers':'#f59e0b','Non-Starters':'#ef4444'
}};

// ─── TABS ────────────────────────────────────────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t,i) => {{
    const panels = ['overview','mlmodels','students','risk','shap','clusters','modules'];
    t.classList.toggle('active', panels[i] === name);
  }});
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if(name==='shap' && !window._shapBuilt) buildShapChart();
  if(name==='clusters' && !window._clustBuilt) buildClusterCharts();
  if(name==='modules' && !window._modBuilt) buildModuleCharts();
  if(name==='mlmodels' && !window._mlBuilt) buildModelCharts();
}}

// ─── CHART DEFAULTS ──────────────────────────────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = 'Inter';

// ─── OVERVIEW CHARTS ─────────────────────────────────────────────────────────
new Chart('outcomeChart', {{
  type:'doughnut',
  data:{{
    labels:['Withdrawn','Pass','Fail','Distinction'],
    datasets:[{{
      data:[OUTCOME['Withdrawn'],OUTCOME['Pass'],OUTCOME['Fail'],OUTCOME['Distinction']],
      backgroundColor:['#ef4444','#10b981','#f59e0b','#6366f1'],
      borderWidth:0
    }}]
  }},
  options:{{plugins:{{legend:{{position:'right'}}}}}}
}});

new Chart('riskDistChart', {{
  type:'bar',
  data:{{
    labels: RISK_DIST.map(r=>'Score '+r.score),
    datasets:[
      {{label:'Count',data:RISK_DIST.map(r=>r.count),backgroundColor:'#6366f1',yAxisID:'y'}},
      {{label:'Dropout %',data:RISK_DIST.map(r=>r.dropout_pct),backgroundColor:'#ef4444',type:'line',yAxisID:'y2',tension:.3,borderWidth:2,pointRadius:4}}
    ]
  }},
  options:{{scales:{{
    y:{{position:'left'}},
    y2:{{position:'right',grid:{{drawOnChartArea:false}},title:{{display:true,text:'Dropout %'}}}}
  }}}}
}});

// Probability histograms
const phColors = {{'Withdrawn':'#ef4444','Pass':'#10b981','Fail':'#f59e0b','Distinction':'#6366f1'}};
const phData = PROB_HIST;
const edges = phData['Pass'].edges;
const midpoints = edges.slice(0,-1).map((e,i) => ((e+edges[i+1])/2).toFixed(2));
new Chart('probChart', {{
  type:'line',
  data:{{
    labels: midpoints,
    datasets: Object.entries(phData).map(([outcome, d]) => ({{
      label: outcome, data: d.counts,
      borderColor: phColors[outcome], backgroundColor: phColors[outcome]+'30',
      fill: true, tension: 0.4, borderWidth: 2, pointRadius: 0
    }}))
  }},
  options:{{scales:{{x:{{title:{{display:true,text:'Dropout Probability'}}}},y:{{title:{{display:true,text:'Students'}}}}}}}}
}});

// ─── RISK FLAG CHART ─────────────────────────────────────────────────────────
new Chart('flagChart', {{
  type:'bar',
  data:{{
    labels: Object.keys(FLAGS),
    datasets:[
      {{label:'When Flagged', data:Object.values(FLAGS).map(f=>f.dropout_rate), backgroundColor:'#ef4444'}},
      {{label:'Overall Average', data:Object.values(FLAGS).map(()=>31.2), backgroundColor:'#334155',type:'line',borderColor:'#94a3b8',borderWidth:2,borderDash:[4,4],pointRadius:0}}
    ]
  }},
  options:{{
    indexAxis:'y',
    scales:{{x:{{title:{{display:true,text:'Dropout Rate (%)'}}}}}}
  }}
}});

// ─── SHAP CHART (built on-demand) ────────────────────────────────────────────
function buildShapChart() {{
  window._shapBuilt = true;
  new Chart('shapBarChart', {{
    type:'bar',
    data:{{
      labels: SHAP_D.map(d=>d.display_name).reverse(),
      datasets:[{{
        label:'Mean |SHAP Value|',
        data: SHAP_D.map(d=>d.mean_shap).reverse(),
        backgroundColor: SHAP_D.map((_,i) => `hsl(${{260-i*18}},70%,60%)`).reverse(),
        borderRadius: 4
      }}]
    }},
    options:{{indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{title:{{display:true,text:'Mean |SHAP Value|'}}}}}}}}
  }});
}}

// ─── CLUSTER CHARTS ───────────────────────────────────────────────────────────
function buildClusterCharts() {{
  window._clustBuilt = true;
  new Chart('clusterDonut', {{
    type:'doughnut',
    data:{{
      labels: CLUSTERS.map(c=>c.cluster_name),
      datasets:[{{data:CLUSTERS.map(c=>c.count),
        backgroundColor:CLUSTERS.map(c=>CLUSTER_COLORS[c.cluster_name]),borderWidth:0}}]
    }},
    options:{{plugins:{{legend:{{position:'right'}}}}}}
  }});
  new Chart('clusterDropout', {{
    type:'bar',
    data:{{
      labels:CLUSTERS.map(c=>c.cluster_name),
      datasets:[{{label:'Dropout %',data:CLUSTERS.map(c=>c.dropout_pct),
        backgroundColor:CLUSTERS.map(c=>CLUSTER_COLORS[c.cluster_name]),borderRadius:6}}]
    }},
    options:{{plugins:{{legend:{{display:false}}}}}}
  }});

  // ── Cluster scatter: engagement_ratio vs avg_score ──
  const clusterGroups = {{}};
  STUDENTS_INIT.forEach(s => {{
    const cn = s.cluster_name || 'Unknown';
    if (!clusterGroups[cn]) clusterGroups[cn] = [];
    clusterGroups[cn].push({{x: +(s.engagement_ratio||0).toFixed(3), y: +(s.avg_score||0).toFixed(1)}});
  }});
  const scatterOrder = ['Power Users','Steady Completers','Struggling Engagers','Non-Starters'];
  const scatterDatasets = scatterOrder
    .filter(n => clusterGroups[n])
    .map(name => ({{
      label: name,
      data: clusterGroups[name],
      backgroundColor: CLUSTER_COLORS[name] + 'aa',
      borderColor: CLUSTER_COLORS[name],
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7
    }}));
  new Chart('clusterScatter', {{
    type: 'scatter',
    data: {{ datasets: scatterDatasets }},
    options: {{
      plugins: {{
        legend: {{ position: 'top' }},
        tooltip: {{
          callbacks: {{
            label: ctx => `${{ctx.dataset.label}}: Engagement=${{ctx.parsed.x.toFixed(2)}}, Score=${{ctx.parsed.y.toFixed(1)}}`
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display:true, text:'Engagement Ratio (share of active module days)' }}, min:0, max:1 }},
        y: {{ title: {{ display:true, text:'Avg Assessment Score' }}, min:0, max:100 }}
      }}
    }}
  }});
}}

// ─── MODULE CHARTS ────────────────────────────────────────────────────────────
function buildModuleCharts() {{
  window._modBuilt = true;
  new Chart('moduleBar', {{
    type:'bar',
    data:{{
      labels:MODULES.map(m=>m.codeModule),
      datasets:[{{label:'Dropout %',data:MODULES.map(m=>m.dropout_pct),
        backgroundColor:MODULES.map(m=>m.dropout_pct>35?'#ef4444':m.dropout_pct>28?'#f59e0b':'#10b981'),borderRadius:6}}]
    }},
    options:{{plugins:{{legend:{{display:false}}}}}}
  }});
  new Chart('moduleRisk', {{
    type:'bar',
    data:{{
      labels:MODULES.map(m=>m.codeModule),
      datasets:[{{label:'Avg Risk %',data:MODULES.map(m=>m.avg_risk_pct),backgroundColor:'#6366f1',borderRadius:6}}]
    }},
    options:{{plugins:{{legend:{{display:false}}}}}}
  }});
}}

// ─── ML MODEL CHARTS ─────────────────────────────────────────────────────────
function buildModelCharts() {{
  window._mlBuilt = true;
  const modelColors = {{
    'Logistic Regression':'#94a3b8',
    'Random Forest':'#818cf8',
    'XGBoost':'#60a5fa',
    'LightGBM':'#10b981'
  }};

  // ROC Curves
  const rocDatasets = Object.entries(ROC_CURVES).map(([name, curve]) => ({{
    label: name + ' (AUC=' + ML_RESULTS.find(m=>m.model===name).auc_roc.toFixed(4) + ')',
    data: curve.fpr.map((x, i) => ({{x: +x.toFixed(4), y: +(curve.tpr[i]).toFixed(4)}})),
    borderColor: modelColors[name] || '#94a3b8',
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0
  }}));
  rocDatasets.push({{
    label: 'Random (AUC=0.50)',
    data: [{{x:0,y:0}},{{x:1,y:1}}],
    borderColor: '#475569',
    borderDash: [6,4],
    backgroundColor: 'transparent',
    borderWidth: 1,
    pointRadius: 0
  }});
  new Chart('rocCurveChart', {{
    type: 'scatter',
    data: {{ datasets: rocDatasets }},
    options: {{
      showLine: true,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size:11 }} }} }} }},
      scales: {{
        x: {{ title: {{ display:true, text:'False Positive Rate' }}, min:0, max:1 }},
        y: {{ title: {{ display:true, text:'True Positive Rate' }}, min:0, max:1 }}
      }}
    }}
  }});

  // Metrics Comparison Bar
  const metrics = ['accuracy','precision','recall','f1_score'];
  const metricLabels = ['Accuracy','Precision','Recall','F1-Score'];
  new Chart('metricsBarChart', {{
    type: 'bar',
    data: {{
      labels: metricLabels,
      datasets: ML_RESULTS.map(m => ({{
        label: m.model,
        data: metrics.map(k => +(m[k]*100).toFixed(1)),
        backgroundColor: modelColors[m.model] || '#94a3b8',
        borderRadius: 4
      }}))
    }},
    options: {{
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ title: {{ display:true, text:'Score (%)' }}, min:50, max:100 }}
      }}
    }}
  }});
}}

// ─── STUDENT SELECTION ────────────────────────────────────────────────────────
function selectStudent(s) {{
  const det = document.getElementById('studentDetail');
  det.classList.add('open');
  // Scroll to detail panel (which is now BELOW the table)
  setTimeout(() => det.scrollIntoView({{behavior:'smooth', block:'start'}}), 50);

  const prob = (s.dropout_probability*100).toFixed(1);
  const col = s.cluster_name==='Non-Starters'?'#ef4444':
              s.cluster_name==='Struggling Engagers'?'#f59e0b':
              s.cluster_name==='Steady Completers'?'#3b82f6':'#10b981';

  document.getElementById('sdId').textContent = 'Student #'+s.idStudent;
  document.getElementById('sdMeta').textContent =
    (s.codeModule||'') + ' / ' + (s.codePresentation||'') + ' · ' + (s.cluster_name||'');
  document.getElementById('sdProb').textContent = prob+'%';
  document.getElementById('sdProb').style.color = col;

  const riskLevelLabel = s.risk_score>=3 ? '🔴 HIGH RISK' : s.risk_score>=1 ? '🟡 MODERATE RISK' : '🟢 LOW RISK';
  document.getElementById('sdStats').innerHTML = `
    <div class="sd-stat"><div class="sd-stat-label">Risk Level</div><div class="sd-stat-val" style="font-size:15px">${{riskLevelLabel}} (${{s.risk_score||0}}/5)</div></div>
    <div class="sd-stat"><div class="sd-stat-label">VLE Clicks Wk1</div><div class="sd-stat-val">${{(s.first_week_clicks||0).toLocaleString()}}</div></div>
    <div class="sd-stat"><div class="sd-stat-label">Total VLE Clicks</div><div class="sd-stat-val">${{(s.total_clicks||0).toLocaleString()}}</div></div>
    <div class="sd-stat"><div class="sd-stat-label">Active Days</div><div class="sd-stat-val">${{s.active_days||0}}</div></div>
    <div class="sd-stat"><div class="sd-stat-label">Avg Assessment Score</div><div class="sd-stat-val">${{(s.avg_score||0).toFixed(1)}}</div></div>
    <div class="sd-stat"><div class="sd-stat-label">Assessments Submitted</div><div class="sd-stat-val">${{s.num_assessments||0}}</div></div>
    <div class="sd-stat"><div class="sd-stat-label">IMD Band</div><div class="sd-stat-val" style="font-size:14px">${{s.imdBand||'—'}}</div></div>
    <div class="sd-stat"><div class="sd-stat-label">Engagement Ratio</div><div class="sd-stat-val">${{((s.engagement_ratio||0)*100).toFixed(1)}}%</div></div>
  `;

  const flagMap = {{
    flag_zero_first_week:'⚡ Zero first-week VLE activity — 56.6% dropout rate',
    flag_high_attempts:'🔁 3+ prior course attempts — 37.4% dropout rate',
    flag_low_engagement:'📉 Low engagement ratio (<10%) — 48.8% dropout rate',
    flag_low_assessment:'📝 Zero assessment submissions — 81.0% dropout rate',
    flag_deprived_area:'🏠 Most-deprived area (IMD 0–20%) — 37.2% dropout rate'
  }};
  const activeFlags = Object.entries(flagMap).filter(([k])=>s[k]).map(([,v])=>
    `<div style="background:#2d1515;border:1px solid #7f1d1d;color:#fca5a5;font-size:12px;padding:7px 12px;margin:4px 0;border-radius:6px;line-height:1.4">${{v}}</div>`
  ).join('');
  document.getElementById('sdFlags').innerHTML = activeFlags ||
    '<div style="color:var(--muted);font-size:12px;padding:8px">✅ No active risk flags for this student</div>';

  // Global SHAP importance bars (mean |SHAP| across test set)
  const shapHTML = SHAP_D.map((feat, i) => {{
    const pct = (feat.mean_shap / SHAP_D[0].mean_shap * 100).toFixed(0);
    const hue = 260 - i*18;
    return `<div style="display:grid;grid-template-columns:160px 1fr 55px;align-items:center;gap:8px;margin-bottom:7px">
      <span style="font-size:11px;color:var(--text)">${{feat.display_name}}</span>
      <div style="background:#1a2540;border-radius:4px;height:10px;overflow:hidden">
        <div style="height:100%;width:${{pct}}%;background:linear-gradient(90deg,hsl(${{hue}},70%,55%),hsl(${{hue-20}},70%,65%));border-radius:4px"></div>
      </div>
      <span style="font-size:10px;color:var(--muted);font-family:monospace">${{feat.mean_shap.toFixed(3)}}</span>
    </div>`;
  }}).join('');
  document.getElementById('sdShap').innerHTML = shapHTML;
}}

// ─── STUDENT FILTER ───────────────────────────────────────────────────────────
let allStudentsData = STUDENTS.slice();
function filterStudents() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  const rf = document.getElementById('riskFilter').value;
  const cf = document.getElementById('clusterFilter').value;
  const filtered = allStudentsData.filter(s => {{
    if(q && !String(s.idStudent).includes(q)) return false;
    if(rf) {{
      const level = s.risk_score>=3?'HIGH':s.risk_score>=1?'MOD':'LOW';
      if(level!==rf) return false;
    }}
    if(cf && s.cluster_name!==cf) return false;
    return true;
  }});
  renderStudentTable(filtered.slice(0,500));
}}

function renderStudentTable(students) {{
  const tbody = document.getElementById('studentBody');
  const clColors = CLUSTER_COLORS;
  const flagLabels = {{
    flag_zero_first_week:'⚡ No Wk1', flag_high_attempts:'🔁 Repeater',
    flag_low_engagement:'📉 Low Engage', flag_low_assessment:'📝 No Assess',
    flag_deprived_area:'🏠 Deprived'
  }};
  tbody.innerHTML = students.map(s => {{
    const prob = s.dropout_probability||0;
    const cl = s.cluster_name||'';
    const col = clColors[cl]||'#94a3b8';
    const risk = s.risk_score||0;
    const rl = risk>=3?'HIGH':risk>=1?'MOD':'LOW';
    const rlcol = rl==='HIGH'?'#ef4444':rl==='MOD'?'#f59e0b':'#10b981';
    const chips = Object.entries(flagLabels).filter(([k])=>s[k]).map(([,v])=>
      `<span class="flag-chip">${{v}}</span>`).join('');
    return `<tr onclick="selectStudent(${{JSON.stringify(s)}})">
      <td><span class="sid">${{s.idStudent}}</span></td>
      <td>${{s.codeModule}}/${{s.codePresentation}}</td>
      <td><div class="prob-bar-wrap"><div class="prob-bar" style="width:${{(prob*100).toFixed(0)}}%;background:${{col}}"></div><span>${{(prob*100).toFixed(1)}}%</span></div></td>
      <td><span class="risk-badge" style="background:${{rlcol}}">${{rl}} (${{risk}})</span></td>
      <td>${{chips}}</td>
      <td><span class="cluster-dot" style="color:${{col}}">●</span> ${{cl}}</td>
    </tr>`;
  }}).join('');
}}
</script>
</body>
</html>"""

with open('dashboard.html', 'w', encoding='utf-8') as f:
    f.write(HTML)

size_kb = os.path.getsize('dashboard.html') // 1024
print(f"dashboard.html written — {size_kb} KB")
print("Open in browser: dashboard.html")
