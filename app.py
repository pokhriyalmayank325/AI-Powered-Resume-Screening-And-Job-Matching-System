"""AI Resume Screener - Streamlit dashboard (dark UI).

Needs Streamlit >= 1.39 (uses keyed containers for card styling):  pip install -U streamlit
"""
import html

import streamlit as st

try:  # optional: loads ANTHROPIC_API_KEY from .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.extraction import extract_text
from src.gap_analysis import learning_plan
from src.info_extractor import EDU_LEVELS
from src.llm_explainer import explain_match, llm_available
from src.matcher import DEFAULT_WEIGHTS, compute_match

st.set_page_config(page_title="AI Resume Screener", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -8%, rgba(74,60,190,.32), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(30,70,170,.28), transparent 55%),
    #070b1a;
  color: #e8ecf8;
}
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1480px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1128 0%, #080d20 100%);
  border-right: 1px solid rgba(99,115,190,.2);
}
[data-testid="stSidebarUserContent"] { padding-top: 1.4rem; }
.brand { display:flex; gap:12px; align-items:center; margin-bottom:22px; }
.logo { width:46px; height:46px; border-radius:13px; display:grid; place-items:center; font-size:24px;
  background: linear-gradient(135deg, rgba(139,92,246,.45), rgba(59,130,246,.35)); border:1px solid rgba(139,120,255,.5); }
.brand-t { font-weight:700; font-size:1.05rem; color:#fff; line-height:1.2; }
.brand-s { font-size:.78rem; color:#93a0cf; }
.side-h { font-size:1.15rem; font-weight:600; color:#e8ecf8; margin:6px 0 6px; }
.side-sub { font-size:.98rem; font-weight:600; color:#e8ecf8; margin:14px 0 2px; }
.side-note { font-size:.78rem; color:#8a97c4; margin:6px 0 12px; }
.side-foot { margin-top:34px; font-size:.78rem; color:#7f8cb8; line-height:1.5; }
.side-foot b { color:#c3cbe8; font-weight:600; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] { color:#dbe2fb; font-weight:500; }
[data-testid="stSlider"] [role="slider"] { background:#8b5cf6; box-shadow:0 0 0 5px rgba(139,92,246,.22); }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"], [data-testid="stSlider"] [data-testid="stThumbValue"] { color:#dbe2fb; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color:#6f7ba6; }

/* ---------- header ---------- */
.hdr { display:flex; align-items:center; gap:18px; margin-bottom:22px; }
.hdr-ic { width:76px; height:76px; border-radius:20px; display:grid; place-items:center; font-size:38px;
  background: linear-gradient(145deg, rgba(139,92,246,.35), rgba(59,130,246,.25)); border:1px solid rgba(139,120,255,.5); }
.hdr-t { font-size:2.05rem; font-weight:800; color:#fff; letter-spacing:-.02em; line-height:1.15; }
.grad { background: linear-gradient(90deg,#4f9bff,#a06bff); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hdr-s { color:#a6b2db; margin-top:6px; font-size:.98rem; }
.ready { margin-left:auto; display:flex; align-items:center; gap:8px; padding:8px 16px; border-radius:999px;
  background: rgba(20,30,66,.8); border:1px solid rgba(99,115,190,.3); color:#dbe2fb; font-size:.88rem; font-weight:500; }
.dot { width:8px; height:8px; border-radius:50%; background:#22c55e; box-shadow:0 0 8px #22c55e; }

/* ---------- cards ---------- */
.card, div[class*="st-key-card_"] {
  background: linear-gradient(160deg, rgba(21,29,66,.88), rgba(11,17,40,.9));
  border: 1px solid rgba(99,115,190,.28); border-radius: 18px; padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
div[class*="st-key-card_in"] { border-color: rgba(96,125,255,.7); box-shadow: 0 0 0 1px rgba(96,125,255,.3), 0 0 30px rgba(80,100,255,.16); }
.card-h { display:flex; align-items:center; gap:12px; margin-bottom:14px; }
.card-ic { width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:20px;
  background: rgba(60,75,160,.35); border:1px solid rgba(120,135,230,.4); flex:none; }
.card-t { font-weight:600; color:#fff; font-size:1.02rem; }
.card-s { color:#93a0cf; font-size:.82rem; }
.tick { margin-left:auto; width:26px; height:26px; border-radius:50%; display:grid; place-items:center; font-size:14px;
  font-weight:700; background:#2f8cff; color:#fff; flex:none; }
.tick.off { background:transparent; border:2px solid rgba(120,135,200,.4); color:transparent; }
.sec-h { font-size:1.15rem; font-weight:700; color:#fff; margin:28px 0 12px; }

/* ---------- inputs ---------- */
.stTextArea textarea { background:#0b1230 !important; color:#e8ecf8 !important; border-radius:12px !important; }
[data-baseweb="textarea"], [data-baseweb="base-input"] { background:#0b1230 !important; border-radius:12px !important;
  border-color: rgba(99,115,190,.35) !important; }
[data-testid="stFileUploader"] section { background:#0b1230; border:1px dashed rgba(99,115,190,.5); border-radius:12px; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:#aab5df; }
[data-testid="stFileUploaderFile"] { background:#0b1230; border:1px solid rgba(99,115,190,.4); border-radius:12px; padding:6px 10px; }
.stButton > button { width:100%; height:52px; border-radius:12px; font-weight:600; font-size:1rem; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg,#7c5cff,#4f8cff); border:0; color:#fff; box-shadow:0 8px 26px rgba(100,110,255,.38); }
.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
  background:transparent; border:1px solid rgba(140,155,220,.35); color:#e8ecf8; }
[data-testid="stExpander"] { background:rgba(15,22,50,.6); border:1px solid rgba(99,115,190,.25); border-radius:14px; }

/* ---------- KPI tiles ---------- */
.kpi { background: linear-gradient(160deg, rgba(21,29,66,.88), rgba(11,17,40,.9)); border:1px solid rgba(99,115,190,.28);
  border-radius:16px; padding:16px 18px; min-height:130px; box-shadow:0 10px 30px rgba(0,0,0,.3); }
.kpi-top { display:flex; align-items:center; gap:14px; margin-bottom:16px; }
.kpi-ic { width:48px; height:48px; border-radius:50%; display:grid; place-items:center; font-size:22px; flex:none; }
.kpi-l { color:#cbd3ef; font-size:.88rem; }
.kpi-v { color:#fff; font-size:1.7rem; font-weight:700; line-height:1.15; }
.kpi-v span { color:#aab5df; font-size:1.25rem; font-weight:600; }
.track { height:6px; border-radius:99px; background:rgba(120,135,200,.18); overflow:hidden; }
.fill { height:100%; border-radius:99px; }

/* ---------- score card ---------- */
.donut-wrap { display:flex; justify-content:center; margin:14px 0 18px; }
.donut { width:214px; height:214px; border-radius:50%; display:grid; place-items:center;
  background: conic-gradient(from 0deg, #8b5cf6 0%, #3b82f6 calc(var(--p) * .55%), #22c55e calc(var(--p) * 1%),
    rgba(120,135,200,.16) calc(var(--p) * 1%) 100%); }
.donut-in { width:172px; height:172px; border-radius:50%; background:#0b1230; display:grid; place-content:center; text-align:center; }
.donut-big { font-size:2.5rem; font-weight:700; color:#fff; line-height:1; }
.donut-sm { font-size:.9rem; color:#93a0cf; margin-top:6px; }
.badge { display:flex; width:fit-content; margin:0 auto 12px; align-items:center; gap:7px; padding:7px 16px; border-radius:999px;
  font-weight:600; font-size:.92rem; }
.score-msg { text-align:center; color:#b4bfe6; font-size:.9rem; line-height:1.55; }
.method { text-align:center; color:#7f8cb8; font-size:.76rem; margin-top:12px; }

/* ---------- breakdown / job analysis ---------- */
.brow { display:flex; align-items:center; gap:14px; margin:14px 0; }
.bic { width:34px; height:34px; border-radius:50%; display:grid; place-items:center; font-size:16px; background:rgba(60,75,160,.3); flex:none; }
.bname { width:92px; color:#e8ecf8; font-size:.95rem; flex:none; }
.btrack { flex:1; height:10px; border-radius:99px; background:rgba(120,135,200,.16); overflow:hidden; }
.bfill { height:100%; border-radius:99px; }
.bval { width:46px; text-align:right; color:#e8ecf8; font-weight:600; font-size:.92rem; }
.jrow { display:flex; gap:12px; align-items:flex-start; margin:11px 0; color:#dbe2fb; font-size:.9rem; line-height:1.4; }
.ji { width:22px; height:22px; border-radius:50%; display:grid; place-items:center; font-size:12px; font-weight:700; flex:none; margin-top:1px; }
.ji.g { background:rgba(34,197,94,.2); color:#4ade80; } .ji.a { background:rgba(245,158,11,.2); color:#fbbf24; }
.ji.r { background:rgba(239,68,68,.2); color:#f87171; }
.jmeta { margin-top:14px; padding-top:12px; border-top:1px solid rgba(99,115,190,.2); color:#93a0cf; font-size:.82rem; }

/* ---------- tabs, chips, lists ---------- */
.stTabs [data-baseweb="tab-list"] { gap:6px; background:rgba(13,20,48,.75); border:1px solid rgba(99,115,190,.25);
  border-radius:12px; padding:4px; width:fit-content; }
.stTabs [data-baseweb="tab"] { height:38px; border-radius:9px; padding:0 18px; color:#9aa6d1; background:transparent; }
.stTabs [aria-selected="true"] { background:rgba(90,105,255,.25); color:#fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 16px; }
.chip { padding:5px 13px; border-radius:999px; font-size:.84rem; font-weight:500; border:1px solid; }
.chip.g { background:rgba(34,197,94,.14); border-color:rgba(34,197,94,.4); color:#86efac; }
.chip.r { background:rgba(239,68,68,.14); border-color:rgba(239,68,68,.4); color:#fca5a5; }
.chip.b { background:rgba(59,130,246,.16); border-color:rgba(59,130,246,.45); color:#93c5fd; }
.chip.n { background:rgba(148,163,184,.12); border-color:rgba(148,163,184,.3); color:#cbd5e1; }
.lbl { color:#dbe2fb; font-weight:600; font-size:.92rem; margin-top:4px; }
.none { color:#7f8cb8; font-size:.86rem; margin:6px 0 16px; }
.ev { padding:12px 14px; border-radius:12px; background:rgba(13,20,48,.7); border:1px solid rgba(99,115,190,.22); margin-bottom:10px; font-size:.9rem; line-height:1.5; }
.ev small { color:#93a0cf; }
.plan { display:flex; gap:14px; align-items:flex-start; padding:14px 16px; border-radius:12px; background:rgba(13,20,48,.7);
  border:1px solid rgba(99,115,190,.22); margin-bottom:10px; }
.plan b { color:#fff; text-transform:capitalize; }
.pill { padding:2px 10px; border-radius:99px; font-size:.74rem; font-weight:600; margin-left:8px; }
.pill.hi { background:rgba(239,68,68,.16); color:#fca5a5; } .pill.lo { background:rgba(148,163,184,.16); color:#cbd5e1; }
.kv { display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid rgba(99,115,190,.18); font-size:.92rem; }
.kv span:first-child { color:#93a0cf; } .kv span:last-child { color:#fff; font-weight:500; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def H(s: str) -> str:
    """Collapse multi-line HTML so Markdown never treats indentation as a code block."""
    return "".join(line.strip() for line in s.splitlines())


def esc(x) -> str:
    return html.escape(str(x))


def show(s: str):
    st.markdown(H(s), unsafe_allow_html=True)


def chips(items, kind):
    if not items:
        return "<div class='none'>None</div>"
    return "<div class='chips'>" + "".join(f"<span class='chip {kind}'>{esc(i)}</span>" for i in items) + "</div>"


def card_head(icon, title, subtitle, done):
    tick = "<div class='tick'>✓</div>" if done else "<div class='tick off'>✓</div>"
    return H(f"""
        <div class='card-h'><div class='card-ic'>{icon}</div>
        <div><div class='card-t'>{title}</div><div class='card-s'>{subtitle}</div></div>{tick}</div>""")


def kpi(icon, icon_bg, label, value_html, pct, bar):
    return H(f"""
        <div class='kpi'><div class='kpi-top'>
        <div class='kpi-ic' style='background:{icon_bg}'>{icon}</div>
        <div><div class='kpi-l'>{label}</div><div class='kpi-v'>{value_html}</div></div></div>
        <div class='track'><div class='fill' style='width:{max(0, min(pct, 1)) * 100:.0f}%;background:{bar}'></div></div></div>""")


def tier(score):
    if score >= 75:
        return ("Excellent Match", "rgba(34,197,94,.18)", "#4ade80",
                "This resume lines up strongly with the role. Review the few remaining gaps.")
    if score >= 55:
        return ("Good Match", "rgba(34,197,94,.16)", "#4ade80",
                "Your profile matches well with the job requirements. Focus on improving the missing skills to increase your chances!")
    if score >= 40:
        return ("Partial Match", "rgba(245,158,11,.16)", "#fbbf24",
                "Some requirements are covered, but there are clear gaps. Check the missing skills below.")
    return ("Low Match", "rgba(239,68,68,.16)", "#f87171",
            "Few requirements are covered. The role needs skills this resume doesn't show yet.")


COMP_META = {  # key: (icon, label, bar gradient)
    "skills": ("🧩", "Skills", "linear-gradient(90deg,#7c5cff,#a78bfa)"),
    "experience": ("💼", "Experience", "linear-gradient(90deg,#10b981,#34d399)"),
    "education": ("🎓", "Education", "linear-gradient(90deg,#f59e0b,#fbbf24)"),
    "semantic": ("💬", "Semantic", "linear-gradient(90deg,#3b82f6,#60a5fa)"),
}


@st.cache_data(show_spinner=False, max_entries=32)
def run_match(resume_text, jd_text, weights_tuple, use_emb):
    return compute_match(resume_text, jd_text, weights=dict(weights_tuple), use_embeddings=use_emb).to_dict()


def clear_all():
    st.session_state["uploader_key"] += 1
    st.session_state["jd_text"] = ""
    for k in ("resume_raw", "jd_raw", "llm_text"):
        st.session_state.pop(k, None)


st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("jd_text", "")
ukey = f"resume_{st.session_state['uploader_key']}"

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    show("""
        <div class='brand'><div class='logo'>🧠</div>
        <div><div class='brand-t'>AI Resume Screener</div><div class='brand-s'>Smart Hiring. Better Matches.</div></div></div>
        <div class='side-h'>Settings</div>""")
    use_emb = st.toggle("Use semantic embeddings", value=True,
                        help="Sentence-transformer model. Falls back to TF-IDF if it can't be loaded.")
    show("<div class='side-sub'>Score Weights</div>")
    weights = {k: st.slider(k.title(), 0.0, 1.0, float(v), 0.05, key=f"w_{k}") for k, v in DEFAULT_WEIGHTS.items()}
    show("<div class='side-note'>Weights are normalised automatically.<br>Components that don't apply are dropped.</div>")
    want_llm = st.toggle("AI explanation (optional)", value=False, disabled=not llm_available(),
                         help="Needs ANTHROPIC_API_KEY in your .env file.")
    show("<div class='side-foot'><b>Powered by AI</b><br>Smarter screening. Better careers.</div>")

if sum(weights.values()) == 0:
    st.sidebar.warning("All weights are zero, so the defaults are being used.")
    weights = dict(DEFAULT_WEIGHTS)

# ----------------------------------------------------------------------------
# Header + inputs
# ----------------------------------------------------------------------------
status = "Result ready" if st.session_state.get("resume_raw") else "Ready"
show(f"""
    <div class='hdr'><div class='hdr-ic'>📄</div>
    <div><div class='hdr-t'><span class='grad'>AI Resume Screening</span> &amp; Job Matching</div>
    <div class='hdr-s'>Upload a resume, paste a job description, and get a transparent match analysis.</div></div>
    <div class='ready'><span class='dot'></span>{status}</div></div>""")

col_resume, col_jd, col_actions = st.columns([5, 5, 2.3], gap="medium")

with col_resume:
    with st.container(key="card_in_resume"):
        has_resume = st.session_state.get(ukey) is not None
        show(card_head("📥", "Resume", "Upload your resume (PDF, DOCX or TXT)", has_resume))
        upload = st.file_uploader("Resume", type=["pdf", "docx", "txt"], key=ukey, label_visibility="collapsed")

with col_jd:
    with st.container(key="card_in_jd"):
        has_jd = bool(st.session_state.get("jd_text", "").strip())
        show(card_head("💼", "Job Description", "Paste the job description here", has_jd))
        jd_input = st.text_area("Job description", key="jd_text", height=170, max_chars=5000,
                                placeholder="Paste the job description here...", label_visibility="collapsed")

with col_actions:
    st.markdown("<div style='height:54px'></div>", unsafe_allow_html=True)
    analyze = st.button("✨  Analyze", type="primary")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.button("⟳  Clear All", on_click=clear_all)

if analyze:
    if not upload or not jd_input.strip():
        st.warning("Upload a resume and paste a job description, then press Analyze.")
    else:
        try:
            text = extract_text(upload)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the resume: {e}")
            text = ""
        if text and len(text.strip()) < 50:
            st.error("Very little text was found. If this is a scanned PDF, it needs OCR first.")
        elif text:
            st.session_state["resume_raw"] = text
            st.session_state["jd_raw"] = jd_input
            st.session_state.pop("llm_text", None)
            st.rerun()

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
resume_raw, jd_raw = st.session_state.get("resume_raw"), st.session_state.get("jd_raw")

if not (resume_raw and jd_raw):
    show("""
        <div class='card' style='margin-top:26px'>
        <div class='card-h'><div class='card-ic'>📊</div><div><div class='card-t'>Your analysis appears here</div>
        <div class='card-s'>Add a resume and a job description, then press Analyze.</div></div></div>
        <div class='score-msg' style='text-align:left'>You'll get a match score, matched and missing skills,
        the resume lines that back the score, and a learning plan for the gaps.<br>
        No files handy? Use <b>data/sample_resume.txt</b> and <b>data/sample_jd.txt</b>.</div></div>""")
    st.stop()

with st.spinner("Analysing..."):
    res = run_match(resume_raw, jd_raw, tuple(sorted(weights.items())), use_emb)

if use_emb and res["method"] == "tfidf":
    st.info("The embedding model couldn't be loaded, so TF-IDF similarity was used instead.")

score = res["overall"]
comps = res["components"]
matched_req, missing_req = res["matched_required"], res["missing_required"]
req_total = len(matched_req) + len(missing_req)
years = res["resume"]["experience_years"]
edu_level = res["resume"]["education_level"]
badge_text, badge_bg, badge_fg, message = tier(score)

left, right = st.columns([1.1, 3], gap="medium")

with left:
    show(f"""
        <div class='card' style='min-height:560px'>
        <div class='donut-wrap'><div class='donut' style='--p:{score:.1f}'>
        <div class='donut-in'><div class='donut-big'>{score:.1f}%</div><div class='donut-sm'>Match Score</div></div></div></div>
        <div class='badge' style='background:{badge_bg};color:{badge_fg}'>● {badge_text}</div>
        <div class='score-msg'>{esc(message)}</div>
        <div class='method'>Similarity method: {esc(res['method'])}</div></div>""")

with right:
    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        show(kpi("🛠️", "rgba(59,130,246,.22)", "Required Skills",
                 f"{len(matched_req)} <span>/ {req_total}</span>",
                 len(matched_req) / req_total if req_total else 0, "linear-gradient(90deg,#3b82f6,#60a5fa)"))
    with k2:
        show(kpi("⚠️", "rgba(239,68,68,.2)", "Missing", f"{len(missing_req)}",
                 len(missing_req) / req_total if req_total else 0, "linear-gradient(90deg,#ef4444,#f87171)"))
    with k3:
        show(kpi("💼", "rgba(16,185,129,.2)", "Experience", f"{years:g} yrs",
                 comps["experience"] if comps["experience"] is not None else 0, "linear-gradient(90deg,#10b981,#34d399)"))
    with k4:
        edu_pct = comps["education"] if comps["education"] is not None else EDU_LEVELS.get(edu_level, 0) / 4
        show(kpi("🎓", "rgba(139,92,246,.22)", "Education", esc(edu_level.title() if edu_level != "none" else "Not found"),
                 edu_pct, "linear-gradient(90deg,#7c5cff,#a78bfa)"))

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    b_col, j_col = st.columns([1.5, 1], gap="small")

    with b_col:
        rows = ""
        for key in ("skills", "experience", "education", "semantic"):
            val = comps.get(key)
            if val is None:
                continue
            icon, label, bar = COMP_META[key]
            rows += (f"<div class='brow'><div class='bic'>{icon}</div><div class='bname'>{label}</div>"
                     f"<div class='btrack'><div class='bfill' style='width:{val * 100:.0f}%;background:{bar}'></div></div>"
                     f"<div class='bval'>{val * 100:.0f}%</div></div>")
        show(f"<div class='card' style='min-height:395px'><div class='card-h'><div class='card-ic'>📊</div>"
             f"<div class='card-t'>Score Breakdown</div></div>{rows}</div>")

    with j_col:
        seen, lines = set(), []
        for jd_line, _cv, s in list(res["evidence"]) + list(res["weak_spots"]):
            if jd_line not in seen:
                seen.add(jd_line)
                lines.append((jd_line, s))
        lines = sorted(lines, key=lambda t: -t[1])[:6]
        items = ""
        for text, s in lines:
            cls, sym = ("g", "✓") if s >= 0.5 else ("a", "!") if s >= 0.25 else ("r", "✕")
            short = text if len(text) <= 100 else text[:97] + "..."
            items += f"<div class='jrow'><div class='ji {cls}'>{sym}</div><div>{esc(short)}</div></div>"
        need = []
        if res["jd"].get("required_years"):
            need.append(f"{res['jd']['required_years']:g}+ yrs experience")
        if res["jd"].get("education_level") not in (None, "none"):
            need.append(f"{res['jd']['education_level'].title()} degree")
        meta = f"<div class='jmeta'>Role asks for: {esc(' · '.join(need))}</div>" if need else ""
        show(f"<div class='card' style='min-height:395px'><div class='card-h'><div class='card-ic'>📋</div>"
             f"<div class='card-t'>Job Analysis</div></div>{items}{meta}</div>")

# ----------------------------------------------------------------------------
# Detailed analysis
# ----------------------------------------------------------------------------
show("<div class='sec-h'>Detailed Analysis</div>")
tab_skills, tab_why, tab_plan, tab_data = st.tabs(["Skills Match", "Why this score?", "Learning Plan", "Extracted Data"])

with tab_skills:
    a, b = st.columns(2, gap="medium")
    with a:
        show("<div class='lbl'>Matched required skills</div>" + chips(matched_req, "g")
             + "<div class='lbl'>Missing required skills</div>" + chips(missing_req, "r"))
    with b:
        show("<div class='lbl'>Preferred skills you have</div>" + chips(res["matched_preferred"], "b")
             + "<div class='lbl'>Preferred skills missing</div>" + chips(res["missing_preferred"], "n")
             + "<div class='lbl'>Other skills on the resume</div>" + chips(res["extra_skills"], "n"))

with tab_why:
    st.markdown("**Best matches between job requirements and resume lines**")
    for jd_line, cv_line, s in res["evidence"]:
        show(f"<div class='ev'><b>Job:</b> {esc(jd_line)}<br><b>Resume:</b> {esc(cv_line)}<br><small>Coverage {s:.0%}</small></div>")
    st.markdown("**Weakest coverage (possible gaps)**")
    for jd_line, _cv, s in res["weak_spots"]:
        show(f"<div class='ev'><b>Job:</b> {esc(jd_line)}<br><small>Closest resume line is only {s:.0%} similar</small></div>")
    st.caption("Weights used: " + ", ".join(f"{k} {v:.0%}" for k, v in res["weights_used"].items()))
    if want_llm:
        if st.button("Generate AI explanation"):
            with st.spinner("Writing explanation..."):
                try:
                    st.session_state["llm_text"] = explain_match(res)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"AI explanation unavailable: {e}")
        if st.session_state.get("llm_text"):
            st.info(st.session_state["llm_text"])

with tab_plan:
    plan = learning_plan(missing_req, res["missing_preferred"])
    if not plan:
        st.success("No skill gaps found for this job.")
    for p in plan:
        hi = p["priority"].startswith("High")
        show(f"<div class='plan'><div><b>{esc(p['skill'])}</b><span class='pill {'hi' if hi else 'lo'}'>{esc(p['priority'])}</span>"
             f"<div class='card-s' style='margin-top:6px;font-size:.9rem;color:#b4bfe6'>{esc(p['first_step'])}</div></div></div>")

with tab_data:
    info = res["resume"]
    rows = [("Name", info.get("name") or "Not found"), ("Email", info.get("email") or "Not found"),
            ("Phone", info.get("phone") or "Not found"),
            ("Education level", (info.get("education_level") or "none").title()),
            ("Experience", f"{info.get('experience_years', 0):g} yrs")]
    show("".join(f"<div class='kv'><span>{k}</span><span>{esc(v)}</span></div>" for k, v in rows)
         + "<div class='lbl' style='margin-top:16px'>Detected skills</div>" + chips(info.get("skills", []), "n"))

with st.expander("Fairness & limitations"):
    st.markdown(
        "- The score uses **skills, semantic similarity, experience and education only**. "
        "Name, email, phone, gender, age and photo are never used.\n"
        "- Skill detection relies on a curated list (`data/skills.json`), so unlisted skills are missed.\n"
        "- Scanned or image PDFs need OCR. Unusual layouts can hurt section detection.\n"
        "- This is a screening aid, not a hiring decision. A person should review every result."
    )
